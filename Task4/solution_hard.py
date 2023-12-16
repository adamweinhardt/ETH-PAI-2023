import json
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union

from torch.optim import Adam


from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_default_device('cpu')

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, output_activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.input_layer = nn.Linear(input_dim, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_dim)

        # Activation function
        if output_activation == 'relu':
            self.output_activation = F.relu
        elif output_activation == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'linear':
            self.output_activation = lambda x: x
        elif output_activation == 'negative_relu':
            self.output_activation = lambda x: F.relu(x)*-1
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'tanh', or 'sigmoid'.")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Implement the forward pass for the neural network you have defined.
        x = self.input_layer(s)
        for layer in self.hidden_layers:
            x = F.relu(layer(x), inplace=False)

        return self.output_layer(x)


class Actor():
    def __init__(self, hidden_size: int, hidden_layers_num: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers_num = hidden_layers_num
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        self.nn = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim*2,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers_num,
            output_activation='linear'
        )

        self.optimizer = Adam(
            params=list(self.nn.parameters()),
            lr=self.actor_lr
        )

        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py.

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        batch_preds = self.nn.forward(state)
        mean = torch.unsqueeze(batch_preds[:, 0], -1)
        log_std = torch.unsqueeze(batch_preds[:, 1], -1)
        log_std = self.clamp_log_std(log_std)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        sampled_point = normal.rsample()
        action = torch.tanh(sampled_point)
        log_prob = normal.log_prob(sampled_point)

        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(0, keepdim=True)
        return action, log_prob


class Critic:

    def __init__(self, hidden_size: int, hidden_layers_num: int, critic_lr: int, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers_num = hidden_layers_num
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.nn = NeuralNetwork(input_dim=self.state_dim + self.action_dim,
                                output_dim=1,
                                hidden_layers=self.hidden_layers_num,
                                hidden_size=self.hidden_size,
                                output_activation='linear'
                                )

        self.optimizer = Adam(params=self.nn.parameters(), lr=self.critic_lr)

    def pred_reward(self, state: np.array, action: np.array):
        inp = torch.concat([state, action], dim=-1)
        return self.nn.forward(inp)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    '''

    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.NAdam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes.
        # Feel free to instantiate any other parameters you feel you might need.

        self.logs = defaultdict(lambda: [])
        self.verbose = False

        self.DISCOUNT_FACTOR = 0.99
        self.ALPHA = TrainableParameter(init_param=0.01,lr_param=1e-3,train_param=True)
        self.alpha_optimizer = optim.NAdam([self.ALPHA.get_log_param()], lr=1e-3)

        self.TAU = 0.05
        self.iteration_idx = 0

        critic_params = {
            'hidden_size': 256,
            'hidden_layers_num': 2,
            'critic_lr': 1e-3,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        self.critic1 = Critic(
            **critic_params
        )
        self.critic1_target = Critic(**critic_params)
        self.critic1_target.nn.load_state_dict(self.critic1.nn.state_dict())

        self.critic2 = Critic(
            **critic_params
        )
        self.critic2_target = Critic(**critic_params)
        self.critic2_target.nn.load_state_dict(self.critic2.nn.state_dict())

        self.critic_optimizer = optim.NAdam(list(self.critic1.nn.parameters()) + list(self.critic2.nn.parameters()), lr=critic_params['critic_lr'])


        self.actor = Actor(
            hidden_size=256,
            hidden_layers_num=2,
            actor_lr=1e-3,
            action_dim=self.action_dim,
            state_dim=self.state_dim
        )

        self.policy_frequency = 2
        self.target_network_frequency = 1

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        state = torch.tensor(s)
        is_batch = state.shape != (3,)
        if not is_batch:
            state = torch.unsqueeze(state, 0)
        actions, _ = self.actor.get_action_and_log_prob(state, deterministic=(not train))

        action = actions.cpu().detach().numpy()
        if not is_batch:
            return action[0]
        #assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)
    def train_agent(self):

        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        with torch.no_grad():
            next_state_actions, next_state_log_ps = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=True)
            q1_target_predict = self.critic1_target.pred_reward(s_prime_batch, next_state_actions)
            q2_target_predict = self.critic2_target.pred_reward(s_prime_batch, next_state_actions)
            q_min = torch.min(q1_target_predict, q2_target_predict)
            entropy_penalty = self.ALPHA.get_param() * next_state_log_ps
            q_next_states = q_min - entropy_penalty
            q_min_value = r_batch.flatten() + (self.DISCOUNT_FACTOR * (q_next_states)).view(-1)

        q1_curr_state = self.critic1.pred_reward(s_batch, a_batch).view(-1)
        q1_loss = F.mse_loss(q1_curr_state, q_min_value)

        q2_curr_state = self.critic2.pred_reward(s_batch, a_batch).view(-1)
        q2_loss = F.mse_loss(q2_curr_state, q_min_value)

        mutual_loss = q1_loss + q2_loss
        self.logs['q_loss'].append(mutual_loss.item())

        # optimize the model
        self.critic_optimizer.zero_grad()
        mutual_loss.backward()
        self.critic_optimizer.step()

        if self.iteration_idx % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                actions, log_ps = self.actor.get_action_and_log_prob(s_batch, deterministic=True)
                q1_pred_value = self.critic1.pred_reward(s_batch, actions)
                q2_pred_value = self.critic2.pred_reward(s_batch, actions)
                min_q = torch.min(q1_pred_value, q2_pred_value)
                actor_loss = ((self.ALPHA.get_param() * log_ps) - min_q).mean()
                self.logs['actor_loss'].append(actor_loss.item())
                self.run_gradient_update_step(self.actor, actor_loss)

                if hasattr(self, 'ALPHA'):
                    with torch.no_grad():
                        _, log_ps = self.actor.get_action_and_log_prob(s_batch, False)
                    alpha_loss = (-self.ALPHA.get_log_param().exp() * (log_ps + (-self.state_dim))).mean()
                    self.logs['alpha_loss'].append(actor_loss.item())
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

        if self.iteration_idx % self.target_network_frequency == 0:
            self.critic_target_update(self.critic1.nn, self.critic1_target.nn, soft_update=True, tau=self.TAU)
            self.critic_target_update(self.critic2.nn, self.critic2_target.nn, soft_update=True, tau=self.TAU)

        self.iteration_idx += 1
# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
def plot_logs(log_dict):

    # Plotting
    plt.figure(figsize=(10, 6))
    for key, values in log_dict.items():
        plt.plot(values, label=key)

    plt.title("Losses Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs.png")
    plt.show()


if __name__ == '__main__':
    try:
        TRAIN_EPISODES = 50
        TEST_EPISODES = 300

        # You may set the save_video param to output the video of one of the evalution episodes, or
        # you can disable console printing during training and testing by setting verbose to False.
        save_video = True
        verbose = True

        agent = Agent()
        env = get_env(g=10.0, train=True)

        for EP in range(TRAIN_EPISODES):
            print("Training episode: "+str(EP) +'/'+str(TRAIN_EPISODES))
            run_episode(env, agent, None, verbose, train=True)

        if verbose:
            print('\n')

        test_returns = []
        env = get_env(g=10.0, train=False)

        if save_video:
            video_rec = VideoRecorder(env, "pendulum_episode.mp4")

        for EP in range(TEST_EPISODES):
            rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
            with torch.no_grad():
                episode_return = run_episode(env, agent, rec, verbose, train=False)
            test_returns.append(episode_return)

        avg_test_return = np.mean(np.array(test_returns))

        print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

        if save_video:
            video_rec.close()
    finally:
        with open('logs.json','w+') as f:
            json.dump(agent.logs,f)
        #plot_logs(agent.logs) #add import matplotlib.pyplot as plt for local debug