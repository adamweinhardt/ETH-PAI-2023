import torch
import torch.optim as optim
from torch.distributions import Normal
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
            self.output_activation = lambda x: x.clone()
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'tanh', or 'sigmoid'.")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Implement the forward pass for the neural network you have defined.
        x = self.input_layer(s)
        for layer in self.hidden_layers:
            x = F.relu(layer(x), inplace=False)

        output = self.output_activation(self.output_layer(x))

        return output

class Actor():
    def __init__(self,hidden_size: int, hidden_layers_num: int, actor_lr: float,
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
        self.network_backbone = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.hidden_size,
            hidden_size=self.hidden_size-1,
            hidden_layers=self.hidden_layers_num,
            output_activation='linear'
        )

        self.mean_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std_layer = nn.Linear(self.hidden_size, self.action_dim)

        self.optimizer = Adam(
            params=list(self.network_backbone.parameters())
                   +list(self.mean_layer.parameters())
                   +list(self.log_std_layer.parameters()),
            lr=0.003
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
        actor_output = self.network_backbone.forward(state)

        log_std = self.log_std_layer(actor_output)

        action = self.mean_layer(actor_output)
        action = F.tanh(action)

        log_prob = self.clamp_log_std(log_std)

        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        #TODO undesrtand xd
        # assert action.shape == (state.shape[0], self.action_dim) and \
        #    log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:


    def __init__(self, hidden_size: int, hidden_layers_num: int, critic_lr: int, state_dim: int = 3,
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
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
                                      output_activation='relu'
                                      )

        self.optimizer = Adam(params = self.nn.parameters(), lr=0.003)


    def pred_reward(self, state: np.array, action: np.array):
        inp = torch.concat([torch.tensor(state), torch.tensor(action)], dim=-1)
        return -1*self.nn.forward(inp)
#TODO
class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes.
        # Feel free to instantiate any other parameters you feel you might need.
        critic_params = {
            'hidden_size': 128,
            'hidden_layers_num': 4,
            'critic_lr': 0.005,
            'state_dim':self.state_dim,
            'action_dim': self.action_dim
        }
        self.critic1 = Critic(
            **critic_params
        )
        self.critic2 = Critic(
            **critic_params
        )
        self.actor = Actor(
            hidden_size=128,
            hidden_layers_num=4,
            actor_lr=0.005,
            action_dim=self.action_dim,
            state_dim=self.state_dim
        )


    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        mean, log_std = self.actor.get_action_and_log_prob(state=torch.Tensor(s), deterministic=False)
        std = torch.exp(log_std)

        action = np.random.normal(mean.detach(), std.detach(), (1,))
        action = np.clip(action,-1,1)


        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
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
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        # TODO: Implement Critic(s) update here.

        pred_action, pred_log_std = self.actor.get_action_and_log_prob(s_batch, deterministic=False)

        sampled_action = pred_action + torch.exp(pred_log_std) * torch.randn_like(pred_action)

        critic1_value = self.critic1.pred_reward(s_batch, sampled_action)
        critic2_value = self.critic2.pred_reward(s_batch, sampled_action)

        critic_value = torch.min(critic1_value, critic2_value)

        pred_next_state_action, _ = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)

        DISCOUNT_FACTOR = 0.99

        target_value = r_batch + DISCOUNT_FACTOR * torch.min(self.critic1.pred_reward(s_prime_batch, pred_next_state_action.detach()),
                                                        self.critic2.pred_reward(s_prime_batch, pred_next_state_action.detach()))

        critic_loss = nn.MSELoss()(critic_value, target_value)

        self.run_gradient_update_step(self.critic1,critic_loss)
        self.critic_target_update(self.critic1.nn, self.critic2.nn,tau=0.2, soft_update=True)

        entropy = 0.5 * torch.log(torch.abs(2 * torch.pi * torch.e * pred_log_std)).sum()
        actor_loss = torch.mean(entropy - critic_value.detach())

        # Backpropagation for the actor
        self.run_gradient_update_step(self.actor,actor_loss)



# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
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
