# --- num
import gpytorch
import numpy as np
import torch
import pandas as pd
from gpytorch import settings
from gpytorch.lazy import MatmulLazyTensor, delazify, lazify
from gpytorch.models.exact_prediction_strategies import (
    DefaultPredictionStrategy,
    clear_cache_hook,
)
from gpytorch.utils.memoize import cached
from scipy.stats.distributions import chi
# --- misc
import time 
import matplotlib.pyplot as plt
from matplotlib import rcParams
# --- ipywidgets
import ipywidgets
from ipywidgets import interact
import IPython
import warnings
warnings.filterwarnings('ignore')
# --- style
plt.style.use('../style/pai.mplstyle')


### --- Regression Util Funcs


def regression_function(x, noise=1e-1):
    """Get observation of function value."""
    return torch.sin(2 * x) / x + noise * torch.randn(len(x))


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel):
        super().__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, value):
        """Set output scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self.covar_module.outputscale = value
        
    @property
    def length_scale(self):
        """Get length scale."""
        ls = self.covar_module.base_kernel.kernels[0].lengthscale
        if ls is None:
            ls = torch.tensor(0.0)
        return ls 

    @length_scale.setter
    def length_scale(self, value):
        """Set length scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        
        try: 
            self.covar_module.lengthscale = value 
        except RuntimeError:
            pass 
        
        try:
            self.covar_module.base_kernel.lengthscale = value
        except RuntimeError:
            pass
    
        try:
            for kernel in self.covar_module.base_kernel.kernels:
                kernel.lengthscale = value 
        except RuntimeError:
            pass
    

def get_kernel(kernel, composition="addition"):
    base_kernel = []
    if "RBF" in kernel:
        base_kernel.append(gpytorch.kernels.RBFKernel())
    if "linear" in kernel:
        base_kernel.append(gpytorch.kernels.LinearKernel())
    if "quadratic" in kernel:
        base_kernel.append(gpytorch.kernels.PolynomialKernel(power=2))
    if "Matern-1/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=1/2))
    if "Matern-3/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=3/2))
    if "Matern-5/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=5/2))
    if "Cosine" in kernel:
        base_kernel.append(gpytorch.kernels.CosineKernel())

    if composition == "addition":
        base_kernel = gpytorch.kernels.AdditiveKernel(*base_kernel)
    elif composition == "product":
        base_kernel = gpytorch.kernels.ProductKernel(*base_kernel)
    else:
        raise NotImplementedError
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    return kernel 

def plot_model(model, train_x, train_y, test_x, inducing_points=None, plot_points=True, plot_samples=True):
    """Plots GP model w/ train data.
    """
    model.eval()
    with torch.no_grad():
        out = model(test_x) # Returns the GP latent function at test_x
        lower, upper = out.confidence_region()
        y_dist = model.likelihood(out)
        y_lower, y_upper = y_dist.confidence_region()
        
    if plot_points:
        plt.plot(train_x, train_y, '.', label='Train Data')
    
    plt.plot(test_x, out.mean, '-', color="#3497d9", lw=3, label='Posterior Mean')
    
    # plot func samples
    if plot_samples:
        plt.plot(test_x, out.sample(), '--', color="black", alpha=0.35, lw=1.2, label='GP Sample')
        for _ in range(3):
            plt.plot(test_x, out.sample(), '--', color="black", alpha=0.35, lw=1.2)
    
    # plot uncertainty
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), 
        color='#7ebffc', alpha=0.75, label='Epistemic Uncertainty', lw=1.2
    )
    plt.fill_between(test_x.numpy(), y_lower.numpy(), lower.numpy(), 
        color='#8ed685ed', alpha=0.5, label='Aleatoric Uncertainty', zorder=10)
    plt.fill_between(test_x.numpy(), upper.numpy(), y_upper.numpy(), 
        color='#8ed685ed', alpha=0.5, zorder=10)
    
    plt.ylim([-2, 3.])
    if inducing_points is not None:
        plt.plot(
            inducing_points,
            torch.zeros_like(inducing_points),
            'p', color="#de7065b2", markersize=6, zorder=50,
            label='inducing_points'
        )
    plt.xlim([min(test_x), max(test_x)])
        
              
def gp_regression(train_x, train_y, test_x, lengthscale, outputscale, noise, kernel, composition):
    kernel = get_kernel(kernel, composition)
    model = ExactGP(train_x, train_y, kernel)

    # Set hyper-parameters
    model.length_scale = lengthscale
    model.output_scale = outputscale
    model.likelihood.noise = torch.tensor([noise])
    
    return model 


### --- DEMO1


def gp_regression_(num_training, lengthscale, outputscale, noise, kernel, composition):
    train_x = (torch.rand(num_training) - 0.5) * 10 # range from -5 to 5
    train_y = regression_function(train_x)
    test_x = torch.linspace(-6, 6, 1000)
    
    model = gp_regression(train_x, train_y, test_x, lengthscale, outputscale, noise, kernel, composition)
    # Evaluate GP Model.
    plot_model(model, train_x, train_y, test_x)
    test_y = regression_function(test_x, noise=0).detach()
    plt.plot(test_x, test_y, '-', lw=2, color="k", label='Noise-free Function')
    plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=6)
    plt.show()


def plot_gp_regression():
    torch.manual_seed(0)
    kernels = ["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2", "Cosine"]
    composition = ["addition", "product"]

    interact(
        gp_regression_,
        num_training=ipywidgets.IntSlider(
            value=25, min=1, max=100, step=1, continuous_update=False),
        lengthscale=ipywidgets.FloatSlider(
            value=1., min=0.01, max=2, step=0.01, continuous_update=False),
        outputscale=ipywidgets.FloatSlider(
            value=1., min=0.01, max=5, step=0.01, continuous_update=False),
        noise=ipywidgets.FloatLogSlider(
            value=0.1, min=-3, max=2, continuous_update=False),
        kernel=ipywidgets.SelectMultiple(
            options=kernels,
            value=["RBF"],
            rows=len(kernels),
            disabled=False),
        composition=ipywidgets.Dropdown(
            options=composition,
            value=composition[0],
        )
    );
    
    
### --- DEMO2


def mll_kernel_(num_training):
    torch.manual_seed(0)
    train_x = (torch.rand(num_training) - 0.5) * 10
    train_y = regression_function(train_x)
    test_x = torch.linspace(-6, 6, 100)

    linestyles = ['-', ':', '-', ':', '-', ':', '-']
    linecolors = ["#440154", "#404387", "#29788E", "#22A784", "#79D151", "#B0E74B", "#FDE724"]
    kernels = ["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2", "Cosine"]
    best_kernel = {}
    plt.figure(figsize=(14.5,6))
    for i, kernel in enumerate(kernels):
        model = ExactGP(train_x, train_y, get_kernel(kernel))
        model.train()
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model) #  marginal log likelihood
        training_iter = 100

        losses = []
        lengthscale = []
        outputscale = []
        noise = []

        for _ in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            losses.append(loss.item())
            lengthscale.append(model.length_scale.item())
            outputscale.append(model.output_scale.item())
            noise.append(model.likelihood.noise.item())
            optimizer.step()
        plt.plot(losses, label=f"{kernel}", linestyle=linestyles[i], color=linecolors[i], linewidth=3)
        best_kernel[kernel] = (loss.item(), model)
    plt.xlim([0, 99])
    plt.legend(bbox_to_anchor=(0.5, -0.125), loc='upper center', ncol=7)
    plt.xlabel("Num Iteration")
    plt.ylabel("MLL Loss")

    plt.show()

    plt.figure() 
    best_model = min(best_kernel.values(), key=lambda x: x[0])[1] # added back
    plot_model(best_model, train_x, train_y, test_x)
    test_y = regression_function(test_x, noise=0).detach()
    plt.plot(test_x, test_y,'k-', label='Noise-free Function', lw=2)
    plt.xlabel("$\mathcal{X}$")
    plt.ylabel("$f$")
    plt.legend(bbox_to_anchor=(0.5, -0.075), loc='upper center', ncol=6)

    plt.show()

    
def plot_mll_kernel():
    torch.manual_seed(0)
    kernels = ["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2", "Cosine"]
    composition = ["addition", "product"]

    interact(
        mll_kernel_,
        num_training=ipywidgets.IntSlider(
            value=25, min=1, max=100, step=1, continuous_update=True)
    );
    
    
### --- DEMO3

def gp_weather(period, kernel, composition):
    df = pd.read_csv("../data/GlobalTemperatures.csv")
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    year = []
    month = []
    dt = pd.to_datetime(df.dt)
    for i in range(len(dt)):
        month.append(dt[i].month)
        year.append(dt[i].year)
    df["year"] = year 
    df["month"] = month
     
    if period == "yearly":
        idx = df["month"] == 7 # only use data in July every year
        x0 = df["year"][idx]
        xlabel = 'Year'
    else:
        idx = df["year"] > 2010
        x0 = df["month"][idx] + 12 * (df["year"][idx] - 2010)
        xlabel = 'Month'

    y = df["LandAverageTemperature"][idx].values

    x = np.arange(len(y))

    x_min, x_max = x.min(), x.max()
    y_mean, y_std = y.mean(), y.std()

    train_x = 6 * (torch.tensor((x - x_min) / (x_max - x_min)) - 0.5).to(torch.float)
    reverse_x = lambda x: (x / 6 + 0.5) * (x_max - x_min) + x_min
    train_y = torch.tensor( (y - y_mean) / y_std).to(torch.float)
    reverse_y = lambda y: y * y_std + y_mean
    test_x = torch.linspace(-3, 5, 1000)

    model = ExactGP(train_x, train_y, get_kernel(kernel))
    
    model.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    training_iter = 100
    for _ in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(test_x)
        lower, upper = out.confidence_region()
        y_dist = model.likelihood(out)
        y_lower, y_upper = y_dist.confidence_region()
    plt.plot(reverse_x(train_x) , reverse_y(train_y), 'k.', label='Train Data')
    plt.plot(reverse_x(test_x), reverse_y(out.mean), '-', color="#3497d9", lw=3.5, label='Posterior Mean')
    plt.plot(reverse_x(test_x), reverse_y(out.sample()), '--', color="black", label='GP Sample', alpha=0.35)
    for _ in range(3):
        plt.plot(reverse_x(test_x), reverse_y(out.sample()), '--', color="black", alpha=0.35, lw=1.2)
    plt.fill_between(reverse_x(test_x).numpy(), reverse_y(lower.numpy()), reverse_y(upper.numpy()), 
                     color='#7ebffc', alpha=0.75, label='Epistemic Uncertainty', lw=1.2)
    plt.fill_between(reverse_x(test_x).numpy(), reverse_y(y_lower.numpy()), reverse_y(lower.numpy()), 
                     color='#8ed685', alpha=0.5, label='Aleatoric Uncertainty')
    plt.fill_between(reverse_x(test_x).numpy(), reverse_y(upper.numpy()), reverse_y(y_upper.numpy()), 
                     color='#8ed685', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(0.5, -0.095), loc='upper center', ncol=6)
    plt.xlim([0, max(reverse_x(test_x))])
    plt.xlabel(xlabel)
    plt.ylabel("Global Temperature")
    plt.show()


def plot_gp_weather():
    interact(
        gp_weather,
        period=ipywidgets.Dropdown(
            options=['yearly', 'monthly after 2010']
        ),
        kernel=ipywidgets.SelectMultiple(
            options=["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2", "Cosine"],
            value=["RBF"],
            rows=7,
            disabled=False),
        composition=ipywidgets.Dropdown(
            options=["addition", "product"],
            value="addition",
        )
    );
    
    
### --- DEMO4

from gpytorch import settings
from gpytorch.lazy import MatmulLazyTensor, delazify, lazify
from gpytorch.models.exact_prediction_strategies import (
    DefaultPredictionStrategy,
    clear_cache_hook,
)
from gpytorch.utils.memoize import cached
from scipy.stats.distributions import chi


class SparsePredictionStrategy(DefaultPredictionStrategy):
    """Prediction strategy for Sparse GPs."""

    def __init__(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
        k_uu,
        root=None,
        inv_root=None,
    ):
        super().__init__(
            train_inputs,
            train_prior_dist,
            train_labels,
            likelihood,
            root=root,
            inv_root=inv_root,
        )
        self.k_uu = k_uu
        self.lik_train_train_covar = train_prior_dist.lazy_covariance_matrix

    @property  # type: ignore
    @cached(name="k_uu_inv_root")
    def k_uu_inv_root(self):
        """Get K_uu^-1/2."""
        train_train_covar = self.k_uu
        train_train_covar_inv_root = delazify(
            train_train_covar.root_inv_decomposition().root
        )
        return train_train_covar_inv_root

    @property  # type: ignore
    @cached(name="mean_cache")
    def mean_cache(self):
        r"""Get mean cache, namely \sigma^-1 k_uf y_f."""
        sigma = self.lik_train_train_covar
        sigma_inv_root = delazify(sigma.root_inv_decomposition().root)
        sigma_inv = sigma_inv_root @ sigma_inv_root.transpose(-2, -1)
        mean_cache = (sigma_inv @ self.train_labels.unsqueeze(-1)).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache


class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, kernel, inducing_points, approximation="DTC"):
        super().__init__(train_x, train_y, kernel)
        self.prediction_strategy = None
        self.xu = inducing_points
        self.approximation = approximation

    def __call__(self, x):
        """Return GP posterior at location `x'."""
        train_inputs = self.xu
        m = train_inputs.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        inputs = x

        if self.prediction_strategy is None:
            x_uf = torch.cat((train_inputs, self.train_inputs[0]), dim=0)
            output = self.forward(x_uf)
            mu_uf, kernel = output.mean, output.lazy_covariance_matrix

            mu_u, mu_f = mu_uf[:m], mu_uf[m:]
            k_uu, k_ff, k_uf = kernel[:m, :m], kernel[m:, m:], kernel[:m, m:]

            if self.approximation == "FITC":
                k_uu_root_inv = k_uu.root_inv_decomposition().root
                z = k_uf.transpose(-2, -1) @ k_uu_root_inv
                q_ff = z @ z.transpose(-2, -1)

                diag = delazify(k_ff - q_ff).diag() + self.likelihood.noise
                diag = lazify(torch.diag(1 / diag))

            elif self.approximation == "SOR" or self.approximation == "DTC":
                diag = lazify(torch.eye(len(self.train_targets))).mul(
                    1.0 / self.likelihood.noise
                )
            else:
                raise NotImplementedError(
                    f"{self.approximation} Not implemented.")

            cov = k_uu + (k_uf @ diag) @ k_uf.transpose(-2, -1)

            prior_dist = gpytorch.distributions.MultivariateNormal(
                mu_u, cov.add_jitter(1e-1)
            )

            # Create the prediction strategy for
            self.prediction_strategy = SparsePredictionStrategy(
                train_inputs=train_inputs,
                train_prior_dist=prior_dist,
                train_labels=(k_uf @ diag) @ (self.train_targets - mu_f),
                likelihood=self.likelihood,
                k_uu=k_uu.add_jitter(1e-1),
            )

        # Concatenate the input to the training input
        batch_shape = inputs.shape[:-2]
        # Make sure the batch shapes agree for training/test data
        if batch_shape != train_inputs.shape[:-2]:
            train_inputs = train_inputs.expand(
                *batch_shape, *train_inputs.shape[-2:])
        full_inputs = torch.cat([train_inputs, inputs], dim=-2)

        # Get the joint distribution for training/test data
        joint_output = self.forward(full_inputs)
        joint_mean, joint_covar = joint_output.loc, joint_output.lazy_covariance_matrix

        # Separate components.
        mu_s = joint_mean[..., m:]
        k_su, k_ss = joint_covar[..., m:, :m], joint_covar[..., m:, m:]
        
        with gpytorch.settings.cholesky_jitter(1e-1):
            pred_mean = mu_s + k_su @ self.prediction_strategy.mean_cache

        sig_inv_root = self.prediction_strategy.covar_cache
        k_su_sig_inv_root = k_su @ sig_inv_root
        rhs = MatmulLazyTensor(
            k_su_sig_inv_root, k_su_sig_inv_root.transpose(-2, -1))

        kuu_inv_root = self.prediction_strategy.k_uu_inv_root
        k_su_kuu_inv_root = k_su @ kuu_inv_root
        q_ss = MatmulLazyTensor(
            k_su_kuu_inv_root, k_su_kuu_inv_root.transpose(-2, -1))

        if self.approximation == "DTC" or self.approximation == "FITC":
            pred_cov = k_ss - q_ss + rhs
        elif self.approximation == "SOR":
            pred_cov = rhs
        else:
            raise NotImplementedError(f"{self.approximation} Not implemented.")

        return joint_output.__class__(pred_mean, pred_cov)
    

def sparse_gp_regression(num_training, num_inducing_points, approximation, method, plot_train_data, plot_samples):
    torch.manual_seed(0)
    np.random.seed(0)
    train_x = (torch.rand(num_training) - 0.5) * 10
    train_y = regression_function(train_x)
    test_x = torch.linspace(-6, 6, 100)
    test_y = regression_function(test_x, noise=0).detach()
    
    # Subsample inducing points
    start = time.time()
    if approximation == "ExactGP":
        model = ExactGP(train_x, train_y, get_kernel("RBF"))
        inducing_points = None 
    else:
        if method == "uniform":
            inducing_points = torch.linspace(-6, 6, num_inducing_points).unsqueeze(-1)
        elif method == "random":
            inducing_points = ((torch.rand(num_inducing_points) - 0.5) * 10).unsqueeze(-1)
        model = SparseGP(train_x, train_y, get_kernel("RBF"), inducing_points, approximation)

    model.length_scale = 1.2
    model.output_scale = 0.9 
    model.likelihood.noise = torch.tensor([0.05])
        
    plot_model(model, train_x, train_y, test_x, inducing_points, plot_points=plot_train_data, plot_samples=plot_samples)
    plt.title(f"{approximation} Inference Time: {time.time() - start} s")
    plt.plot(test_x, test_y,'k-', label='Noise-free Function', lw=2)
    plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=6)
    plt.show()


def plot_sparse_gp():
    interact(
        sparse_gp_regression,
        num_training=ipywidgets.IntSlider(
            value=500, min=100, step=100, max=10000, continuous_update=False, 
            description='Number training points:', style={'description_width': 'initial'}
        ),
        num_inducing_points=ipywidgets.IntSlider(
            value=5, min=1, max=20, continuous_update=False, 
            description='Number inducing points:', style={'description_width': 'initial'}),
        approximation=ipywidgets.Dropdown(
            options=["ExactGP", "DTC", "FITC", "SOR"], 
            value="DTC",
            description='Sparse Approximation:', style={'description_width': 'initial'}),
        method=["uniform", "random"],
        plot_train_data=ipywidgets.Checkbox(
            value=True, description='plot train data',
        ),
        plot_samples=ipywidgets.Checkbox(
            value=False, description='plot function samples',
        ),

    );


### --- DEMO5


class RandomFeatureGP(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        kernel,
        num_features,
        approximation="RFF",
    ):
        super().__init__(train_x, train_y, kernel)
        self.num_features = num_features
        self.approximation = approximation

        self.dim = train_x.shape[-1]
        self.w, self.b, self._feature_scale = self._sample_features()
        self.full_predictive_covariance = True

    @property
    def scale(self):
        """Return feature scale."""
        return torch.sqrt(self._feature_scale * self.output_scale)

    def sample_features(self):
        """Sample a new set of features."""
        self.w, self.b, self._feature_scale = self._sample_features()

    def _sample_features(self):
        """Sample a new set of random features."""
        # Only squared-exponential kernels are implemented.
        if self.approximation == "RFF":
            w = torch.randn(self.num_features, self.dim) / \
                torch.sqrt(self.length_scale)
            scale = torch.tensor(1.0 / self.num_features)

        elif self.approximation == "OFF":
            q, _ = torch.qr(torch.randn(self.num_features, self.dim))
            diag = torch.diag(
                torch.tensor(
                    chi.rvs(df=self.num_features, size=self.num_features),
                    dtype=torch.get_default_dtype(),
                )
            )
            w = (diag @ q) / torch.sqrt(self.length_scale)
            scale = torch.tensor(1.0 / self.num_features)

        elif self.approximation == "QFF":
            q = int(np.floor(np.power(self.num_features, 1.0 / self.dim)))
            self._num_features = q ** self.dim
            omegas, weights = np.polynomial.hermite.hermgauss(2 * q)
            omegas = torch.tensor(omegas[:q], dtype=torch.get_default_dtype())
            weights = torch.tensor(
                weights[:q], dtype=torch.get_default_dtype())

            omegas = torch.sqrt(1.0 / self.length_scale) * omegas
            w = torch.cartesian_prod(*[omegas.squeeze()
                                       for _ in range(self.dim)])
            if self.dim == 1:
                w = w.unsqueeze(-1)

            weights = 4 * weights / np.sqrt(np.pi)
            scale = torch.cartesian_prod(*[weights for _ in range(self.dim)])
            if self.dim > 1:
                scale = scale.prod(dim=1)
        else:
            raise NotImplementedError(f"{self.approximation} not implemented.")

        b = 2 * torch.tensor(np.pi) * torch.rand(self.num_features)
        self.prediction_strategy = None  # reset prediction strategy.
        return w, b, scale

    def __call__(self, x):
        """Return GP posterior at location `x'."""
        train_inputs = torch.zeros(2 * self.num_features, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        inputs = x

        if self.prediction_strategy is None:
            x = self.train_inputs[0]
            zt = self.forward(x).transpose(-2, -1)

            mean = train_inputs.squeeze(-1)

            cov = lazify(zt @ zt.transpose(-1, -2)).add_jitter(1e-1)

            y = self.train_targets - self.mean_module(x)
            labels = zt @ y

            prior_dist = gpytorch.distributions.MultivariateNormal(mean, cov)
            self.prediction_strategy = DefaultPredictionStrategy(
                train_inputs=train_inputs,
                train_prior_dist=prior_dist,
                train_labels=labels,
                likelihood=self.likelihood,
            )
        #
        z = self.forward(inputs)
        pred_mean = self.mean_module(
            inputs) + z @ self.prediction_strategy.mean_cache

        if self.full_predictive_covariance:
            precomputed_cache = self.prediction_strategy.covar_cache
            covar_inv_quad_form_root = z @ precomputed_cache

            pred_cov = (
                MatmulLazyTensor(
                    covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(
                        -1, -2)
                )
                .mul(self.likelihood.noise)
                .add_jitter()
            )
        else:
            dim = pred_mean.shape[-1]
            pred_cov = 1e-6 * torch.eye(dim)

        return gpytorch.distributions.MultivariateNormal(pred_mean, pred_cov)

    def forward(self, x):
        """Compute features at location x."""
        z = x @ self.w.transpose(-2, -1) + self.b
        return torch.cat([self.scale * torch.cos(z), self.scale * torch.sin(z)], dim=-1)
    
    
def rff_gp_regression(num_training, num_features, approximation, plot_train_data, plot_samples):
    torch.manual_seed(0)
    np.random.seed(0)
    train_x = (torch.rand(num_training) - 0.5) * 10
    train_y = regression_function(train_x)
    test_x = torch.linspace(-6, 6, 100)
    test_y = regression_function(test_x, noise=0).detach()

    def sample_features():
        IPython.display.clear_output(wait=True)
        plt.close()
        
        start = time.time()
        if approximation == "ExactGP":
            model = ExactGP(train_x, train_y, get_kernel("RBF"))
        else:
            model = RandomFeatureGP(train_x.unsqueeze(-1), 
                            train_y, get_kernel("RBF"), num_features, approximation)
        model.length_scale = 1.2
        model.output_scale = 0.9 
        model.likelihood.noise = torch.tensor([0.01])
        try:
            model.sample_features()
        except AttributeError:
            pass 
        
        plot_model(model, train_x, train_y, test_x, plot_points=plot_train_data, plot_samples=plot_samples)
        plt.plot(test_x, test_y,'k-', label='Noise-free Function', lw=2, zorder=50)
        plt.title(f"{approximation} Inference Time: {time.time() - start} s")
        plt.legend(bbox_to_anchor=(0.5, -0.095), loc='upper center', ncol=6)
        plt.show()

    sample_features()

    
def plot_rff_gp_regression():
    interact(
        rff_gp_regression,
        num_training=ipywidgets.IntSlider(
            value=500, min=100, step=100, max=10000, continuous_update=False, 
            description='Number training points:', style={'description_width': 'initial'}
        ),
        num_features=ipywidgets.IntSlider(
            value=20, min=2, max=75, continuous_update=False, 
            description='Number features:', style={'description_width': 'initial'}),
        approximation=ipywidgets.Dropdown(
            options=["ExactGP", "RFF",  "QFF", "OFF"], 
            value="RFF",
            description='Kernel Approximation:', style={'description_width': 'initial'}),
        plot_train_data=ipywidgets.Checkbox(
            value=True, description='plot train data',
        ),
        plot_samples=ipywidgets.Checkbox(
            value=False, description='plot function samples',
        ),
    );