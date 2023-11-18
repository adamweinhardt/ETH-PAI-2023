# --- num
import gpytorch
import numpy as np
import torch
import pandas as pd 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
# --- misc
import time
# --- plot
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.animation import FuncAnimation
# --- ipython
import ipywidgets as widgets
from ipywidgets import interact, Layout
import IPython
from IPython.display import display, HTML
import warnings


def gen_data(num, lims=[-10, 10]):
    """Generates indexed variables that uniformly cover domain.
       Observed function values are pre-computed, sampled from GP prior.
    """
    x = torch.linspace(lims[0], lims[1], num)
    y = torch.zeros(num)
    with open("../data/rand_funcs.csv", "r") as file:
        dat = file.read()
        for idx, val in enumerate(dat.split(",")):
            if idx > num-1:
                break
            y[idx] = float(val)

    return x, y


def cond_mean(y, K, num_cond):
    """Returns Y_A|Y_B
    """
    K11 = K[:num_cond, :num_cond]
    K21 = K[num_cond:, :num_cond]
    cond_mean = K21 @ np.linalg.inv(K11) @ y[:num_cond].numpy()
    return cond_mean


def cond_covar(K, num_cond):
    """Returns S_A|S_B
    """
    K11 = K[:num_cond, :num_cond]
    K12 = K[:num_cond, num_cond:]
    K21 = K[num_cond:, :num_cond]
    K22 = K[num_cond:, num_cond:]

    cond_cov = K22 - K21 @ np.linalg.inv(K11) @ K12

    return cond_cov


def forward_sampling(num_cond, kernel="RBF", l=5, s=0.1):
    # clear
    IPython.display.clear_output()

    x, y = gen_data(10)
    y = y[:num_cond]
    if kernel == "RBF":
        model = GaussianProcessRegressor(
            kernel=kernels.RBF(length_scale=l))
    elif kernel == "Matern-1/2":
        model = GaussianProcessRegressor(
            kernel=kernels.Matern(nu=0.5, length_scale=l))
    elif kernel == "Matern-3/2":
        model = GaussianProcessRegressor(
            kernel=kernels.Matern(nu=1.5, length_scale=l))
    elif kernel == "Matern-5/2":
        model = GaussianProcessRegressor(
            kernel=kernels.Matern(nu=2.5, length_scale=l))
    elif kernel == "Linear":
        k = kernels.PairwiseKernel(
            gamma=l, metric="linear",
            pairwise_kernels_kwargs={"sigma_0": 0}
        )
        kernel = lambda x, y: s*k(x,y) + np.eye(10)*8e-6 +\
            np.ones((10,10))*1e-4
        model = GaussianProcessRegressor(kernel=kernel)
    else:
        raise NotImplementedError

    # get kernel
    K = model.kernel(x.view(10, -1), x.view(10, -1))
    K_ = cond_covar(K, num_cond)

    # set fig
    fig = plt.figure(figsize=[14, 7])
    gs = gridspec.GridSpec(1, 2)
    ax2 = plt.subplot(gs[0])
    ax1 = inset_axes(ax2, width="30%", height="30%", loc=2, borderpad=1)
    axs = [ax1, ax2]

    # plot covar matrix
    M = np.zeros_like(K)
    M[num_cond:, num_cond:] = K_
    axs[0].imshow(M, cmap="viridis")
    axs[0].set_xlabel("covariance", fontsize=10)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # forward sample
    fwd_f = []
    K = model.kernel(x.view(10, -1), x.view(10, -1))
    for i in range(0, 10-num_cond):
        K_ = cond_covar(K, num_cond+i)
        L = np.linalg.cholesky(K_)
        f = cond_mean(y, K, num_cond+i)
        f += L @ torch.randn(10-num_cond-i).numpy()
        f = f[0]
        y = torch.cat([y, torch.tensor([f.item()])])
        fwd_f.append(f)

    # animate
    axs[1].plot(range(0, num_cond), y[:num_cond], "o-")
    line, = axs[1].plot([], [], "o-", markerfacecolor="none")
    axs[1].set_ylim([-3, 3])
    axs[1].set_xlim([-1, 10])
    axs[1].set_xlabel("variable index")
    axs[1].set_ylabel("$f$")
    axs[1].set_ylim([-3, 3])
    axs[1].set_xlim([-1, 10])
    axs[1].set_xlabel("variable index")
    axs[1].set_ylabel("$f$")
    plt.tight_layout()

    def animate_fwd(i):
        line.set_data(range(num_cond, num_cond+i+1), fwd_f[:i+1])
        return line,

    ani = FuncAnimation(fig, animate_fwd, frames=10-num_cond, interval=150, blit=True)
    display(HTML(ani.to_jshtml()))


def cholesky_sampling(num_cond, kernel="RBF", l=5, s=0.1):
    # clear
    IPython.display.clear_output()

    x, y = gen_data(10)
    if kernel == "RBF":
        kernel = kernels.RBF(length_scale=l)
    elif kernel == "Matern-1/2":
        kernel = kernels.Matern(nu=0.5, length_scale=l)
    elif kernel == "Matern-3/2":
        kernel = kernels.Matern(nu=1.5, length_scale=l)
    elif kernel == "Matern-5/2":
        kernel = kernels.Matern(nu=2.5, length_scale=l)
    elif kernel == "Linear":
        k = kernels.PairwiseKernel(
            gamma=l, metric="linear",
            pairwise_kernels_kwargs={"sigma_0": 0}
        )
        kernel = lambda x, y: s*k(x,y) + np.eye(10)*8e-6 +\
            np.ones((10,10))*1e-4
    else:
        raise NotImplementedError

    # condition on ys
    K = kernel(x.view(10, -1), x.view(10, -1))
    K_ = cond_covar(K, num_cond)
    L = np.linalg.cholesky(K_)
    f = cond_mean(y, K, num_cond)
    f += L @ torch.randn(10-num_cond).numpy()

    # set fig
    fig = plt.figure(figsize=[14, 7])
    gs = gridspec.GridSpec(1, 2)
    ax2 = plt.subplot(gs[0])
    ax1 = inset_axes(ax2, width="30%", height="30%", loc=2, borderpad=1)
    axs = [ax1, ax2]

    # plot covar matrix
    M = np.zeros_like(K)
    M[num_cond:, num_cond:] = K_
    axs[0].imshow(M, cmap="viridis")
    axs[0].set_xlabel("covariance", fontsize=10)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # plot func vals
    axs[1].plot(range(0, num_cond), y[:num_cond], "o-")
    axs[1].plot(range(num_cond, 10), f, "o-", markerfacecolor="none")
    if len(f) > 0:
        axs[1].plot(range(num_cond-1, num_cond+1), [y[num_cond-1], f[0]], color="gray")
    axs[1].set_ylim([-3, 3])
    axs[1].set_xlim([-1, 10])
    axs[1].set_xlabel("variable index")
    axs[1].set_ylabel("$f$")

    plt.tight_layout()
    plt.show()


def prior_sampling(kernel="RBF", l=4, s=0.1):
    # clear
    IPython.display.clear_output()

    x, _ = gen_data(10)
    if kernel == "RBF":
        kernel = kernels.RBF(length_scale=l)
    elif kernel == "Matern-1/2":
        kernel = kernels.Matern(nu=0.5, length_scale=l)
    elif kernel == "Matern-3/2":
        kernel = kernels.Matern(nu=1.5, length_scale=l)
    elif kernel == "Matern-5/2":
        kernel = kernels.Matern(nu=2.5, length_scale=l)
    elif kernel == "Linear":
        k = kernels.PairwiseKernel(
            gamma=l, metric="linear",
            pairwise_kernels_kwargs={"sigma_0": 0}
        )
        kernel = lambda x, y: s*k(x/10,y/10) + np.eye(10)*1e-4 + np.ones((10,10))*1e-1
    else:
        raise NotImplementedError

    # sample
    K = kernel(x.view(10, -1), x.view(10, -1))
    L = np.linalg.cholesky(K)
    f = L @ torch.randn(10).numpy()

    # set fig
    fig = plt.figure(figsize=[14, 7])
    gs = gridspec.GridSpec(1, 2)
    ax2 = plt.subplot(gs[0])
    ax1 = inset_axes(
        ax2, width="30%", height="30%", loc=2, borderpad=1)
    axs = [ax1, ax2]

    # plot covar matrix
    axs[0].imshow(K, cmap="viridis", vmin=0, vmax=1)
    axs[0].set_xlabel("covariance", fontsize=10)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_axisbelow(True)

    # plot func vals
    axs[1].plot(range(0, 10), f, "o-", markerfacecolor="none", color="#453781FF")
    axs[1].set_ylim([-3, 3])
    axs[1].set_xlim([-1, 10])
    axs[1].set_xlabel("variable index")
    axs[1].set_ylabel("$f$")

    plt.tight_layout()
    plt.show()


def plot_sample():
    num_cond_wid = widgets.IntSlider(
        value=3, min=3, step=1, max=9,
        continuous_update=True,
        description='Number of points to condition on:',
        layout=widgets.Layout(width="500px"),
        style={'description_width': 'initial'}
    )
    length_scale_wid = widgets.FloatSlider(
        value=4, min=1, step=0.5, max=8,
        continuous_update=True,
        description='lengthscale:',
        layout=widgets.Layout(width="500px"),
        style={'description_width': 'initial'}
    )
    sigma_wid = widgets.FloatSlider(
        value=0.5, min=0.001, step=0.05, max=1,
        continuous_update=True,
        description='sigma:',
        layout=widgets.Layout(width="500px"),
        style={'description_width': 'initial'}
    )
    kernel_wid = widgets.Dropdown(
        options=[
            'RBF', 'Matern-1/2', 'Matern-3/2', 'Matern-5/2', 'Linear'
        ],
        value='RBF',
        description='Kernel selection:',
        style={'description_width': 'initial'})
    kind_wid = widgets.Dropdown(
        options=['Prior (Cholesky)', 'Posterior (Cholesky)', 'Posterior (Forward)'],
        value='Prior (Cholesky)',
        description='Sampling type:',
        style={'description_width': 'initial'})

    output = widgets.Output()

    def resample(_):
        num_cond = num_cond_wid.value
        kernel = kernel_wid.value
        kind = kind_wid.value
        length_scale = length_scale_wid.value
        sigma = sigma_wid.value

        with output:
            output.clear_output()
            if kind == "Posterior (Forward)":
                forward_sampling(num_cond, kernel, length_scale, sigma)
            elif kind == "Posterior (Cholesky)":
                cholesky_sampling(num_cond, kernel, length_scale, sigma)
            elif kind == "Prior (Cholesky)":
                prior_sampling(kernel, length_scale, sigma)

    resample(None)
    resample_button = widgets.Button(description="Resample")
    resample_button.on_click(resample)

    display(
        length_scale_wid, sigma_wid, num_cond_wid, kernel_wid, kind_wid,
        resample_button, output
    )
