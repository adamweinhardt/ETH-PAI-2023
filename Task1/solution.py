import datetime
import logging
import os
import typing
from datetime import time

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
from sklearn.model_selection import train_test_split

EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

import subprocess
import sys

def install(package):
    from subprocess import check_output, STDOUT, CalledProcessError

    try:
        output = check_output([sys.executable, "-m", "pip", "install", package], stderr=STDOUT)
    except CalledProcessError as exc:
        print(exc.output)

install('GPy')
import GPy
class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.seed(0)

        # TODO: Add custom initialization for your model here if necessary
        self.kernel_sigma = 1 
        self.lenght_scale = 1
        self.kernel_bounds = (1e-4, 1e-4) #default:(1e-5, 1e-5)

        # Constant kernel:
        self.Constant_kernel = self.kernel_sigma * ConstantKernel(constant_value = 1.0, constant_value_bounds = (1e-5, 10))

        # RBF (Gaussian, Squared Exponential) kernel:
        self.RBF_kernel = self.kernel_sigma * RBF(length_scale = self.lenght_scale, length_scale_bounds = self.kernel_bounds)

        self.RBF_kernel_GPy = GPy.kern.RBF(input_dim=1)

        # Matérn kernel: smaller the nu, less smooth the function is
        self.Matern_kernel = self.kernel_sigma * Matern(nu = 1.5, length_scale = self.lenght_scale, length_scale_bounds = self.kernel_bounds)
        
        # Rational Quadratic kernel: alpha is the Scale mixture parameter
        self.Rational_Quadratic_kernel = self.kernel_sigma * RationalQuadratic(alpha = 1, length_scale = self.lenght_scale, 
                                                               length_scale_bounds = (1e-5, 1e-5), alpha_bounds = (1e-5, 1e-5))
        
        # Exp-Sine-Squared kernel:
        self.Exp_Sine_Squared_kernel = self.kernel_sigma * ExpSineSquared(length_scale = self.lenght_scale, periodicity = 1,
                                                            length_scale_bounds = (1e-5, 1e-5), periodicity_bounds = (1e-5, 1e-5))
        
        # Dot-Product kernel:
        self.Dot_Product_kernel = self.kernel_sigma * DotProduct(sigma_0 = 1, sigma_0_bounds = (1e-5, 1e-5))

        # Kernel choosing:
        self.kernel =  (self.RBF_kernel)

        # GP initialization:
        self.regressor = GaussianProcessRegressor(kernel = self.kernel, optimizer = "fmin_l_bfgs_b", 
                                                  n_restarts_optimizer = 1)

    def calculate_optimal_residential_threshold(self, mean: int, std: int):
        # TODO find optimal threshold
        return mean + 0.7 * std

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use the GP posterior to form your predictions here
        #gp_mean, gp_std = self.regressor.predict(test_x_2D, return_std=True)

        gp_mean, gp_std = self.GPy_model.predict(test_x_2D, include_likelihood=False)

        preds = np.zeros(shape=(len(gp_mean, )))
        for idx in range(len(gp_mean)):
            is_residential = test_x_AREA[idx]
            pred = gp_mean[idx]
            if is_residential:
                pred = self.calculate_optimal_residential_threshold(gp_mean[idx], gp_std[idx])
            preds[idx] = pred

        return preds, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray, train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        #SUBSAMPLE_SIZE = 1000
        #indices = np.random.randint(0, len(train_y), SUBSAMPLE_SIZE)

        #x_train, y_train = train_x_2D[indices], train_y[indices]

        #self.regressor.fit(x_train, y_train)
        self.Matern52_kernel = GPy.kern.Matern52(input_dim=2, variance=10, lengthscale=1.0) + GPy.kern.White(input_dim=2, variance=1)
        self.Rational_Quadratic_kernel = GPy.kern.RatQuad(input_dim=2, variance=10, lengthscale=1.0, power=2) + GPy.kern.White(input_dim=2, variance=1)

        y_train = np.reshape(train_y,(-1,1))

        x_variance = np.ones_like(train_x_2D)

        #U, sigma, V = np.linalg.svd(train_x_2D)
        inducing_points = train_x_2D[::10]

        model = GPy.models.SparseGPRegression(X=train_x_2D, Y=y_train, kernel=self.Matern52_kernel, Z=inducing_points)
        model.inference_method = GPy.inference.latent_function_inference.FITC()
        model.optimize(optimizer="lbfgs", messages=True, max_iters=50)

        self.GPy_model = model

        return


# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0]) ** 2 + (coor[1] - circle_coor[1]) ** 2 < circle_coor[2] ** 2


# You don't have to change this function
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                        [0.79915856, 0.46147936, 0.1567626],
                        [0.26455561, 0.77423369, 0.10298338],
                        [0.6976312, 0.06022547, 0.04015634],
                        [0.31542835, 0.36371077, 0.17985623],
                        [0.15896958, 0.11037514, 0.07244247],
                        [0.82099323, 0.09710128, 0.08136552],
                        [0.41426299, 0.0641475, 0.04442035],
                        [0.09394051, 0.5759465, 0.08729856],
                        [0.84640867, 0.69947928, 0.04568374],
                        [0.23789282, 0.934214, 0.04039037],
                        [0.82076712, 0.90884372, 0.07434012],
                        [0.09961493, 0.94530153, 0.04755969],
                        [0.88172021, 0.2724369, 0.04483477],
                        [0.9425836, 0.6339977, 0.04979664]])

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i, coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA


# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    print(figure_path)
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = train_x[:, :2].astype(float)
    train_x_AREA = train_x[:, 2].astype(bool)
    test_x_2D = test_x[:, :2].astype(float)
    test_x_AREA = test_x[:, 2].astype(bool)

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA


def validate(model, valid_x_2D, valid_x_AREA, valid_y):

    predictions, mean, std = model.make_predictions(valid_x_2D, valid_x_AREA)

    loss = 0

    for idx in range(len(valid_y)):
        pred = predictions[idx]
        truth = valid_y[idx]

        loss_weight = 50 if pred < truth and valid_x_AREA[idx] else 1

        loss += loss_weight*(pred-truth)**2

    loss /= len(valid_y)
    of = f'validation_loss_{str(datetime.datetime.now())}.txt'

    with open(of, 'w+') as f:
        f.writelines(['loss: '+str(loss)])
    logging.warning("HELLO")
    print('cleaner is up', flush=True)
    print(f'VALIDATION loss: written to {of}')



# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

    # Extract the city_area information
    valid_x_2D, valid_x_AREA, _, _ = extract_city_area_information(valid_x, test_x)
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y, train_x_2D)
    # Predict on the test features

    print('Validating')
    validate(model, valid_x_2D, valid_x_AREA, valid_y)

    print('Predicting on test features')
    predictions, mean, std = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='../../../../Downloads/task1_handout_d3d63876')


if __name__ == "__main__":
    main()
