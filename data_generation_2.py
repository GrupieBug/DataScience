from tkinter import X
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from matplotlib import pyplot as plt
from scipy.linalg import solve_toeplitz


def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))


def sim_logistic_regression(n_features, coefs, n_samples=1000, corr=0.5):
    """"
    Simulation of a logistic regression model

    Parameters
    coefs: `numpy.array', shape(n_features,), coefficients of the model
    n_samples: `int', number of samples to simulate
    corr: `float', correlation of the features

    Returns
    A: `numpy.ndarray', shape(n_samples, n_features)
       Simulated features matrix. It samples of a centered Gaussian vector with covariance
       given bu the Toeplitz matrix

    b: `numpy.array', shape(n_samples,), Simulated labels
    """
    cov = toeplitz(corr ** np.arange(0, n_features))
    A = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    p = sigmoid(A.dot(coefs))
    b = np.random.binomial(1, p, size=n_samples)
    b = 2 * b - 1
    return A, b


def prox_lasso(x, lam):
    if x >= lam:
        return x - lam
    elif x <= -lam:
        return x + lam
    else:
        return 0


def gradient_descent(x_prev, eta, gradient_func):
    return x_prev - eta * gradient_func


def gradient(A, b, n_samples, x):
    ab = np.dot(A.transpose(), -b)
    numerator = np.exp(ab * x) * ab
    denominator = 1 + np.exp(ab * x)
    return (1 / n_samples) * sum(numerator / denominator)


def main():
    n_features = 50  # The dimension of the feature is set to 50
    n_samples = 1000  # Generate 1000 training data

    idx = np.arange(n_features)
    coefs = ((-1) ** idx) * np.exp(-idx / 10.)
    coefs[20:] = 0.

    A, b = sim_logistic_regression(n_features, coefs)

    num_iter = 100000
    x_curr = 0  # what is x_0
    eta = 0.1
    lam = 0.01

    # Calculate lasso L1 using Nesterov
    for i in range(0, num_iter):
        gradient_result = gradient(A, b, n_samples, x_curr)
        gradient_descent_result = gradient_descent(x_curr, eta, gradient_result)
        x_next = prox_lasso(gradient_descent_result, lam)

        # x_next = prox_lasso(gradient_descent(x_curr, eta, gradient(A, b, n_samples, x_curr)), lam)
        x_curr = x_next

    x_star = x_curr



if __name__ == "__main__":
    main()