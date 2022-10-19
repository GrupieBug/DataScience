# This # -*- coding: utf-8 -*-
# """
# Data generation for logistic regression
# """
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from matplotlib import pyplot as plt


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


# def F_l_2(A, b, n_samples, lam, idx, x_star):
#     exponent = - b * A.transpose() * idx
#     sum_l2 = sum(np.log(1 + np.exp(exponent)))
#     lam_term = (lam * idx.transpose() * idx)
#
#     return ((1 / (2 * n_samples)) * sum_l2) + lam_term
#
#
# def F_l_1(A, b, n_samples, lam, idx, x_star):
#     exponent = - idx * x_star.transpose()
#     sum_l1 = sum(np.log(1 + np.exp(exponent)))
#     lam_term = (lam * idx.transpose() * idx)
#
#     return ((1 / (2 * n_samples)) * sum_l1) + lam_term


def proximal_gradient_l_2(A, b, n_samples, lam, idx, x_star):
    exponent = - idx * x_star.transpose()
    sum_l2 = sum(np.log(1 + np.exp(exponent)))
    lam_term = (lam * idx.transpose() * idx)

    # l2 = (1 / (2 * n_samples)) * sum(np.log(1 + np.exp(-A.transpose() * idx * b))) + (lam * idx.transpose() * idx)
    l2 = ((1 / (2 * n_samples)) * sum_l2) + lam_term

    return l2, min(l2)


def proximal_gradient_l_1(A, b, n_samples, lam, idx, x_star):
    exponent = - idx * x_star.transpose()
    sum_l1 = sum(np.log(1 + np.exp(exponent)))
    lam_term = (lam * idx)

    # l2 = (1 / (2 * n_samples)) * sum(np.log(1 + np.exp(-A.transpose() * idx * b))) + (lam * idx.transpose() * idx)
    l1 = ((1 / (2 * n_samples)) * sum_l1) + lam_term

    return l1, min(l1)


def main():
    n_features = 50  # The dimension of the feature is set to 50
    n_samples = 1000  # Generate 1000 training data

    idx = np.arange(n_features)
    coefs = ((-1) ** idx) * np.exp(-idx / 10.)
    coefs[20:] = 0.

    A, b = sim_logistic_regression(n_features, coefs)
    x_star = A.transpose() * b
    print(b.transpose())

    lam = 0.001

    F_l2_steps, l2_min = proximal_gradient_l_2(A, b, n_samples, lam, idx, x_star)
    F_l1_steps, l1_min = proximal_gradient_l_1(A, b, n_samples, lam, idx, x_star)

    F_l2_x_star = np.empty(np.shape(F_l2_steps))
    F_l2_x_star.fill(l2_min)

    F_l1_x_star = np.empty(np.shape(F_l1_steps))
    F_l1_x_star.fill(l1_min)

    x_l2 = idx
    y_l2 = F_l2_steps - F_l2_x_star
    plt.plot(x_l2, y_l2)

    x_l1 = idx
    y_l1 = F_l1_steps - F_l1_x_star
    plt.plot(x_l1, y_l1)

    plt.show()


if __name__ == "__main__":
    main()