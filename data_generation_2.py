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


def prox_lasso(x, lam):
    """
    This is the proximal function for l1
    :param x: Current iterative x value
    :param lam: lambda constant
    :return:
    """
    if x >= lam:
        return x - lam
    elif x <= -lam:
        return x + lam
    else:
        return 0


def prox_l2(x, lam):
    """
    This is the proximal function we have derived for l2
    :param x: Current iterative x value
    :param lam:
    :return:
    """
    if x > 0:
        return x / (1 - (2 * lam))
    elif x < 0:
        return x / (1 + (2 * lam))
    else:
        return 0


def gradient_descent(x_curr, eta, gradient_func):
    """
    Calculates the gradient descent result before passing into the prox function
    :param x_curr: The current x iterative value
    :param eta: The eta parameter constant
    :param gradient_func: The result of the gradient function previously calculated
    :return: Gradient descent result
    """
    return x_curr - eta * gradient_func


def gradient(A, b, n_samples, x_curr):
    """
    The value of the gradient evaluated at the current x iterative value
    :param A: Given matrix
    :param b: Given matrix
    :param n_samples: 1000, given by the number of training data points
    :param x_curr: current x iterative value
    :return: Gradient value
    """
    ab = np.dot(A.transpose(), -b)
    # ab = -b * A.transpose()
    numerator = np.exp(ab * x_curr) * ab
    denominator = 1 + np.exp(ab * x_curr)
    return (1 / (2 * n_samples)) * sum(numerator / denominator)


def F_minimizer_l1(A, b, n_samples, lam, x_k):
    """
    The final F(x) function we wish to compute for x^k and x^*
    :param A: Given matrix
    :param b: Given matrix
    :param n_samples: 1000, given by the number of training data points
    :param lam: lambda parameter
    :param x_k: current x value of the saved x^k vector or just x^* obtained
    :return: Current F(x) at point provided
    """
    ab = np.dot(A.transpose(), -b)
    return (1 / (2 * n_samples)) * sum(np.log(1 + np.exp(ab * x_k))) + (lam * prox_lasso(x_k, lam))


def F_minimizer_l2(A, b, n_samples, lam, x_k):
    """
    The final F(x) function we wish to compute for x^k and x^* for prox l2
    :param A: Given matrix
    :param b: Given matrix
    :param n_samples: 1000, given by the number of training data points
    :param lam: lambda parameter
    :param x_k: current x value of the saved x^k vector or just x^* obtained
    :return: Current F(x) at point provided
    """
    ab = np.dot(A.transpose(), -b)
    return (1 / (2 * n_samples)) * sum(np.log(1 + np.exp(ab * x_k))) + (lam * prox_l2(x_k, lam))


def calc_prox_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Calculates the x^* for the proximal gradient method. x^* is obtained iteratively
    :param x_curr: current x value, x^t
    :param x_k: The vector we are building of all historic x_t values
    :param A: Given matrix
    :param b: Given matrix
    :param num_iter: How many times we will iterate to converge to x^*
    :param n_samples: The number of given data points
    :param eta: parameter value
    :param lam: parameter lambda
    :return: The historic x_k points and final x_* (the last value of x_k after all iterations)
    """
    # Calculate lasso L1
    x_k = []
    for i in range(0, num_iter):
        gradient_result = gradient(A, b, n_samples, x_curr)  # Calculate gradient at current point
        gradient_descent_result = x_curr - (eta * gradient_result)  # Calculate inside of proximal function
        x_next = prox_lasso(gradient_descent_result, lam)  # Pass to proximal function
        x_k.append(x_curr)  # add to history

        x_curr = x_next  # reset to next iteration
    return x_k, x_curr


def calc_nesterov_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Calculates the x^* for the nestrov method for l1. x^* is obtained iteratively
    :param x_curr: current x value, x^t
    :param x_k: The vector we are building of all historic x_t values
    :param A: Given matrix
    :param b: Given matrix
    :param num_iter: How many times we will iterate to converge to x^*
    :param n_samples: The number of given data points
    :param eta: The number of given data points
    :param eta: parameter value
    :param lam: parameter lambda
    :return: The historic x_k points and final x_* (the last value of x_k after all iterations)
    """
    x_k = []
    # Calculate lasso L1 using Nesterov
    y_curr = x_curr
    for i in range(0, num_iter):
        gradient_result = gradient(A, b, n_samples, y_curr)
        gradient_descent_result = gradient_descent(y_curr, eta, gradient_result)
        x_next = prox_lasso(gradient_descent_result, lam)
        y_next = x_next + ((i / (i + 3)) * (x_next - x_curr))
        x_k.append(x_curr)

        x_curr = x_next
        y_curr = y_next
    return x_k, x_curr


def calc_heavy_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Calculates the x^* for the heavy ball method for l1. x^* is obtained iteratively and takes into account previous points
    obtained for momentum
    :param x_curr: current x value, x^t
    :param x_k: The vector we are building of all historic x_t values
    :param A: Given matrix
    :param b: Given matrix
    :param num_iter: How many times we will iterate to converge to x^*
    :param n_samples: The number of given data points
    :param eta: parameter value
    :param lam: parameter lambda
    :return: The historic x_k points and final x_* (the last value of x_k after all iterations)
    """
    beta = 0.5
    momentum_term = 0
    for i in range(0, num_iter):
        if i >= 1:
            momentum_term = beta * (x_k[i-1] - x_k[i - 2])
        gradient_result = gradient(A, b, n_samples, x_curr)
        gradient_descent_result = gradient_descent(x_curr, eta, gradient_result) + momentum_term
        x_next = prox_lasso(gradient_descent_result, lam)
        x_k.append(x_curr)

        x_curr = x_next

    return x_k, x_curr


def calc_prox_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Same implementation of proximal gradient descent using l2 regularization
    :param x_curr:
    :param x_k:
    :param A:
    :param b:
    :param num_iter:
    :param n_samples:
    :param eta:
    :param lam:
    :return:
    """
    # Calculate lasso L1 using Nesterov
    for i in range(0, num_iter):
        gradient_result = gradient(A, b, n_samples, x_curr)
        gradient_descent_result = gradient_descent(x_curr, eta, gradient_result)
        x_next = prox_l2(gradient_descent_result, lam)
        x_k.append(x_curr)

        x_curr = x_next
    return x_k, x_curr


def calc_nesterov_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Same implementation of Nesterov using l2 regularlization
    :param x_curr:
    :param x_k:
    :param A:
    :param b:
    :param num_iter:
    :param n_samples:
    :param eta:
    :param lam:
    :return:
    """


    # Calculate lasso L1 using Nesterov
    y_curr = x_curr
    for i in range(0, num_iter):
        gradient_result = gradient(A, b, n_samples, y_curr)
        gradient_descent_result = gradient_descent(y_curr, eta, gradient_result)
        x_next = prox_l2(gradient_descent_result, lam)
        y_next = x_next + ((num_iter / (num_iter + 3)) * (x_next - x_curr))
        x_k.append(x_curr)

        x_curr = x_next
        y_curr = y_next
    return x_k, x_curr


def calc_heavy_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam):
    """
    Same implementation of heavy ball using l2 regularization
    :param x_curr:
    :param x_k:
    :param A:
    :param b:
    :param num_iter:
    :param n_samples:
    :param eta:
    :param lam:
    :return:
    """
    beta = 0.1
    momentum_term = 0
    for i in range(0, num_iter):
        if i >= 1:
            momentum_term = beta * (x_k[i-1] - x_k[i - 2])  # not sure about the i's
        gradient_result = gradient(A, b, n_samples, x_curr)
        gradient_descent_result = gradient_descent(x_curr, eta, gradient_result) + momentum_term
        x_next = prox_l2(gradient_descent_result, lam)
        x_k.append(x_curr)

        x_curr = x_next

    return x_k, x_curr


def calc_f_diff(x_star, x_k, A, b, n_samples, lam, num_iter, l1):
    """
    Creates an array of F evaluated at x^star and an array of
    :param x_star:
    :param x_k:
    :param A:
    :param b:
    :param n_samples:
    :param lam:
    :param num_iter:
    :param l1:
    :return:
    """
    # Fill array with x_star to plot and find norm
    x_star_arr = np.empty(np.shape(x_k))
    x_star_arr.fill(x_star)
    norm_k_1 = np.linalg.norm(x_k, 1)

    # calculate F(x^*)
    if l1:
        F_star = F_minimizer_l1(A, b, n_samples, lam, x_star)
    else:
        F_star = F_minimizer_l2(A, b, n_samples, lam, x_star)
    F_k = []

    # calculate F(x^k) at each point
    for i in range(0, num_iter):
        F_k.append(F_minimizer_l1(A, b, n_samples, lam, x_k[i]))

    # Fill array with F(x^*) single result
    F_star_arr = np.empty(np.shape(F_k))
    F_star_arr.fill(F_star)

    # Find difference array to plot, make array with iteration points
    diff = F_k - F_star_arr

    return diff


def main():
    n_features = 50  # The dimension of the feature is set to 50
    n_samples = 1000  # Generate 1000 training data

    idx = np.arange(n_features)
    coefs = ((-1) ** idx) * np.exp(-idx / 10.)
    coefs[20:] = 0.

    A, b = sim_logistic_regression(n_features, coefs)

    # setting parameters
    num_iter = 1000
    x_start = 1
    x_curr = x_start  # what is x_0
    x_k = [x_curr]
    eta = 0.00001
    lams = [0.005, 0.01, 0.05, 0.1]

    for lam in lams:
        # l1 implementations
        x_k_prox1, x_star_prox1 = calc_prox_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)
        x_k_heavy1, x_star_heavy1 = calc_heavy_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)
        x_k_nes1, x_star_nes1 = calc_nesterov_l1(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)

        # calculating differences for each method
        prox1_diff = calc_f_diff(x_star_prox1, x_k_prox1, A, b, n_samples, lam, num_iter, l1=True)
        heavy1_diff = calc_f_diff(x_star_heavy1, x_k_heavy1, A, b, n_samples, lam, num_iter, l1=True)
        nes1_diff = calc_f_diff(x_star_nes1, x_k_nes1, A, b, n_samples, lam, num_iter, l1=True)

        # l2 implementations
        x_k_prox2, x_star_prox2 = calc_prox_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)
        x_k_heavy2, x_star_heavy2 = calc_heavy_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)
        x_k_nes2, x_star_nes2 = calc_nesterov_l2(x_curr, x_k, A, b, num_iter, n_samples, eta, lam)

        # calculating differences for each method
        prox2_diff = calc_f_diff(x_star_prox2, x_k_prox2, A, b, n_samples, lam, num_iter, l1=False)
        heavy2_diff = calc_f_diff(x_star_heavy2, x_k_heavy2, A, b, n_samples, lam, num_iter, l1=False)
        nes2_diff = calc_f_diff(x_star_nes2, x_k_nes2, A, b, n_samples, lam, num_iter, l1=False)

        x_iter = np.arange(num_iter)  # x axis

        # Plot results
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        l1_results = plt.figure(f"Lasso Regularization for lambda = {lam}")
        plt.plot(x_iter, prox1_diff)
        plt.plot(x_iter, heavy1_diff)
        plt.plot(x_iter, nes1_diff)
        l2_results = plt.figure(f"Ridge Regularization for lambda = {lam}")
        plt.plot(x_iter, prox2_diff)
        plt.plot(x_iter, heavy2_diff)
        plt.plot(x_iter, nes2_diff)
        plt.show()


if __name__ == "__main__":
    main()