import numpy
from numpy.random import multivariate_normal
from scipy import spatial
from scipy.linalg.special_matrices import toeplitz
from matplotlib import pyplot as plt
import scipy.io as sio


def main():
    k = 4
    d = 4
    # obtain data points
    matlab = sio.loadmat('kmeansgaussian.mat')
    data = matlab['X']
    n = len(data)

    # generate k number of centroids
    centroids = generate_k_centroids(k, d)

    # assign centroid for each point
    assignments = assign_centroids(data, k, n, centroids)

    # calculate new cluster location for each k cluster
    new_centroids = reassign_centroids(centroids, data, assignments, k, d, n)


def generate_k_centroids(k, d):
    centroids = numpy.random.random((k, d))
    return centroids


def assign_centroids(data, k, n, centroids):
    """
    :return: n x 1 matrix where each data point is assigned to one of k clusters
    """
    assigned_clusters = []
    for d in range(0, n):
        min_distance = 0
        assigned_cluster = 0
        for c in range(0, k):
            if c == 0:
                min_distance = calc_euclidean_distance(data[d], centroids[c])
            else:
                distance = calc_euclidean_distance(data[d], centroids[c])
                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = c
        assigned_clusters.append(assigned_cluster)
    return assigned_clusters


def calc_euclidean_distance(data_coordinate, centroid_coordinate):
    return numpy.linalg.norm(centroid_coordinate - data_coordinate)


def calc_cosine_distance(data_coordinate, centroid_coordinate):
    return 1 - spatial.distance.cosine(centroid_coordinate, data_coordinate)


def reassign_centroids(centroids, data, assignments, k, d, n):
    new_centroids = []
    for c in range(0, k):
        clustered_data = []
        for i in range(0, n):
            if assignments[i] == c:
                clustered_data.append(data[i])
        clustered_data = numpy.stack(clustered_data)

        summed_columns = [sum(x) for x in zip(*clustered_data)]
        new_centroid = numpy.divide(summed_columns, len(clustered_data))

        new_centroids.append(summed_columns)
    return new_centroids


if __name__ == "__main__":
    main()
