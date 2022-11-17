import numpy
from numpy.random import multivariate_normal
from scipy import spatial
from scipy.linalg.special_matrices import toeplitz
from matplotlib import pyplot as plt
import scipy.io as sio


def main():
    d = 4
    # obtain data points
    matlab = sio.loadmat('kmeansgaussian.mat')
    data = matlab['X']
    n = len(data)
    n_arr = numpy.arange(1, n)

    sum_square_distance = []

    for k in range(1, n):
        # generate k number of centroids
        centroids = generate_k_centroids(k, d)

        # assign centroid for each point
        assignments = assign_centroids(data, k, n, centroids)

        # calculate new cluster location for each k cluster
        new_centroids = reassign_centroids(centroids, data, assignments, k, d, n)
        new_centroids = numpy.stack(new_centroids)

        while not (centroids == new_centroids).all():
            centroids = new_centroids
            # assign centroid for each point
            assignments = assign_centroids(data, k, n, centroids)

            # calculate new cluster location for each k cluster
            new_centroids = reassign_centroids(centroids, data, assignments, k, d, n)
            new_centroids = numpy.stack(new_centroids)

        centroids = new_centroids

        sum_square_distance.append(sum_distance(centroids, data, assignments, k, n))

    plt.plot(n_arr, sum_square_distance)
    plt.show()


def generate_k_centroids(k, d):
    centroids = numpy.random.random((k, d))  # idea: assign the initial centroids to be pre-existing data points
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
    clustered_data = dict()
    for i in range(0, n):
        assignment = assignments[i]
        if assignment not in clustered_data:
            clustered_data[assignment] = []
            clustered_data[assignment].append(data[i])
        else:
            clustered_data[assignment].append(data[i])
    for c in range(0, k):
        if c in clustered_data:
            cluster = numpy.stack(clustered_data[c])

            summed_columns = [sum(x) for x in zip(*cluster)]
            new_centroid = numpy.divide(summed_columns, len(cluster))

            new_centroids.append(new_centroid)
        else:
            new_centroids.append(centroids[c])  # or we can reassign to new one with random coordinates
    return new_centroids


def sum_distance(centroids, data, assignments, k, n):
    error = 0
    # for a in range(0, n):
    #     for c in range(0, k):
    #         if assignments[a] == c:
    #             error += numpy.abs(calc_euclidean_distance(data[a], centroids[c]))
    for a in range(0, n):
        error += numpy.abs(calc_euclidean_distance(data[a], centroids[assignments[a]]))
    return error


if __name__ == "__main__":
    main()
