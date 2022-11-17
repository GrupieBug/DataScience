import numpy
from numpy.random import multivariate_normal
from scipy import spatial
from scipy.linalg.special_matrices import toeplitz
from matplotlib import pyplot as plt
import scipy.io as sio
import pandas


def main():
    classification_dict = {'Iris-setosa': 0,
                           'Iris-versicolor': 1,
                           'Iris-virginica': 2}
    d = 4
    # obtain data points
    df = pandas.read_csv('iris.data.txt', header=None)
    df = df.replace(classification_dict)
    classifications = df[4]
    data = df.drop(4, axis=1)
    data = data.to_numpy()
    n = len(data)
    n_arr = numpy.arange(1, n)

    sum_square_distance = []

    k = 3

    # generate k number of centroids
    centroids = generate_initial_centroids(k, d, data)

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
    mismatched_assignments = []

    for i in range(0, n):
        if classifications[i] != assignments[i]:
            mismatched_assignments.append(i)

    mismatch_count = [0, 0, 0]

    for i in range(0, len(mismatched_assignments)):
        ca = classifications[mismatched_assignments[i]]
        mismatch_count[classifications[mismatched_assignments[i]]] += 1

    mismatch_rate = numpy.divide(mismatch_count, 50)

    print(mismatched_assignments)
    print(mismatch_rate)


# generating k centroids at known data points
def generate_initial_centroids(k, d, data):
    row, col = data.shape
    centroids = numpy.empty([k, col])
    split_by_classification = numpy.split(data, k)
    # assigning each new centroid a known data point (accessing coordinates via index of data point)
    for i in range(0, k):
        centroids[i] = numpy.mean(split_by_classification[i], axis=0)
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
    for a in range(0, n):
        error += numpy.abs(calc_euclidean_distance(data[a], centroids[assignments[a]]))
    return error


if __name__ == "__main__":
    main()