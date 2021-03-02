import random
import string

from torch import tensor


def get_random_string(length: int) -> str:
    """Generates random string of ascii chars"""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return "tsne_" + result_str


def squared_euc_dists(x: tensor) -> tensor:
    """
    Calculates squared euclidean distances between rows
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared euclidean distances ||x_i - x_j||^2 (n_points, n_points)
    """
    sq_norms = (x ** 2).sum(dim=1)
    return sq_norms + sq_norms.unsqueeze(1) - 2 * x @ x.t()


def jaccard_distances(x: tensor) -> tensor:
    """
    Calculates jaccard dissimilarities between rows.
    Rows should be binary vectors. Pretty fast function.
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    n_ones = x.sum(dim=1)
    intersection = x @ x.t()
    sum_of_ones = n_ones + n_ones.unsqueeze(1)
    similarity = intersection / (sum_of_ones - intersection)
    return 1 - similarity


def squared_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates squared jaccard dissimilarities between rows.
    Rows should be binary vectors. Pretty fast function.
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    return jaccard_distances(x) ** 2


def general_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates jaccard dissimilarities between rows.
    Rows may be continuous vectors. Pretty slow function
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    lesser = (x < x.unsqueeze(1))
    greater = lesser.__invert__()
    lesser = lesser.float()
    greater = greater.float()
    minimums = (x * lesser + x.unsqueeze(1) * greater).sum(2)
    maximums = (x * greater + x.unsqueeze(1) * lesser).sum(2)
    similarity = minimums / maximums
    return 1 - similarity


def squared_general_jaccard_distances(x: tensor) -> tensor:
    """
    Calculates squared jaccard dissimilarities between rows.
    Rows may be continuous vectors. Pretty slow function
    :param x: Matrix of input points (n_points, n_dimensions)
    :return: Matrix of squared jaccard dissimilarities between x_i and x_j (n_points, n_points)
    """
    return general_jaccard_distances(x) ** 2


def squared_cosine_distances(x: tensor) -> tensor:
    raise NotImplementedError


distance_functions = {"euc": squared_euc_dists,
                      "jaccard": squared_jaccard_distances,
                      "jaccard_general": squared_general_jaccard_distances,
                      "cosine": squared_cosine_distances}


def entropy(p: tensor) -> tensor:
    """
    Calculates Shannon Entropy for every row of a conditional probability matrix
    :param p: Conditional probability matrix, where every row sums up to 1
    :return: 1D tensor of entropies, (n_points,)
    """
    return -(p * p.log2()).sum(dim=1)
