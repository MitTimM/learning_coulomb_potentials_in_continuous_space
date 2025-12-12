import numpy as np
from config import *


def space_vector(index, scale):
    """Convert discrete index to continuous space vector.
    Args:
        index (np.array): Discrete index.
        domain_size (np.array): Size of the domain.
        m_discrete (np.array): Number of discrete points along each dimension.
    Returns:
        np.array: Continuous space vector.
    """

    return np.multiply(index + np.array([0.5, 0.5, 0.5]), scale)


def max_box_index(max_index, shape):
    """Find the index of the box furthest from the maximum index along each dimension.
    Args:
        max_index (np.array): Index of the maximum value.
        shape (np.array): Shape of the omega array.
    Returns:
        np.array: Index of the box furthest from the maximum index.
    """

    max_dist_index = np.zeros(3, dtype=np.int32)
    for i in range(0, 3):
        if max_index[i] < shape[i] / 2:
            max_dist_index[i] = shape[i] - 1
        else:
            max_dist_index[i] = 0
    return max_dist_index


def find_on_vector(max_index, max_dist_index, omega):
    """Find the index along the vector from max_index to max_dist_index where the change in eta exceeds numerr.
    Args:
        max_index (np.array): Index of the maximum value.
        max_dist_index (np.array): Index of the box furthest from the maximum index.
        omega (np.array): Array of local averages.
    Returns:
        tuple: Index along the vector and the dimension.
    """

    shape = np.array(omega.shape, dtype=np.int32)
    dim = np.argmax(np.abs(max_index + 1 - 0.5 * shape))
    omega0 = omega[tuple(max_dist_index)]
    e1 = np.eye(1, 3, k=dim, dtype=np.int32)[0]
    eta1 = (
        1 / omega0**2
        - 1 / omega[tuple(np.abs(e1 - max_dist_index, dtype=np.int32))] ** 2
    )
    eta2 = (
        1 / omega0**2
        - 1 / omega[tuple(np.abs(2 * e1 - max_dist_index, dtype=np.int32))] ** 2
    )

    for i in range(2, shape[dim]):
        if abs(eta1 - eta2) / i > numerr:
            index4 = np.abs(i * e1 - max_dist_index, dtype=np.int32)
            break
        elif i == shape[dim] - 1:
            raise KeyError("Could not find suitable index along dimension.")
        else:
            eta2 = (
                1 / omega0**2
                - 1
                / omega[tuple(np.abs((i + 1) * e1 - max_dist_index, dtype=np.int32))]
                ** 2
            )

    return index4, dim
