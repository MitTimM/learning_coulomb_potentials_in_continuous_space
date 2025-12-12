"""Single Coulomb charge parameter estimation from noisy local averages."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from geometry import find_on_vector, max_box_index, space_vector


def solve_linear_equations(omega, domain_size):
    """Estimate charge λ and position y from local averages using inverse square law.

    Args:
        omega (np.ndarray): 3D array of local averages.
        domain_size (np.ndarray): Physical domain size [Lx, Ly, Lz].

    Returns:
        tuple: (lam_est, y_est) - estimated charge and position.
    """
    shape = np.array(omega.shape, dtype=np.int32)

    # Find maximum (closest to source) and farthest points
    max_index = np.array(np.unravel_index(np.argmax(omega), shape), dtype=np.int32)
    max_dist_index = max_box_index(max_index, shape)

    # Select 4 strategic points for linear system
    index = np.zeros((4, 3), dtype=np.int32)
    index[3], dim = find_on_vector(max_index, max_dist_index, omega)

    ce = np.eye(1, 3, k=dim, dtype=np.int32)[0]
    index[dim] = np.abs(ce - max_dist_index, dtype=np.int32)

    for i in range(3):
        if i != dim:
            ce = np.eye(1, 3, k=i, dtype=np.int32)[0]
            index[i] = np.abs(ce - max_dist_index, dtype=np.int32)

    # Convert indices to physical coordinates
    p_max_dist_vector = space_vector(
        tuple(max_dist_index), np.divide(domain_size, shape)
    )
    p_vectors = np.array(
        [space_vector(tuple(index[i]), np.divide(domain_size, shape)) for i in range(4)]
    )

    # Set up linear system from inverse square law: 1/ω² ∝ ||p - y||²
    eta_diff = np.ones(4) / omega[tuple(max_dist_index)] ** 2 - 1 / np.array(
        [omega[tuple(index[i])] ** 2 for i in range(4)]
    )
    p_diff = 2 * (np.vstack([p_max_dist_vector] * 4) - p_vectors)
    v_diff = np.ones(4) * np.linalg.norm(p_max_dist_vector) ** 2 - np.power(
        np.linalg.norm(p_vectors, axis=1), 2
    )

    # Solve for λ² by eliminating y
    lam_est = (v_diff[3] / p_diff[3, dim] - v_diff[dim] / p_diff[dim, dim]) / (
        eta_diff[3] / p_diff[3, dim] - eta_diff[dim] / p_diff[dim, dim]
    )

    # Back-substitute to find y
    y_est = np.delete(v_diff - eta_diff * lam_est, 3)
    y_est = np.divide(y_est, np.array([p_diff[i, i] for i in range(3)]))

    lam_est = np.sqrt(lam_est)

    return lam_est, y_est
