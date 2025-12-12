"""Multi Coulomb parameter estimation: iterative refinement using local isolation."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Tuple
import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure
from scipy.optimize import linear_sum_assignment
from geometry import find_on_vector, max_box_index, space_vector
from data_acquisition import local_averages_fast
from config import (
    m_discrete,
    domain_size,
    num_coulomb,
    numerr,
    box_size1,
    box_size2,
    max_approx_step,
    weight,
    prec_step,
)


# --- Utilities --------------------------------------------------------------


def best_permutation(y_true: np.ndarray, y_est: np.ndarray) -> np.ndarray:
    """Find best matching between two sets of 3D vectors using Hungarian algorithm.

    Args:
        y_true (ndarray): True positions, shape (K, 3).
        y_est (ndarray): Estimated positions, shape (K, 3).

    Returns:
        ndarray: Permutation indices for y_est to match y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_est = np.asarray(y_est, dtype=float)

    if y_true.shape != y_est.shape or y_true.ndim != 2 or y_true.shape[1] != 3:
        raise ValueError("Both arrays must have shape (K, 3)")

    K = y_true.shape[0]
    cost = np.array(
        [[np.linalg.norm(y_true[i] - y_est[j]) for j in range(K)] for i in range(K)]
    )

    return linear_sum_assignment(cost)[1]


def boxing_function(y_center, m_discrete, box_size):
    """Create box bounds around center position in discrete grid.

    Args:
        y_center (ndarray): Center position indices.
        m_discrete (list): Grid size.
        box_size (int): Half-width of box.

    Returns:
        ndarray: Box bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
    """
    bounds = np.zeros((3, 2), dtype=np.int32)
    for k in range(3):
        bounds[k, 0] = max(0, y_center[k] - box_size)
        bounds[k, 1] = min(m_discrete[k] - 1, y_center[k] + box_size)
    return bounds


# --- Charge estimation ------------------------------------------------------


def rough_lambda_estimate(omega, y_est, scale, box_size):
    """Estimate λ from positions y_est by solving linear system.

    Args:
        omega (ndarray): Local averages.
        y_est (ndarray): Estimated positions, shape (K, 3).
        scale (ndarray): Grid scale.
        box_size (int): Box size for sampling.

    Returns:
        ndarray: Estimated charges λ, shape (K,).
    """
    K = len(y_est)
    y_est_index = np.array([y_est[i] // scale for i in range(K)], dtype=np.int32)

    # Shift indices away from singularities
    index = y_est_index.copy()
    for k in range(K):
        for d in range(3):
            if index[k, d] - box_size < 0:
                index[k, d] += box_size
            else:
                index[k, d] -= box_size

    # Build matrix M[k,j] = local_avg at index[k] due to unit charge at y_est[j]
    M = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            M[k, j] = local_averages_fast(index[k], scale, 1.0, y_est[j], n_per_dim=24)

    cond = np.linalg.cond(M)
    if cond > 1e12:
        raise RuntimeError(
            f"Ill-conditioned system (cond={cond:.3e}). "
            "Reduce num_coulomb or increase grid resolution."
        )

    omega_vec = np.array([omega[tuple(index[k])] for k in range(K)])
    lam_est = np.linalg.solve(M, omega_vec)

    return lam_est


# --- Position refinement ----------------------------------------------------


def isolated_local_averages(omega_jpos, y_est_red, lam_est_red, scale, jpos, weight):
    """Compute isolated local average by subtracting contributions from other sources.

    Args:
        omega_jpos (float): Observed local average at jpos.
        y_est_red (ndarray): Positions of other sources (K-1, 3).
        lam_est_red (ndarray): Charges of other sources (K-1,).
        scale (ndarray): Grid scale.
        jpos (ndarray): Position index.
        weight (float): Scaling factor.

    Returns:
        float: Isolated local average.
    """
    omega_others = local_averages_fast(
        jpos, scale, lam_est_red, y_est_red, n_per_dim=24
    )
    return (omega_jpos - omega_others) / weight


def define_local_positions(num_coulomb, local_box_bounds, y_est_index, omega):
    """Define strategic points for each source in local boxes.

    Args:
        num_coulomb (int): Number of sources.
        local_box_bounds (ndarray): Box bounds for each source (K, 3, 2).
        y_est_index (ndarray): Estimated position indices (K, 3).
        omega (ndarray): Local averages.

    Returns:
        tuple: (max_dist_index_global, index_global, dim)
    """
    max_dist_idx = np.zeros((num_coulomb, 3), dtype=np.int32)
    idx_global = np.zeros((num_coulomb, 4, 3), dtype=np.int32)
    dim = np.zeros(num_coulomb, dtype=np.int32)

    for k in range(num_coulomb):
        # Extract local box
        box = local_box_bounds[k]
        shape = box[:, 1] - box[:, 0] + 1
        max_idx_local = y_est_index[k] - box[:, 0]
        max_dist_local = max_box_index(max_idx_local, shape)
        max_dist_idx[k] = max_dist_local + box[:, 0]

        omega_local = omega[
            box[0, 0] : box[0, 1] + 1,
            box[1, 0] : box[1, 1] + 1,
            box[2, 0] : box[2, 1] + 1,
        ]

        # Find 4 strategic points
        idx_local = np.zeros((4, 3), dtype=np.int32)
        idx_local[3], dim[k] = find_on_vector(
            max_idx_local, max_dist_local, omega_local, numerr
        )

        ce = np.eye(1, 3, k=dim[k], dtype=np.int32)[0]
        idx_local[dim[k]] = np.abs(ce - max_dist_local, dtype=np.int32)

        for i in range(3):
            if i != dim[k]:
                ce = np.eye(1, 3, k=i, dtype=np.int32)[0]
                idx_local[i] = np.abs(ce - max_dist_local, dtype=np.int32)

        idx_global[k] = idx_local + box[:, 0]

    return max_dist_idx, idx_global, dim


def solve_isolated_positions(
    num_coulomb, max_dist_idx, idx_global, omega, y_est, lam_est, scale, dim, weight
):
    """Refine positions using isolated local averages (single Coulomb method).

    Args:
        num_coulomb (int): Number of sources.
        max_dist_idx (ndarray): Farthest points from each source (K, 3).
        idx_global (ndarray): Strategic points (K, 4, 3).
        omega (ndarray): Local averages.
        y_est (ndarray): Current position estimates (K, 3).
        lam_est (ndarray): Current charge estimates (K,).
        scale (ndarray): Grid scale.
        dim (ndarray): Principal dimension for each source (K,).
        weight (float): Scaling factor.

    Returns:
        ndarray: Refined positions (K, 3).
    """
    omega_isolated = np.zeros(4)

    for k in range(num_coulomb):
        # Exclude k-th source
        mask = np.ones(num_coulomb, dtype=bool)
        mask[k] = False

        # Compute isolated ω values
        omega_iso_0 = isolated_local_averages(
            omega[tuple(max_dist_idx[k])],
            y_est[mask],
            lam_est[mask],
            scale,
            max_dist_idx[k],
            weight,
        )

        for i in range(4):
            omega_isolated[i] = isolated_local_averages(
                omega[tuple(idx_global[k, i])],
                y_est[mask],
                lam_est[mask],
                scale,
                idx_global[k, i],
                weight,
            )

        # Convert to physical coordinates
        p_max_dist = space_vector(tuple(max_dist_idx[k]), scale)
        p_vecs = np.array(
            [space_vector(tuple(idx_global[k, i]), scale) for i in range(4)]
        )

        # Set up linear system (same as single Coulomb)
        eta_diff = 1 / omega_iso_0**2 - 1 / omega_isolated**2
        p_diff = 2 * (p_max_dist - p_vecs)
        v_diff = np.linalg.norm(p_max_dist) ** 2 - np.linalg.norm(p_vecs, axis=1) ** 2

        # Solve for λ²
        lam_sq = (
            v_diff[3] / p_diff[3, dim[k]] - v_diff[dim[k]] / p_diff[dim[k], dim[k]]
        ) / (
            eta_diff[3] / p_diff[3, dim[k]] - eta_diff[dim[k]] / p_diff[dim[k], dim[k]]
        )

        # Solve for y
        y_est[k] = np.delete(v_diff - eta_diff * lam_sq, 3)
        y_est[k] = y_est[k] / np.array([p_diff[i, i] for i in range(3)])

    return y_est


# --- Main solver ------------------------------------------------------------


def solve_multi_coulomb(omega, lam, y, compl_rate=True):
    """Iteratively estimate λ and y for multiple Coulomb sources.

    Args:
        omega (ndarray): Noisy local averages.
        lam (ndarray): True charges (for error tracking).
        y (ndarray): True positions (for error tracking).

    Returns:
        tuple: (lam_est, y_est, error_y, error_lam, it)
    """
    scale = np.divide(domain_size, m_discrete)

    # Initial estimate: find local maxima
    neighborhood = generate_binary_structure(3, 3)
    local_max = maximum_filter(omega, footprint=neighborhood) == omega
    y_est_index = np.argwhere(local_max)

    # Sort by omega value and take top num_coulomb
    omega_vals = omega[tuple(y_est_index.T)]
    sorted_idx = np.argsort(omega_vals)[::-1]
    y_est_index = y_est_index[sorted_idx[:num_coulomb]]

    y_est = space_vector(y_est_index, scale)
    lam_est = rough_lambda_estimate(omega, y_est, scale, box_size2)

    # Match to true values
    perm = best_permutation(y, y_est)
    y_est = y_est[perm]
    lam_est = lam_est[perm]
    y_est_index = y_est_index[perm]

    # Define local boxes
    local_boxes = np.array(
        [
            boxing_function(y_est_index[k], m_discrete, box_size1)
            for k in range(num_coulomb)
        ]
    )

    max_dist_idx, idx_global, dim = define_local_positions(
        num_coulomb, local_boxes, y_est_index, omega
    )

    # Iterative refinement
    error_y = np.zeros(max_approx_step)
    error_lam = np.zeros(max_approx_step)
    y_prev, lam_prev = np.zeros_like(y_est), np.zeros_like(lam_est)
    it = 0

    while (
        np.max(np.linalg.norm(y_est - y_prev, axis=1)) > prec_step
        or np.max(np.abs(lam_est - lam_prev)) > prec_step
    ) and it < max_approx_step:

        error_y[it] = np.linalg.norm(y - y_est)
        error_lam[it] = np.linalg.norm(lam - lam_est)

        if compl_rate:
            print(
                f"Iteration {it+1}: |Δy| = {error_y[it]:.6e}, |Δλ| = {error_lam[it]:.6e}"
            )

        y_prev, lam_prev = y_est.copy(), lam_est.copy()

        # Refine positions and charges
        y_est = solve_isolated_positions(
            num_coulomb,
            max_dist_idx,
            idx_global,
            omega,
            y_est,
            lam_est,
            scale,
            dim,
            weight,
        )
        lam_est = rough_lambda_estimate(omega, y_est, scale, box_size2)

        it += 1

    return lam_est, y_est, error_y, error_lam, it
