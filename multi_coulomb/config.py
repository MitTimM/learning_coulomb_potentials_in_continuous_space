"""Configuration for multi Coulomb parameter estimation.

This config file is organized by module dependencies:
  1. Shared utilities (domain, grid, bump functions)
  2. Data acquisition (noise, sampling, integration)
  3. Coulomb estimation (optimization, refinement)
  4. Main workflow (I/O settings)
"""

import numpy as np

# ============================================================================
# SHARED UTILITIES
# ============================================================================

# --- Domain and Grid ---
domain_size = [1.0, 1.0, 1.0]  # Physical domain [Lx, Ly, Lz]
m_discrete = [40, 40, 40]  # Grid resolution [nx, ny, nz]

# --- Bump Function (used by data_acquisition and coulomb_estimator) ---
V_norm = 0.002668522135922286**2  # Normalization constant


def norm_bump_sq(x1, x2, x3):
    """Normalized squared bump function (scalar).

    Args:
        x1, x2, x3 (float): Coordinates.

    Returns:
        float: Normalized bump² value.
    """
    r2 = x1**2 + x2**2 + x3**2
    if r2 < 0.25:
        y = np.exp(-2 / (0.25 - r2))
    else:
        y = 0.0
    return y / V_norm


def vectorized_bump_sq(x, y, z):
    """Normalized squared bump function (vectorized).

    Args:
        x, y, z (array): Coordinate arrays.

    Returns:
        array: Normalized bump² values.
    """
    r2 = x * x + y * y + z * z
    out = np.zeros_like(r2)
    mask = r2 < 0.25
    out[mask] = np.exp(-2.0 / (0.25 - r2[mask]))
    return out / V_norm


# ============================================================================
# DATA ACQUISITION (data_acquisition.py)
# ============================================================================

# --- Coulomb Source Parameters ---
num_coulomb = 2  # Number of Coulomb sources
lambda_range = (1.0, 2.0)  # Charge λ range (min, max)
y_dist = 0.3  # Minimum separation between sources
sample_limit = 100  # Max attempts to sample well-separated sources

# --- Noise Parameters ---
epsilon = 0  # 1e-5  # Precision of local averages
delta = 0  # 0.01  # Failure probability
sigma = 0  # np.sqrt(delta * epsilon**2)  # Noise std (derived from ε and δ)

# --- Integration Settings ---
numerr = 5 * 1e-6  # Error tolerance (adaptive mode)
n_per_dim = 30  # Quadrature points per dimension (fast mode)
integration_mode = "fast"  # "fast": Gauss-Legendre, "adaptive": scipy nquad

# ============================================================================
# COULOMB ESTIMATION (coulomb_estimator.py)
# ============================================================================

# --- Iterative Refinement Parameters ---
box_size1 = 3  # Half-width for position refinement box
box_size2 = 5  # Half-width for charge estimation box
max_approx_step = 100  # Max iterations for iterative refinement
prec_step = 1e-10  # Convergence tolerance for iterative refinement
weight = 100.0  # Weight parameter (unused in current implementation)

# ============================================================================
# MAIN WORKFLOW (main)
# ============================================================================

# --- Data Handling ---
los = "1"  # "0": load, "1": save new data, "2": sample without saving
multi_filename = f"multi{num_coulomb}_data_m{m_discrete[0]}_v2.pkl"  # Filename (main prepends data/ directory)
real_opt = True  # Whether to use the real values for optimization benchmarking

# ============================================================================
# PLOT
# ============================================================================
pltnoise = True  # Whether to plot error vs noise
pltconv = True  # Whether to plot convergence
