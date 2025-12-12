"""Configuration for single Coulomb parameter estimation.

This config file is organized by module dependencies:
  1. Shared utilities (domain, grid, bump functions)
  2. Data acquisition (noise, sampling, integration)
  3. Coulomb estimation (linear system solver)
  4. Main workflow (I/O settings)
"""

import numpy as np

# ============================================================================
# SHARED UTILITIES
# ============================================================================

# --- Domain and Grid ---
domain_size = [1.0, 1.0, 1.0]  # Physical domain [Lx, Ly, Lz]
m_discrete = [10, 10, 10]  # Grid resolution [nx, ny, nz]

# --- Bump Function (used by single_data_acquisition) ---
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
# DATA ACQUISITION (single_data_acquisition.py)
# ============================================================================

# --- Coulomb Source Parameters ---
lambda_range = (1.0, 2.0)  # Charge λ range (min, max)

# --- Noise Parameters ---
epsilon = 1e-5  # Precision of local averages
delta = 0.01  # Failure probability
sigma = np.sqrt(delta * epsilon**2)  # Noise std (derived from ε and δ)

# --- Integration Settings ---
numerr = 5 * 1e-6  # Error tolerance (adaptive mode)
n_per_dim = 40  # Quadrature points per dimension (fast mode, 24-40 sufficient)
integration_mode = "fast"  # "fast": Gauss-Legendre, "adaptive": scipy nquad

# ============================================================================
# COULOMB ESTIMATION (single_coulomb_estimator.py)
# ============================================================================
# Note: Uses only geometry functions and omega data
# No specific config parameters needed

# ============================================================================
# MAIN WORKFLOW (single_main)
# ============================================================================

# --- Data Handling ---
los = "0"  # "0": load, "1": save new data, "2": sample without saving
single_filename = (
    f"single_data_m{m_discrete[0]}.pkl"  # Filename (main prepends data/ directory)
)
