"""Multi Coulomb data acquisition: sampling, integration, and noise addition."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import nquad
from geometry import space_vector
from config import (
    vectorized_bump_sq,
    norm_bump_sq,
    numerr,
    integration_mode,
    sample_limit,
    y_dist,
    m_discrete,
    num_coulomb,
    lambda_range,
    domain_size,
    sigma,
    n_per_dim,
)


# --- Sampling ---------------------------------------------------------------


def sample_parameters():
    """Sample λ (shape K,) and y (shape K,3) with min pairwise separation."""
    # Normalize domain to [(0, Lx), (0, Ly), (0, Lz)]
    bounds = (
        [(0, domain_size[0])] * 3
        if len(domain_size) == 1
        else [(0, d) for d in domain_size]
    )

    lam = np.random.uniform(*lambda_range, size=num_coulomb)

    for _ in range(sample_limit):
        y = np.array(
            [[np.random.uniform(*b) for b in bounds] for _ in range(num_coulomb)]
        )
        d = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
        if np.all(d[np.triu_indices_from(d, k=1)] >= y_dist):
            return lam, y

    raise RuntimeError(
        f"Failed to sample {num_coulomb} points with separation ≥ {y_dist} "
        f"after {sample_limit} attempts."
    )


# --- Local averages ---------------------------------------------------------


def local_averages(jpos, scale, lam, y):
    """Adaptive quadrature over [-0.5,0.5]^3."""
    K = len(lam)

    def integrand(x1, x2, x3):
        phys = np.multiply(scale, np.array([x1, x2, x3]) + 0.5 + jpos)
        val = 0.0
        for k in range(K):
            val += lam[k] * norm_bump_sq(x1, x2, x3) / np.linalg.norm(phys - y[k])
        return val

    res, err = nquad(
        integrand,
        [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        opts=[{"epsabs": numerr}] * 3,
    )
    return res, err


def local_averages_fast(jpos, scale, lam, y, n_per_dim=n_per_dim):
    """Fast tensor-product Gauss-Legendre quadrature."""
    a, b = -0.5, 0.5
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n_per_dim)
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    nodes = mid + half * nodes_1d
    weights = weights_1d * half

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    W = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]

    bump_sq = vectorized_bump_sq(X, Y, Z)

    coords = np.stack((X, Y, Z), axis=-1)
    phys = scale * (coords + 0.5 + np.asarray(jpos))

    lam_arr = np.atleast_1d(np.asarray(lam))
    y_arr = np.asarray(y)

    diff = phys[..., None, :] - y_arr[None, None, None, ...]
    denom = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)

    contrib = lam_arr[None, None, None, :] / denom
    integrand_vals = bump_sq * np.sum(contrib, axis=-1)

    return float(np.sum(integrand_vals * W))


# --- Consistency check ------------------------------------------------------


def test_data(omega, lam, y, scale):
    """Return (num_coulomb+1)-th largest Newton error (exclude singularities)."""
    shape = np.array(omega.shape, dtype=np.int32)
    err = np.zeros_like(omega)

    for ix in range(shape[0]):
        for iy in range(shape[1]):
            for iz in range(shape[2]):
                p = space_vector(np.array([ix, iy, iz]), scale)
                exact = np.sum(lam / np.maximum(np.linalg.norm(p - y, axis=1), 1e-12))
                err[ix, iy, iz] = abs(omega[ix, iy, iz] - exact)

    s = np.sort(err.ravel())
    return s[-(9 * num_coulomb + 1)]


# --- Noise ------------------------------------------------------------------


def normal_noise(data):
    """Add Gaussian noise with std sigma."""
    return data + np.random.normal(0, sigma, size=data.shape)


# --- Driver -----------------------------------------------------------------


def generate_probe_points(
    m_discrete, lambda_range, domain_size, sigma, compl_rate=True
):
    """Generate λ, y, and noisy omega on the grid."""
    lam, y = sample_parameters()
    if compl_rate:
        print(f"Sampled parameters:\nλ = {lam}\ny =\n{y}")

    scale = np.divide(domain_size, m_discrete)
    omega = np.zeros(tuple(m_discrete))

    total = int(np.prod(m_discrete))
    i = 0

    if integration_mode == "fast":
        if compl_rate:
            print("Using fast integration mode.")
        for jx in range(m_discrete[0]):
            for jy in range(m_discrete[1]):
                for jz in range(m_discrete[2]):
                    omega[jx, jy, jz] = local_averages_fast(
                        np.array([jx, jy, jz]), scale, lam, y
                    )
                    i += 1
                    if compl_rate:
                        print(f"{i / total * 100:.0f}%", end="\r")

    elif integration_mode == "adaptive":
        if compl_rate:
            print("Using adaptive integration mode.")
        for jx in range(m_discrete[0]):
            for jy in range(m_discrete[1]):
                for jz in range(m_discrete[2]):
                    omega[jx, jy, jz] = local_averages(
                        np.array([jx, jy, jz]), scale, lam, y
                    )[0]
                    i += 1
                    if compl_rate:
                        print(f"{i / total * 100:.0f}%", end="\r")

    # Consistency check (clean data)
    max_err = test_data(omega, lam, y, scale)
    if max_err < numerr:
        if compl_rate:
            print("Data integrity test passed.")
    else:
        raise ValueError(
            f"Data integrity test failed: error = {max_err:.6e} > {numerr}. "
            f"Increase numerr or n_per_dim."
        )

    # Add noise
    omega = normal_noise(omega)
    return lam, y, omega
