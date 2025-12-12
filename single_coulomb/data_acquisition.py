"""Single Coulomb data acquisition: sampling, integration, and noise addition."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.integrate import nquad
import numpy as np
from geometry import space_vector
from config import (
    vectorized_bump_sq,
    norm_bump_sq,
    numerr,
    integration_mode,
    n_per_dim,
)


def sample_parameters(lambda_range, domain_size):
    """Sample λ and y from uniform distributions.

    Args:
        lambda_range (tuple): (min, max) for λ.
        domain_size (list): Domain size per dimension.

    Returns:
        tuple: (λ, y) sampled values.
    """
    if len(domain_size) == 1:
        domain_size = [(0, domain_size[0])] * 3
    elif len(domain_size) == 3:
        domain_size = [(0, y) for y in domain_size]

    lam = np.random.uniform(*lambda_range)
    y = np.array([np.random.uniform(*y) for y in domain_size])

    return lam, y


def local_averages(jpos, scale, lam, y):
    """Compute local average using adaptive quadrature.

    Args:
        jpos (array): Index position.
        scale (array): Scale for averaging.
        lam (float): Charge λ.
        y (array): Position y.

    Returns:
        tuple: (result, error) from integration.
    """

    def integrand(x1, x2, x3):
        return (
            lam
            * norm_bump_sq(x1, x2, x3)
            / np.linalg.norm(
                np.multiply(scale, np.array([x1, x2, x3]) + 0.5 + jpos) - y
            )
        )

    result, error = nquad(
        integrand,
        [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        opts=[{"epsabs": numerr}] * 3,
    )

    return result, error


def local_averages_fast(jpos, scale, lam, y, n_per_dim=24):
    """Fast tensor-product Gauss-Legendre quadrature over [-0.5, 0.5]³.

    Args:
        jpos (array): Index position.
        scale (array): Scale for averaging.
        lam (float): Charge λ.
        y (array): Position y.
        n_per_dim (int): Quadrature points per dimension.

    Returns:
        float: Integrated value.
    """
    a, b = -0.5, 0.5
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n_per_dim)

    # Map nodes from [-1,1] to [a,b]
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    nodes = mid + half * nodes_1d
    weights = weights_1d * half

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    W = np.multiply.outer(np.multiply.outer(weights, weights), weights).reshape(
        (n_per_dim, n_per_dim, n_per_dim)
    )

    # Compute integrand: λ * bump² / ||scale*(coords + 0.5 + jpos) - y||
    bump_sq = vectorized_bump_sq(X, Y, Z)
    coords = np.stack((X, Y, Z), axis=-1)
    phys = scale * (coords + 0.5 + np.asarray(jpos))
    denom = np.maximum(np.linalg.norm(phys - np.asarray(y), axis=-1), 1e-12)

    integrand_vals = lam * bump_sq / denom
    result = np.sum(integrand_vals * W)

    return float(result)


def normal_noise(data, sigma):
    """Add Gaussian noise to data.

    Args:
        data (array): Clean data.
        sigma (float): Noise standard deviation.

    Returns:
        array: Noisy data.
    """
    return data + np.random.normal(0, sigma, size=data.shape)


def test_data(omega, lam, y, domain_size, m_discrete):
    """Compute maximum Newton error (excluding singularity).

    Args:
        omega (array): Local averages.
        lam (float): Charge λ.
        y (array): Position y.
        domain_size (array): Domain size.
        m_discrete (array): Discrete grid size.

    Returns:
        float: Maximum Newton error.
    """
    shape = np.array(omega.shape, dtype=np.int32)
    max_index = np.array(np.unravel_index(np.argmax(omega), shape), dtype=np.int32)
    max_newton_error = 0.0

    for ix in range(shape[0]):
        for iy in range(shape[1]):
            for iz in range(shape[2]):
                # Skip point nearest to singularity
                if (ix != max_index[0]) or (iy != max_index[1]) or (iz != max_index[2]):
                    p = space_vector(
                        np.array([ix, iy, iz]), np.divide(domain_size, m_discrete)
                    )
                    error = np.abs(omega[ix, iy, iz] - lam / np.linalg.norm(p - y))
                    if error > max_newton_error:
                        max_newton_error = error

    return max_newton_error


def generate_probe_points(
    m_discrete, lambda_range, domain_size, sigma, compl_rate=True
):
    """Generate probe points and compute noisy local averages.

    Args:
        m_discrete (list): Grid size per dimension.
        lambda_range (tuple): (min, max) for λ.
        domain_size (list): Domain size.
        sigma (float): Noise standard deviation.

    Returns:
        tuple: (λ, y, omega) with omega = noisy local averages.
    """
    # Sample parameters
    lam, y = sample_parameters(lambda_range, domain_size)
    if compl_rate:
        print(f"Sampled parameters: λ = {lam}, y = {y}")

    scale = np.divide(domain_size, m_discrete)
    omega = np.zeros(tuple(m_discrete))

    # Compute local averages
    if integration_mode == "fast":
        if compl_rate:
            print("Using fast integration mode.")
        i = 1
        for jx in range(m_discrete[0]):
            for jy in range(m_discrete[1]):
                for jz in range(m_discrete[2]):
                    omega[jx, jy, jz] = local_averages_fast(
                        np.array([jx, jy, jz]), scale, lam, y, n_per_dim
                    )
                    if compl_rate:
                        print(f"{i / np.prod(m_discrete) * 100:.0f}%", end="\r")
                    i += 1

    elif integration_mode == "adaptive":
        if compl_rate:
            print("Using adaptive integration mode.")
        i = 1
        for jx in range(m_discrete[0]):
            for jy in range(m_discrete[1]):
                for jz in range(m_discrete[2]):
                    omega[jx, jy, jz] = local_averages(
                        np.array([jx, jy, jz]), scale, lam, y
                    )[0]
                    if compl_rate:
                        print(f"{i / np.prod(m_discrete) * 100:.0f}%", end="\r")
                    i += 1

    # Data integrity test
    if test_data(omega, lam, y, domain_size, m_discrete) < numerr:
        if compl_rate:
            print("Data integrity test passed.")
    else:
        raise ValueError(
            "Data integrity test failed. Increase error tolerance or decrease quadrature points."
        )

    # Add noise
    omega = normal_noise(omega, sigma)

    return lam, y, omega
