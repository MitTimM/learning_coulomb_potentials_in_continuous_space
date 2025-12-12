"""Multi Coulomb estimation error analysis: plot errors vs noise levels."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from coulomb_estimator import solve_multi_coulomb
from data_acquisition import generate_probe_points
from config import (
    num_coulomb,
    m_discrete,
    lambda_range,
    domain_size,
    los,
    multi_filename,
    pltnoise,
    pltconv,
)


def plot_error_vs_noise(num_errors, num_samples):
    """Plot estimation errors over varying noise levels.

    Args:
        num_errors (int): Number of noise levels (log-spaced 10^-3 to 10^-(num_errors+2)).
        num_samples (int): Samples per noise level.
    """
    # Generate noise levels (log-spaced from 10^-12 to 10^-2)
    sigma_values = 10 ** (-np.arange(3, num_errors + 3, 1, dtype=np.float64))
    abs_errors = np.zeros((num_errors, num_samples))

    start = time.time()
    for i in range(num_errors):
        for j in range(num_samples):
            # Generate data and estimate parameters
            lam, y, omega = generate_probe_points(
                m_discrete, lambda_range, domain_size, sigma_values[i], False
            )
            error_y, error_lam, it = solve_multi_coulomb(omega, lam, y, False)[2:]

            # Record maximum error (charge or position)
            abs_errors[i, j] = np.max([error_y[it - 1], error_lam[it - 1]])
        print(f"{i / num_errors * 100:.0f}%", end="\r")

    end = time.time()
    print(f"Elapsed time: {end - start:.6f} seconds")

    # Plot
    plt.figure(figsize=(7, 5))

    # Scatter individual samples
    for i in range(num_errors):
        plt.scatter(
            sigma_values[i] * np.ones(num_samples),
            abs_errors[i, :],
            color="blue",
            s=2,
            alpha=0.5,
        )

    # Plot median
    median_errors = np.median(abs_errors, axis=1)
    plt.scatter(
        sigma_values, median_errors, marker="D", color="red", label="Median error", s=20
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise level σ (log scale)")
    plt.ylabel("Max of absolute errors |Δλ|, |Δy| (log scale)")
    plt.title(
        f"Estimation errors vs noise (grid: {m_discrete[0]}×{m_discrete[1]}×{m_discrete[2]})"
    )
    # plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()

    # Save figure to multi_coulomb/figures/
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fname = os.path.join(fig_dir, f"multi_errors_c{num_coulomb}_m{m_discrete[0]}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


def plot_convergence(m_discrete, sigma):
    """Plot convergence of iterative refinement (single run)."""

    # Construct data path inside multi_coulomb/data/
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, os.path.basename(multi_filename))

    # Data acquisition
    if los == "0":  # Load
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        lam = data["charge"]
        y = data["position"]
        omega = data["local_averages"]

    elif los == "1":  # Sample and save
        start = time.time()
        lam, y, omega = generate_probe_points(
            m_discrete, lambda_range, domain_size, sigma, False
        )
        data = {
            "charge": lam,
            "position": y,
            "local_averages": omega,
            "Noise_std": sigma,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        end = time.time()
        print(f"Elapsed time - data acquisition: {end - start:.2f} s")
        print(f"Data saved to: {filepath}")

    elif los == "2":  # Sample without saving
        start = time.time()
        lam, y, omega = generate_probe_points(
            m_discrete, lambda_range, domain_size, sigma, False
        )
        end = time.time()
        print(f"Elapsed time - data acquisition: {end - start:.2f} s")

    else:
        raise ValueError("Invalid 'los' in config. Use '0', '1', or '2'.")

    # Parameter estimation
    start = time.time()
    error_y, error_lam, it = solve_multi_coulomb(omega, lam, y, False)[2:]
    end = time.time()
    print(f"Elapsed time - estimation: {end - start:.2f} s")

    # Print per-iteration errors
    print("\nConvergence (per iteration):")
    for k in range(it):
        print(f"  it {k+1:2d}: |Δy|={err_y[k]:.3e}, |Δλ|={err_lam[k]:.3e}")

    # Plot convergence
    plt.semilogy(
        range(1, it + 1), error_lam[0:it], marker="D", label="Charge error |Δλ|"
    )
    plt.semilogy(
        range(1, it + 1), error_y[0:it], marker="D", label="Position error |Δy|"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Absolute errors (log scale)")
    plt.title(f"Convergence (K={num_coulomb}, σ={sigma:g}, m={m_discrete[0]})")
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save figure to multi_coulomb/figures/
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fname = os.path.join(fig_dir, f"multi_convergence_c{num_coulomb}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


if __name__ == "__main__":
    if pltnoise:
        plot_error_vs_noise(num_errors=1, num_samples=1)
    if pltconv:
        plot_convergence(m_discrete=[100, 100, 100], sigma=1e-6)
