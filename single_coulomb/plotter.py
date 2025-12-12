"""Single Coulomb estimation error analysis: plot errors vs noise levels."""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from coulomb_estimator import solve_linear_equations
from data_acquisition import generate_probe_points
from config import lambda_range, domain_size


def plot_error_vs_noise(num_errors, num_samples, m_discrete):
    """Plot estimation errors over varying noise levels.

    Args:
        num_errors (int): Number of different noise levels to test.
        num_samples (int): Number of samples per noise level.
        m_discrete (list): Grid size [nx, ny, nz].
    """
    # Generate noise levels (log-spaced from 10^-12 to 10^-2)
    sigma_values = 10 ** (-np.arange(3, num_errors + 3, 1, dtype=np.float64))
    abs_errors = np.zeros((num_errors, num_samples))

    start = time.time()
    for i in range(num_errors):
        print(f"Testing noise level {i+1}/{num_errors}: σ = {sigma_values[i]:.2e}")

        for j in range(num_samples):
            # Generate data and estimate parameters
            lam, y, omega = generate_probe_points(
                m_discrete, lambda_range, domain_size, sigma_values[i], False
            )
            lam_est, y_est = solve_linear_equations(omega, domain_size)

            # Record maximum error (charge or position)
            abs_errors[i, j] = np.max([abs(lam - lam_est), np.linalg.norm(y - y_est)])

        print(f"{i / num_errors * 100:.0f}%", end="\r")

    end = time.time()
    print(f"Elapsed time: {end - start:.6f} seconds")

    # --- Plot ---
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

    # Plot median errors
    median_errors = np.median(abs_errors, axis=1)
    plt.scatter(
        sigma_values, median_errors, marker="D", color="red", label="Median error", s=10
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise level σ (log scale)")
    plt.ylabel("Max of absolute errors |Δλ|, |Δy| (log scale)")
    plt.title(f"Estimation errors vs noise")
    # plt.legend()
    plt.gca().invert_xaxis()
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    # Save figure
    os.makedirs("figures", exist_ok=True)
    fname = f"figures/single_errors_m{m_discrete[0]}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {fname}")
    plt.show()


if __name__ == "__main__":
    plot_error_vs_noise(num_errors=8, num_samples=10, m_discrete=[8, 8, 8])
