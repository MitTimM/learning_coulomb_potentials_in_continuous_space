"""Multi Coulomb parameter estimation workflow: data acquisition and iterative estimation."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
from data_acquisition import generate_probe_points
from coulomb_estimator import solve_multi_coulomb
from config import sigma, m_discrete, lambda_range, domain_size, multi_filename, los


def main():
    """Run multi Coulomb estimation workflow: load/generate data, then estimate parameters."""

    # Ensure data directory exists and construct full path
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, os.path.basename(multi_filename))

    # --- Data Acquisition ---
    if los == "0":  # Load existing data
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        lam = data["charge"]
        y = data["position"]
        omega = data["local_averages"]

    elif los == "1":  # Sample and save
        start = time.time()
        lam, y, omega = generate_probe_points(
            m_discrete, lambda_range, domain_size, sigma
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
            m_discrete, lambda_range, domain_size, sigma
        )
        end = time.time()
        print(f"Elapsed time - data acquisition: {end - start:.2f} s")

    else:
        raise ValueError(
            "Invalid 'los' in config. Use '0' (load), '1' (save), or '2' (no save)."
        )

    # --- Parameter Estimation ---
    start = time.time()
    lam_est, y_est, error_y, error_lam, it = solve_multi_coulomb(omega, lam, y)
    end = time.time()

    # Print results
    print(f"\nTrue parameters:")
    print(f"  λ = {lam}")
    print(f"  y =\n{y}")
    print(f"\nEstimated parameters:")
    print(f"  λ = {lam_est}")
    print(f"  y =\n{y_est}")
    print(f"\nAbsolute errors (iteration {it}):")
    print(f"  |Δλ| = {error_lam[it-1]:.6e}")
    print(f"  |Δy| = {error_y[it-1]:.6e}")
    print(f"\nNoise std: {sigma}")
    print(f"Elapsed time - estimation: {end - start:.2f} s\n")

    return omega, lam, y


if __name__ == "__main__":
    main()
