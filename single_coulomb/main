"""Single Coulomb parameter estimation workflow: data acquisition and estimation."""

import sys
import os

# Add parent directory (Numerics) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
from data_acquisition import generate_probe_points
from coulomb_estimator import solve_linear_equations
from config import single_filename, los, m_discrete, domain_size, sigma, lambda_range


def main():
    """Run single Coulomb estimation workflow: load/generate data, then estimate parameters."""

    # Ensure data directory exists and construct full path
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, os.path.basename(single_filename))

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
        print(f"Elapsed time - data acquisition: {end - start:.6f} seconds")

    elif los == "2":  # Sample without saving
        start = time.time()
        lam, y, omega = generate_probe_points(
            m_discrete, lambda_range, domain_size, sigma
        )
        end = time.time()
        print(f"Elapsed time - data acquisition: {end - start:.6f} seconds")

    # --- Parameter Estimation ---
    start = time.time()
    lam_est, y_est = solve_linear_equations(omega, domain_size)
    end = time.time()

    # Print results
    print(f"\nTrue parameters:      λ = {lam}, y = {y}")
    print(f"Estimated parameters: λ = {lam_est}, y = {y_est}")
    print(
        f"Absolute errors:      |Δλ| = {abs(lam - lam_est):.6e}, "
        f"|Δy| = {np.linalg.norm(y - y_est):.6e}"
    )
    print(f"Noise std: {sigma}")
    print(f"Elapsed time - estimation: {end - start:.6f} seconds\n")


if __name__ == "__main__":
    main()
