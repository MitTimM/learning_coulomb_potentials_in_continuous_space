# Single Coulomb Parameter Estimation

## Overview

Estimates charge (λ) and position (y) of a single Coulomb source from noisy local averages using a linear system solver based on the inverse square law.

## Algorithm

**Linear System Approach:**
1. Find maximum point (closest to source)
2. Select strategic points in 3D domain
3. Use inverse square law to construct linear system
4. Solve for position and charge

## Files

| File | Purpose |
|------|---------|
| `single_main.py` | Entry point; handles data loading/generation and estimation |
| `single_data_acquisition.py` | Generates synthetic data with noise; computes local averages |
| `single_coulomb_estimator.py` | Core linear system solver for parameter estimation |
| `single_plotter.py` | Error analysis; plots estimation error vs. noise levels |
| `config.py` | Configuration parameters (grid, noise, integration settings) |
| `data/` | Data storage directory (created at runtime) |
| `figures/` | Output plots and figures |

## Configuration

Edit `config.py` for key parameters:
- `domain_size`, `m_discrete` - domain and grid resolution
- `sigma` - noise level (higher = harder)
- `lambda_range` - charge parameter range
- `integration_mode` - "fast" (Gauss-Legendre) or "adaptive" (scipy nquad)
- `los` - "0": load, "1": save, "2": sample only

## Usage

**Run estimation:**
```bash
python single_main.py
```

**Error analysis:**
```bash
python single_plotter.py
```

## Algorithm

Uses inverse square law: Φ(x) = λ / ||x - y||

Local averages at strategic grid points provide noisy measurements, enabling parameter recovery via linear system solving.

## Troubleshooting

- **No data:** Set `los = "1"` to generate
- **Poor quality:** Reduce `sigma` or increase `m_discrete`
- **Slow:** Use `integration_mode = "fast"` or reduce `n_per_dim`
