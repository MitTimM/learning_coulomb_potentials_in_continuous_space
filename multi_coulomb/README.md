# Multi Coulomb Parameter Estimation

## Overview

Estimates charge (λ) and position (y) for multiple Coulomb sources from noisy local averages using iterative refinement with Hungarian algorithm matching.

## Algorithm

**Iterative Refinement:**
1. Detect source candidates from potential maxima
2. Iteratively refine positions (search in local boxes)
3. Estimate charges (using isolated regions)
4. Stop when convergence threshold reached

## Files

| File | Purpose |
|------|---------|
| `multi_main.py` | Entry point; orchestrates data generation and estimation |
| `data_acquisition.py` | Generates synthetic data; computes local averages |
| `coulomb_estimator.py` | Core iterative refinement algorithm |
| `plotter.py` | Error analysis; plots errors vs. noise levels |
| `config.py` | Configuration parameters (grid, noise, optimization) |
| `data/` | Data storage directory (created at runtime) |
| `figures/` | Output plots and figures |

## Configuration

Edit `config.py` for key parameters:
- `domain_size`, `m_discrete` - domain and grid resolution
- `num_coulomb` - number of sources (typical: 2-5)
- `y_dist` - minimum separation between sources
- `sigma` - noise level (higher = harder)
- `lambda_range` - charge parameter range
- **Refinement:** `box_size1`, `box_size2`, `max_approx_step`, `prec_step`
- `los` - "0": load, "1": save, "2": sample only

## Usage

**Run estimation:**
```bash
python multi_main.py
```

**Error analysis:**
```bash
python plotter.py
```

## Algorithm

Uses superposed potential: Φ(x) = Σ λ_n / ||x - y_n||

Iterative refinement:
1. Refine positions in local boxes
2. Estimate charges in isolated regions
3. Match sources with Hungarian algorithm
4. Repeat until convergence

## Key Parameters

- `box_size1` - position search width (tight)
- `box_size2` - charge estimation width (wider)
- `max_approx_step` - iteration limit
- `prec_step` - convergence tolerance
- `y_dist` - minimum source separation

## Troubleshooting

- **Convergence:** Increase `max_approx_step`, reduce `prec_step`
- **Poor position:** Increase `box_size1` or `m_discrete`
- **Poor charge:** Increase `box_size2` or `y_dist`
- **Slow:** Use `integration_mode = "fast"` or reduce grid size
- **Separation error:** Increase `domain_size` or reduce `num_coulomb`
