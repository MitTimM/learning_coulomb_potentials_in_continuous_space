# Learning Coulomb Potentials in Continuous Space

## Overview

This repository contains code for learning and estimating Coulomb potential parameters in continuous space, as presented in the paper:

**"Learning Coulomb Potentials and Beyond with Free Fermions in Continuous Space"**

arXiv: [10.48550/arXiv.2510.08471](https://arxiv.org/abs/2510.08471)

## Project Structure

```
├── geometry.py                 # Utility functions for coordinate transformations
├── LICENSE                     # CC0 1.0 Universal License
├── README.md                   # This file
├── single_coulomb/             # Single Coulomb source estimation module
│   ├── README.md              # Single Coulomb module documentation
│   ├── config.py              # Configuration for single Coulomb experiments
│   ├── single_main            # Main entry point
│   ├── single_data_acquisition.py     # Data generation and sampling
│   ├── single_coulomb_estimator.py    # Parameter estimation algorithm
│   ├── single_plotter.py      # Visualization and analysis
│   ├── data/                  # Data files (generated at runtime)
│   └── figures/               # Output plots and figures
└── multi_coulomb/              # Multi Coulomb sources estimation module
    ├── README.md              # Multi Coulomb module documentation
    ├── config.py              # Configuration for multi Coulomb experiments
    ├── main                   # Main entry point
    ├── data_acquisition.py    # Data generation and sampling
    ├── coulomb_estimator.py   # Parameter estimation algorithm
    ├── plotter.py             # Visualization and analysis
    ├── data/                  # Data files (generated at runtime)
    └── figures/               # Output plots and figures
```

## Features

### Single Coulomb Module
- **Linear system solver** for estimating single Coulomb source parameters
- Estimates both charge (λ) and position (y) from noisy local averages
- Uses inverse square law and strategic point selection

### Multi Coulomb Module
- **Iterative refinement algorithm** for multiple Coulomb sources
- Handles multiple charge-position pairs with automatic source separation
- Includes iterative position and charge estimation with convergence tracking
- Hungarian algorithm-based permutation matching

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for plotting)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MitTimM/learning_coulomb_potentials_in_continuous_space.git
cd learning_coulomb_potentials_in_continuous_space
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Quick Start

### Single Coulomb Estimation

Navigate to the single_coulomb folder and run:
```bash
python single_main
```

This will:
- Load or generate test data (configured in `config.py`)
- Estimate the Coulomb source parameters
- Display error metrics and convergence information

See [single_coulomb/README.md](single_coulomb/README.md) for detailed information.

### Multi Coulomb Estimation

Navigate to the multi_coulomb folder and run:
```bash
python main
```

This will:
- Load or generate test data for multiple sources
- Run iterative refinement algorithm
- Provide error analysis and convergence metrics

See [multi_coulomb/README.md](multi_coulomb/README.md) for detailed information.

## Configuration

Each module (single and multi coulomb) has a `config.py` file with the following key settings:

**Domain & Grid:**
- `domain_size`: Physical domain dimensions [Lx, Ly, Lz]
- `m_discrete`: Grid resolution [nx, ny, nz]

**Noise Parameters:**
- `epsilon`: Precision of local averages
- `delta`: Failure probability
- `sigma`: Noise standard deviation

**Sampling & Integration:**
- `lambda_range`: Charge parameter range
- `n_per_dim`: Quadrature points per dimension
- `integration_mode`: "fast" (Gauss-Legendre) or "adaptive" (scipy nquad)

**Data Handling:**
- `los`: "0" (load), "1" (save), "2" (sample without saving)

See the respective `config.py` files for all available parameters.

## Usage Examples

### Generate and Analyze Data

Edit the `config.py` file in the desired module:
```python
los = "1"  # Generate and save new data
m_discrete = [20, 20, 20]  # Adjust grid resolution
sigma = 1e-4  # Adjust noise level
```

Run the main script:
```bash
python single_main  # or 'python main' for multi_coulomb
```

### Run Error Analysis

Run the plotter module:
```bash
python single_plotter.py  # Single Coulomb error analysis
python plotter.py  # Multi Coulomb error analysis
```

This generates plots of estimation errors vs. noise levels.

## File Organization

- **config.py**: Centralized configuration organized by module dependencies
- **data_acquisition.py**: Data generation, sampling, and integration
- **coulomb_estimator.py** / **single_coulomb_estimator.py**: Core estimation algorithms
- **plotter.py** / **single_plotter.py**: Visualization and error analysis
- **geometry.py**: Coordinate transformation utilities

## License

This project is licensed under the **Creative Commons CC0 1.0 Universal License**, which dedicates the work to the public domain. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{TimMiller2024,
  title={Learning Coulomb Potentials and Beyond with Free Fermions in Continuous Space},
  author={...},
  journal={arXiv preprint arXiv:2510.08471},
  year={2024}
}
```

## Contact & Support

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated:** December 2024
