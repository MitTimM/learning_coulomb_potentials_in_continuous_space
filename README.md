# Learning Coulomb Potentials in Continuous Space

Code for estimating Coulomb potential parameters in continuous space.

**Paper:** [arXiv:2510.08471](https://arxiv.org/abs/2510.08471) - "Learning Coulomb Potentials and Beyond with Fermions in Continuous Space"

## Quick Start

**Installation:**
```bash
git clone https://github.com/MitTimM/learning_coulomb_potentials_in_continuous_space.git
pip install numpy scipy matplotlib
```

**Single Coulomb (linear solver):**
```bash
cd single_coulomb
python single_main.py
```

**Multi Coulomb (iterative refinement):**
```bash
cd multi_coulomb
python multi_main.py
```

See module READMEs for detailed documentation.

## Project Structure

```
├── geometry.py                 # Shared utilities (space/grid helpers)
├── single_coulomb/             # Single source estimation (linear solver)
│   ├── config.py               # Config (domain, noise, integration, I/O)
│   ├── single_main.py          # Entry point (load/generate → estimate)
│   ├── single_data_acquisition.py   # Probe generation + local averages
│   ├── single_coulomb_estimator.py  # Linear system solver (λ, y)
│   ├── single_plotter.py       # Error vs noise plots
│   ├── data/                   # Generated data
│   └── figures/                # Generated figures
└── multi_coulomb/              # Multiple sources (iterative refinement)
  ├── config.py               # Config (sources, refinement, I/O)
  ├── multi_main.py          # Entry point (load/generate → estimate)
  ├── data_acquisition.py     # Probe generation + local averages
  ├── coulomb_estimator.py    # Iterative refinement + matching
  ├── plotter.py              # Error vs noise + convergence plots
  ├── data/                   # Generated data
  └── figures/                # Generated figures
```

## Configuration

Each module has `config.py` with key parameters:
- `domain_size`, `m_discrete` - domain and grid
- `sigma` - noise level
- `lambda_range` - charge parameter range
- `los` - "0": load, "1": save, "2": sample only

See module documentation for full details.

## License

- Code: CC0 1.0 Universal (Public Domain)
- Paper: Creative Commons Attribution 4.0 International

## Citation

```bibtex
@misc{Bluhm.2025,
  doi = {10.48550/ARXIV.2510.08471},
  url = {https://arxiv.org/abs/2510.08471},
  author = {Bluhm,  Andreas and Lemm,  Marius and M\"{o}bus,  Tim and Siebert,  Oliver},
  title = {Learning Coulomb Potentials and Beyond with Fermions in Continuous Space},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
