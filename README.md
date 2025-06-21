# Fanti Mechanical Dating — Reproducible Evaluation Codebase

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15098838.svg)](https://doi.org/10.5281/zenodo.15098838)

Python code, data and figures supporting the article  
**“Critical Assessment of Mechanical Dating Methods for Ancient Flax Fibres: Viscoelastic Modelling Versus Multilinear Regression.”**

---

## 📝 Associated Article

> **Title :** *Critical Assessment of Mechanical Dating Methods for Ancient Flax Fibres: Viscoelastic Modelling Versus Multilinear Regression*  
> **Author :** Michel Bakhtaoui  
> **Concept DOI :** 10.5281/zenodo.15098838 (always resolves to the latest release)
---

## 🔍 Project Overview

This repository contains:

- **Statistical re-analysis** of Fanti et al. (2015) mechanical dating data.  
- **Physics-based viscoelastic modelling** (multi-branch Maxwell).  
- **Bayesian inference** and Monte-Carlo uncertainty propagation.  
- **Reproducible figures** and scripts used in the manuscript.

All code is MIT-licensed and fully reproducible.

---

## 🗂 Repository Structure
```text
.
├── CITATION.cff
├── data
│   └── fanti_mechanical_data.csv
├── LICENSE
├── README.md
├── requirements.txt
└── scripts
    ├── modeling
    │   ├── fanti_comparison.py
    │   ├── figure1.png
    │   ├── figure2.png
    │   ├── figure4.png
    │   ├── figure6.png
    │   ├── figure_comparison_fanti.png
    │   └── modelisation.py
    └── statistical-analysis
        ├── analysis.py
        ├── fanti_crossed_analysis.py
        ├── fanti_cross_validation.py
        ├── fanti_experiment.py
        ├── fanti_multiple_regression.py
        ├── fanti_power_analysis.py
        ├── fanti_uncertainty_propagation.py
        ├── figure2.png
        ├── figure3.png
        └── figure5.png
```

---

## 📊 Figures Generated

| Figure | Description                                        | Script                              |
|--------|----------------------------------------------------|-------------------------------------|
| 1      | Age vs mechanical resistance (Maxwell fit)         | `modelisation.py`                   |
| 2      | Regression model performance (fit & cross-val)     | `fanti_multiple_regression.py`      |
| 3      | Monte-Carlo R² distribution                        | `analysis.py`                       |
| 4      | Posterior age distribution (Bayesian)              | `modelisation.py`                   |
| 5      | Variance Inflation Factor (VIF)                    | `fanti_multiple_regression.py`      |
| 6      | MCMC trace & convergence diagnostics               | `modelisation.py`                   |
| **7**  | **Fanti data vs viscoelastic model comparison**    | **`fanti_comparison.py`**           |

---

## 🚀 Quick Start

### 1 • Installation
```bash
git clone https://github.com/mab-plus/fanti-mechanical-dating-analysis.git
cd fanti-mechanical-dating-analysis
python -m pip install -r requirements.txt
```

### 2 • Run full viscoelastic model
```bash
cd scripts/modeling
python modelisation.py          # generates Figures 1, 4, 6
```

### 3 • Reproduce Fanti cross-validation
```bash
cd ../statistical-analysis
python fanti_cross_validation.py  # generates cross-val metrics & Figure 2
```

### 4 • Physics vs empirical comparison
```bash
cd ../modeling
python fanti_comparison.py        # generates Figure 7
```

All generated figures are saved in the corresponding script folders.

---

## 📚 Data Sources

| Dataset | Location | Licence |
|---------|----------|---------|
| Mechanical data (Fanti et al. 2015, 9 samples) | `data/fanti_mechanical_data.csv` | CC-BY |
| Example Maxwell parameters                     | in-code (`modelisation.py`)      | MIT   |

---

## 🤝 How to Cite

> Bakhtaoui M. (2025). *Fanti Mechanical Dating Analysis*. Zenodo.  
> https://doi.org/10.5281/zenodo.15098838

---

## 📑 Licence

This project is released under the **MIT License** (see `LICENSE`).

---

## 🙏 Acknowledgements

Thanks to Dr Alain Bourmaud for providing nano-indentation data, and to the open-source community (NumPy, SciPy, Matplotlib, etc.).
