# Fanti Mechanical Dating Analysis

This repository contains the full set of scripts used in the critical evaluation of the mechanical and opto-chemical dating method proposed by Fanti et al. (2015). The project provides a dual approach:

- **Statistical reanalysis** of the multilinear regression model
- **Viscoelastic modelling** using a multi-branch Maxwell model calibrated on ancient flax fibre data

All analyses support a methodology-driven interpretation of the data, in line with radiocarbon dating standards.

## 📄 Associated Article

> **Title:** *Critical Assessment of Mechanical Dating Methods for Ancient Flax Fibres: Viscoelastic Modelling Versus Multilinear Regression*  
> **Author:** Michel Bakhtaoui  
> **Submitted to:** *Archaeometry*  
> **DOI (Zenodo):** _[to be added after DOI is created]_

## 🗂 Repository Structure

```
scripts/
├── modeling
│   └── modelisation.py
└── statistical-analysis
    ├── analysis.py
    ├── fanti_crossed_analysis.py
    ├── fanti_cross_validation.py
    ├── fanti_experiment.py
    ├── fanti_multiple_regression.py
    ├── fanti_power_analysis.py
    ├── fanti_uncertainty_propagation.py
```

## 📊 Figures Generated

Each script generates one or more of the following figures used in the article:

| Figure | Description  | Script  |
|--------|--------------------------------------------------|---------------------------------|
| 1  | Age vs mechanical resistance (Maxwell)   | `modelisation.py`   |
| 2  | Regression model (fit, R², RMSE) | `fanti_multiple_regression.py`  |
| 3  | Monte Carlo analysis (R² variation)  | `analysis.py`   |
| 4  | Posterior age distribution (Bayesian)| `modelisation.py`   |
| 5  | Variance Inflation Factor (VIF)  | `fanti_multiple_regression.py`  |
| 6  | MCMC trace and convergence (Bayesian inference)  | `modelisation.py`   |

All figures are saved at 300 dpi and exported in PNG format.

## 🧪 Requirements

To run the code, install the required Python packages:

```bash
pip install -r requirements.txt
```

Basic packages:
- `numpy`
- `scipy`
- `matplotlib`
- `arviz`
- `pandas`
- `scikit-learn`
- `seaborn`

## ▶️ How to Run

Example:

```bash
python scripts/modeling/modelisation.py
```

Figures will be saved automatically in the working directory. See individual scripts for more options.

## 📜 License

MIT License — see the [LICENSE](LICENSE) file.

## 📬 Contact

Michel Bakhtaoui  
Aix-Marseille Université  
[ORCID: 0009-0006-6710-7787](https://orcid.org/0009-0006-6710-7787)
