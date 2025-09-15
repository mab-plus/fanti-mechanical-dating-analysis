# Standard imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from scipy.stats import zscore

# sklearn imports
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning

# Local imports
from fanti_multiple_regression import FantiMultipleRegression
from fanti_uncertainty_propagation import FantiUncertaintyPropagation
from fanti_power_analysis import PowerAnalysis
from fanti_cross_validation import CrossValidation
from fanti_crossed_analysis import FantiCrossedAnalysis

# 1) PDF vectoriel avec texte "vrai" (polices embarquées)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'

def save_figure(fig, basename, kind='combo', width_mm=90, height_mm=None, min_height_in=3.0):
    w_in = width_mm / 25.4
    w0, h0 = fig.get_size_inches()
    if height_mm is not None:
        h_in = height_mm / 25.4
    else:
        aspect = (h0 / w0) if w0 else 0.75
        h_in = max(w_in * aspect, min_height_in)   # <-- évite les panneaux “écrasés”
    # fig.set_size_inches(w_in, h_in, forward=True)
    fig.tight_layout()
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")             # PDF vectoriel
    dpi = 1000 if kind == 'lineart' else 600
    fig.savefig(f"{basename}.png", dpi=dpi, bbox_inches="tight")    # PNG conforme

# --------------------------------------------------------------
# 1) POSSIBLE ADDITION: Simple outlier management (example)
# --------------------------------------------------------------

def detect_outliers_zscore(X, y, threshold=2.5):
    """
    Simple example of outlier detection based on residual z-scores
    from a multiple linear regression (OLS).
    Returns: masks (bool) for outliers / inliers
    """
    # Quick OLS fit
    # Add column of 1s for intercept:
    X_ones = np.column_stack((np.ones(len(X)), X))
    beta, _, _, _ = np.linalg.lstsq(X_ones, y, rcond=None)
    y_pred = X_ones @ beta
    residuals = y - y_pred

    # Calculate residual z-scores
    zs = zscore(residuals)
    outliers_mask = np.abs(zs) > threshold
    inliers_mask = ~outliers_mask
    return outliers_mask, inliers_mask

def detect_outliers_ransac(X, y):
    """
    Example using RANSAC to find inliers/outliers.
    Returns two boolean masks: inliers/outliers.
    """
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask
    return outlier_mask, inlier_mask

# --------------------------------------------------------------
# 2) POSSIBLE ADDITION: Cross-validation for multiple regression
# --------------------------------------------------------------

def multiple_regression_cross_val(X, y, n_splits=3):
    """
    Performs KFold cross-validation on OLS multiple regression
    and returns mean R² (or MSE) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # OLS fit
        X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))
        beta, _, _, _ = np.linalg.lstsq(X_train_ones, y_train, rcond=None)

        # Prediction
        X_test_ones = np.column_stack((np.ones(len(X_test)), X_test))
        y_pred = X_test_ones @ beta

        # Calculate R²
        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2) + 1e-10
        r2 = 1 - ss_res/ss_tot
        scores.append(r2)

    return np.mean(scores), np.std(scores)

# --------------------------------------------------------------
# 3) POSSIBLE ADDITION: Extended Monte Carlo (perturb both X,y)
# --------------------------------------------------------------

def monte_carlo_data_perturbation(X, y,
                                  n_sim=5000,
                                  x_std=0.01,  # 'Fictitious' standard deviation for X variables
                                  y_std=20.0,  # 'Fictitious' standard deviation for dates in y
                                  random_state=42):
    """
    Generates simulated samples by perturbing X and y around observed
    values, then calculates the distribution of predicted dates (or R²).
    x_std, y_std: control the magnitude of perturbations.
    Returns list of simulated R² values, for example.
    """
    rng = np.random.default_rng(seed=random_state)
    n = len(y)
    r2_list = []

    for _ in range(n_sim):
        # Create perturbed X' and y'
        X_pert = X + rng.normal(loc=0.0, scale=x_std, size=X.shape)
        y_pert = y + rng.normal(loc=0.0, scale=y_std, size=n)

        # Fit OLS
        X_ones = np.column_stack((np.ones(len(X_pert)), X_pert))
        beta, _, _, _ = np.linalg.lstsq(X_ones, y_pert, rcond=None)

        # Prediction & R²
        y_pred = X_ones @ beta
        ss_res = np.sum((y_pert - y_pred)**2)
        ss_tot = np.sum((y_pert - np.mean(y_pert))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        r2_list.append(r2)

    return r2_list

# --------------------------------------------------------------
# Main file to orchestrate analyses
# --------------------------------------------------------------

def main():
    # --- 1) Data from Fanti's article (Table 2) ---
    samples = {
        'B':   {'date': 2000, 'sigma_r': 1076, 'Ef': 24.8, 'Ei': 32.2, 'eta_d': 4.8,  'eta_i': 1.6},
        'DII': {'date': 1000, 'sigma_r': 678,  'Ef': 19.0, 'Ei': 23.3, 'eta_d': 5.3,  'eta_i': 3.3},
        'D':   {'date': 575,  'sigma_r': 63.2,  'Ef': 4.20, 'Ei': 5.36, 'eta_d': 7.4,  'eta_i': 5.2},
        'FII': {'date': 65,   'sigma_r': 150,   'Ef': 7.38, 'Ei': 9.67, 'eta_d': 7.9,  'eta_i': 3.7},
        'NII': {'date': -250, 'sigma_r': 119,   'Ef': 4.55, 'Ei': 6.88, 'eta_d': 8.0,  'eta_i': 4.6},
        'E':   {'date': -400, 'sigma_r': 140,   'Ef': 4.34, 'Ei': 2.98, 'eta_d': 8.5,  'eta_i': 3.3}
        # ... etc., according to your needs ...
    }

    # Build X and y for multiple regression, e.g.
    # Variables: ln(sigma_r), ln(Ei), eta_i (standard example)
    # Ignore or skip if data is not available
    data_list = []
    for s in samples.values():
        # Filter missing data if necessary
        data_list.append([
            np.log(s['sigma_r']),
            np.log(s['Ei']),
            s['eta_i']
        ])
    X = np.array(data_list)
    y = np.array([s['date'] for s in samples.values()])

    # --- 2) Example of outlier exclusion
    outliers_mask, inliers_mask = detect_outliers_ransac(X, y)
    print(f"Number of outliers detected (RANSAC): {np.sum(outliers_mask)}")
    X_in = X[inliers_mask]
    y_in = y[inliers_mask]

# --- 3) Multiple regression + print results
    mult_reg = FantiMultipleRegression(samples)
    # We haven't modified the class, but if you want to restrict to inliers:
    mult_reg.X = np.column_stack((np.ones(len(X_in)), X_in[:,0], X_in[:,1], X_in[:,2]))
    mult_reg.y = y_in

    mult_reg.fit()
    mult_reg.print_results()
    mult_reg.plot_fit()

# --- 4) Cross-validation on multiple regression
    mean_r2_cv, std_r2_cv = multiple_regression_cross_val(X_in, y_in, n_splits=2)
    print(f"Mean R² in CV (inliers): {mean_r2_cv:.3f} ± {std_r2_cv:.3f}")

# --- 5) Power analysis on reduced sample
    power = PowerAnalysis(n_samples=len(y_in))
    power.print_results()

# --- 6) Example of extended Monte Carlo analysis
    # Perturb X and y around x_std=0.05, y_std=50 e.g. (adjust as needed)
    r2_distrib = monte_carlo_data_perturbation(X_in, y_in, n_sim=1000,
                                               x_std=0.05, y_std=50.0)
    print(f"Mean R² via Monte Carlo data-perturbation: {np.mean(r2_distrib):.3f}")


# --- 7) Additional cross-analysis ---
    print("\n=== In-depth cross-validation analysis ===")
    try:
        # Analysis with FantiCrossedAnalysis on unfiltered data
        print("\nResults on complete data:")
        crossed_analysis_full = FantiCrossedAnalysis(X, y, k_folds=min(5, len(X)-1))
        results_full = crossed_analysis_full.run_complete_analysis()
        crossed_analysis_full.print_results()

        # Analysis with FantiCrossedAnalysis on filtered data
        if len(X_in) >= 3:  # Check that enough data remains
            print("\nResults on filtered data (RANSAC):")
            crossed_analysis_in = FantiCrossedAnalysis(X_in, y_in, k_folds=min(5, len(X_in)-1))
            results_in = crossed_analysis_in.run_complete_analysis()
            crossed_analysis_in.print_results()
        else:
            print("\nToo few samples after filtering for in-depth cross-analysis")

    except Exception as e:
        print(f"Error during in-depth cross-analysis: {str(e)}")

    # Small distribution illustration
    plt.hist(r2_distrib, bins=30, alpha=0.6, color='b')
    plt.title("R² Distribution when perturbing X and y (Monte Carlo)")
    plt.xlabel("R²")
    plt.ylabel("Frequency")

    fig = plt.gcf()
    save_figure(fig, "Fig2_predictive_performance_cv_rmse", kind='lineart', width_mm=90)

    plt.close()

from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main()
