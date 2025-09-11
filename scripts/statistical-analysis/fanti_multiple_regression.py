import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

class FantiMultipleRegression:
    def __init__(self, samples, robust=False, weights=None):
        """
        robust  : if True, uses RLM (M-estimator method) instead of OLS.
        weights : optional weight array for weighted regression (WLS).
        """
        self.samples = samples
        self.y = np.array([s['date'] for s in samples.values()])
        self.X1 = np.log(np.array([s['sigma_r'] for s in samples.values()]))
        self.X2 = np.log(np.array([s['Ei'] for s in samples.values()]))
        self.X3 = np.array([s['eta_i'] for s in samples.values()])
        self.X = np.column_stack((self.X1, self.X2, self.X3))

        self.beta = None
        self.y_pred = None
        self.residuals = None
        self.R2 = None
        self.vif = None
        self.bp_pvalue = None
        self.robust = robust
        self.weights = weights

    def fit(self):
        """
        If robust=True => uses RLM,
        Otherwise => simple OLS or WLS if weights is defined.
        """
        X_const = add_constant(self.X)
        if self.robust:
            model = RLM(self.y, X_const)  # robust linear model
            results = model.fit()
        elif self.weights is not None:
            # Weighted LS => WLS
            from statsmodels.regression.linear_model import WLS
            model = WLS(self.y, X_const, weights=self.weights)
            results = model.fit()
        else:
            model = OLS(self.y, X_const)
            results = model.fit()

        self.beta = results.params
        self.y_pred = results.predict(X_const)
        self.residuals = self.y - self.y_pred
        self.R2 = results.rsquared if not self.robust else 1 - results.ssr / np.sum((self.y - np.mean(self.y))**2)

        # Breusch-Pagan test
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(results.resid, X_const)
        self.bp_pvalue = bp_pvalue

        # VIF calculation for X variables
        self.vif = {}
        for i in range(1, X_const.shape[1]):  # exclude the column of 1s
            vif_val = variance_inflation_factor(X_const, i)
            self.vif[f'VIF_{i}'] = vif_val

    def print_results(self):
        if self.beta is None:
            print("Model not fitted yet. Call the fit() method.")
            return
        print("=== Multiple Regression ===")
        if self.robust:
            print("Mode: RLM (robust M-estimator)")
        elif self.weights is not None:
            print("Mode: WLS (weighted least squares)")
        else:
            print("Mode: Classic OLS")

        print(f"Coefficients (intercept + X): {self.beta}")
        print(f"Global R²                   : {self.R2:.3f}")
        print(f"Breusch-Pagan p-value       : {self.bp_pvalue:.3f}")
        for var, vif_value in self.vif.items():
            print(f"{var}: {vif_value:.3f}")

    def plot_fit(self):
        if self.y_pred is None:
            print("Model not fitted yet. Call the fit() method.")
            return
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y, self.y_pred, color='blue', label="Predictions")
        min_val = min(min(self.y), min(self.y_pred))
        max_val = max(max(self.y), max(self.y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Identity line")
        plt.xlabel("Observed dates")
        plt.ylabel("Predicted dates")
        plt.title("Observed vs Predicted Comparison (Multiple Regression)")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("figure2.png", dpi=300, bbox_inches="tight")
        plt.close()

        # noms/valeurs
        names  = [r'$\ln(\sigma_r)$', r'$\ln(E_f)$', r'$\eta_i$']
        values = list(self.vif.values())

        fig, ax = plt.subplots(figsize=(6, 4))

        # barres
        bars = ax.bar(names, values, color='skyblue')

        # limites Y (méthode Axes, pas plt.set_ylim)
        ax.set_ylim(0, 1.15 * max(values))

        # seuils VIF
        ax.axhline(4,  ls='--', lw=1,  color='0.4')
        ax.axhline(10, ls='--', lw=1.5, color='r', dashes=(6, 3), label='VIF = 10')

        # labels au-dessus des barres (padding en points)
        ax.bar_label(bars, fmt='%.1f', padding=3)

        # habillage
        ax.set_ylabel("VIF")
        ax.set_title("Variance Inflation Factors (VIF)")
        handles = [Line2D([0],[0], ls='--', color='0.4'), Line2D([0],[0], ls='--', color='r', dashes=(6,3))]
        labels  = ['VIF = 4 (investigate)', 'VIF = 10 (serious)']
        ax.legend(handles, labels, title='Thresholds')

        fig.tight_layout()
        fig.savefig("figure5.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

