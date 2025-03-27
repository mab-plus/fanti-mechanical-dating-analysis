import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FantiExperiment:
    def __init__(self, samples, detect_outliers=False):
        """
        detect_outliers : si True, on appliquera un petit filtrage
        d'outliers basé sur z-score pour chaque régression simple.
        """
        self.samples = samples
        self.dates = np.array([s['date'] for s in samples.values()])
        self.parameters = {
            'Breaking Strength': np.log(np.array([s['sigma_r'] for s in samples.values()])),
            'Final Young Modulus': np.log(np.array([s['Ef'] for s in samples.values()])),
            'Inverse Young Modulus': np.log(np.array([s['Ei'] for s in samples.values()])),
            'Direct Loss Factor': np.array([s['eta_d'] for s in samples.values()]),
            'Inverse Loss Factor': np.array([s['eta_i'] for s in samples.values()])
        }
        self.detect_outliers = detect_outliers
        self.results = {}

    def perform_regression(self):
        for name, data in self.parameters.items():
            # On détecte éventuellement les outliers avant la régression
            if self.detect_outliers:
                data, date_filtered = self._filter_outliers(data, self.dates)
            else:
                date_filtered = self.dates

            slope, intercept, r_value, p_value, stderr = stats.linregress(data, date_filtered)
            residuals = date_filtered - (slope * data + intercept)
            w_stat, p_shapiro = stats.shapiro(residuals)
            half = len(data) // 2
            _, p_levene = stats.levene(data[:half], data[half:])

            X_bp = add_constant(data)
            lm = OLS(date_filtered, X_bp).fit()
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(lm.resid, X_bp)
            vif = variance_inflation_factor(X_bp, 1)

            self.results[name] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'stderr': stderr,
                'residuals': residuals,
                'p_shapiro': p_shapiro,
                'p_levene': p_levene,
                'bp_pvalue': bp_pvalue,
                'vif': vif
            }

    def _filter_outliers(self, x, y, z_thresh=2.5):
        """
        Petit helper pour enlever des outliers en régression simple.
        On fait d'abord un fit, on calcule les résidus, on vire ceux
        au-dessus d'un certain seuil en z-score.
        """
        slope, intercept, _, _, _ = stats.linregress(x, y)
        res = y - (slope*x + intercept)
        zs = np.abs(stats.zscore(res))
        mask_inliers = zs < z_thresh
        return x[mask_inliers], y[mask_inliers]

    def print_regression_results(self):
        for name, res in self.results.items():
            print(f"--- Paramètre : {name} ---")
            print(f"  Pente               : {res['slope']:.3f}")
            print(f"  Ordonnée à l'origine: {res['intercept']:.3f}")
            print(f"  R²                  : {res['r_squared']:.3f}")
            print(f"  p-value (régression): {res['p_value']:.3f}")
            print(f"  Erreur type         : {res['stderr']:.3f}")
            print(f"  Shapiro-Wilk p-value (normalité résidus): {res['p_shapiro']:.3f}")
            print(f"  Levene p-value (homoscédasticité)       : {res['p_levene']:.3f}")
            print(f"  Breusch-Pagan p-value                   : {res['bp_pvalue']:.3f}")
            print(f"  VIF (Multicolinéarité)                  : {res['vif']:.3f}")
            print("-" * 40)

    def plot_regressions(self, turin_values=None):
        """
        Exemple de tracé. turin_values : dict optionnel pour rajouter un point
        """
        n_params = len(self.parameters)
        fig, axs = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axs = [axs]
        for i, (name, data) in enumerate(self.parameters.items()):
            ax = axs[i]
            slope = self.results[name]['slope']
            intercept = self.results[name]['intercept']
            ax.scatter(data, self.dates, color='blue', label='Données')
            x_range = np.linspace(np.min(data), np.max(data), 100)
            y_pred = slope * x_range + intercept
            ax.plot(x_range, y_pred, color='red', label=f"R²={self.results[name]['r_squared']:.2f}")
            if turin_values and name in turin_values:
                val = turin_values[name]
                ax.scatter(val, slope*val + intercept,
                           color='green', marker='*', s=150, label='Suaire')
            ax.set_xlabel(f"{name} (transformé)")
            ax.set_ylabel("Date (AD)")
            ax.set_title(name)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig("figure1.png", dpi=300, bbox_inches="tight")
        plt.close()
