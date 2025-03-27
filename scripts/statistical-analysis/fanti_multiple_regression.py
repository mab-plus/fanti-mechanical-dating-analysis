import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

class FantiMultipleRegression:
    def __init__(self, samples, robust=False, weights=None):
        """
        robust  : si True, on utilise RLM (méthode M-estimator) au lieu de OLS.
        weights : array de poids optionnel pour une régression pondérée (WLS).
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
        Si robust=True => utilise RLM,
        Sinon => OLS simple ou WLS si weights est défini.
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

        # Test de Breusch-Pagan
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(results.resid, X_const)
        self.bp_pvalue = bp_pvalue

        # Calcul du VIF pour les variables X
        self.vif = {}
        for i in range(1, X_const.shape[1]):  # on exclut la colonne de 1
            vif_val = variance_inflation_factor(X_const, i)
            self.vif[f'VIF_{i}'] = vif_val

    def print_results(self):
        if self.beta is None:
            print("Le modèle n'est pas encore ajusté. Appelez la méthode fit().")
            return
        print("=== Régression Multiple ===")
        if self.robust:
            print("Mode : RLM (robust M-estimator)")
        elif self.weights is not None:
            print("Mode : WLS (weighted least squares)")
        else:
            print("Mode : OLS classique")

        print(f"Coefficients (intercept + X) : {self.beta}")
        print(f"R² global                   : {self.R2:.3f}")
        print(f"Breusch-Pagan p-value       : {self.bp_pvalue:.3f}")
        for var, vif_value in self.vif.items():
            print(f"{var} : {vif_value:.3f}")

    def plot_fit(self):
        if self.y_pred is None:
            print("Le modèle n'est pas encore ajusté. Appelez la méthode fit().")
            return
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y, self.y_pred, color='blue', label="Prédictions")
        min_val = min(min(self.y), min(self.y_pred))
        max_val = max(max(self.y), max(self.y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ligne identité")
        plt.xlabel("Dates observées")
        plt.ylabel("Dates prédites")
        plt.title("Comparaison Observé vs Prévu (Régression Multiple)")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("figure2.png", dpi=300, bbox_inches="tight")
        plt.close()

        names = list(self.vif.keys())
        values = list(self.vif.values())

        plt.figure(figsize=(6, 4))
        bars = plt.bar(names, values, color='skyblue')
        plt.axhline(y=10, color='red', linestyle='--', label='VIF = 10 threshold')
        plt.ylabel("VIF")
        plt.title("Variance Inflation Factors (VIF)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure5.png", dpi=300, bbox_inches="tight")
        plt.close()
