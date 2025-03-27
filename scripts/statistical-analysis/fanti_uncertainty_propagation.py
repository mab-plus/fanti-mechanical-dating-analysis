import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

class FantiUncertaintyPropagation:
    def __init__(self, mult_reg_model, turin_values, n_simulations=10000):
        """
        mult_reg_model : instance de FantiMultipleRegression déjà ajustée.
        turin_values   : dict des valeurs (transformées) pour le Suaire.
        n_simulations  : nb de simulations Monte Carlo.
        """
        self.model = mult_reg_model
        self.turin_values = turin_values
        self.n_sim = n_simulations

        if self.model.beta is None or self.model.y_pred is None:
            raise ValueError("Le modèle mult_reg_model doit être ajusté (fit()) avant.")

        self.X = self.model.X  # déjà sans la colonne de 1
        self.y = self.model.y
        self.beta_hat = self.model.beta
        self.residuals = self.model.residuals
        self.n, self.p = self.X.shape

        ss_res = np.sum(self.residuals**2)
        self.s2 = ss_res / (self.n - self.p - 1)  # variance résiduelle
        self.XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.x_turin = self._build_turin_vector()

        # Test de normalité des résidus
        self.shapiro_pvalue = stats.shapiro(self.residuals)[1]

        self.simulated_dates_parametric = None
        self.simulated_dates_residual = None
        self.simulated_dates_data = None

    def _build_turin_vector(self):
        """
        Construit le vecteur [ln(sigma_r), ln(Ei), eta_i] pour le Suaire,
        en lui ajoutant un 1 éventuellement en externe.
        """
        x1 = self.turin_values.get('Breaking Strength')  # déjà en ln ?
        x2 = self.turin_values.get('Inverse Young Modulus')
        x3 = self.turin_values.get('Inverse Loss Factor')
        if any(val is None for val in [x1, x2, x3]):
            raise ValueError("turin_values doit contenir Breaking Strength, Inverse Young Modulus, Inverse Loss Factor.")
        return np.array([x1, x2, x3])

    def run_parametric_monte_carlo(self):
        """
        Génère des échantillons de beta (coeffs) selon une distribution
        multivariée normale, puis prédit la date du Suaire.
        """
        X_const = np.column_stack((np.ones(len(self.X)), self.X))
        cov_beta = self.s2 * np.linalg.inv(X_const.T @ X_const)
        # rendre la matrice symétrique si besoin
        cov_beta = (cov_beta + cov_beta.T) / 2
        min_eig = np.min(np.linalg.eigvals(cov_beta))
        if min_eig < 0:
            cov_beta -= 10*min_eig * np.eye(*cov_beta.shape)

        beta_samples = np.random.multivariate_normal(mean=self.beta_hat, cov=cov_beta, size=self.n_sim)
        # On construit x_turin avec un 1 pour l'intercept
        x_turin_const = np.insert(self.x_turin, 0, 1.0)

        # On obtient la date prédite pour chaque tirage
        self.simulated_dates_parametric = np.dot(beta_samples, x_turin_const)

    def summarize_parametric(self, alpha=0.05):
        """
        Résume la distribution paramétrique simulée.
        """
        if self.simulated_dates_parametric is None:
            raise RuntimeError("Appelez run_parametric_monte_carlo() avant.")
        return self._summary_stats(self.simulated_dates_parametric, alpha)

    def run_data_perturbation_monte_carlo(self, x_std=0.01, y_std=20.0, seed=42):
        """
        Exemple d’extension : On perturbe X et y (bruit gaussien),
        on ré-estime le modèle, et on calcule la date du Suaire à chaque simulation.
        """
        rng = np.random.default_rng(seed)
        X_const_original = np.column_stack((np.ones(len(self.X)), self.X))
        n = len(self.y)
        simulated_dates = []

        for _ in range(self.n_sim):
            # Perturbations
            X_pert = self.X + rng.normal(0, x_std, size=self.X.shape)
            y_pert = self.y + rng.normal(0, y_std, size=n)

            X_pert_const = np.column_stack((np.ones(n), X_pert))
            beta_hat_pert, _, _, _ = np.linalg.lstsq(X_pert_const, y_pert, rcond=None)

            # On prédit la date du Suaire
            x_turin_const = np.insert(self.x_turin, 0, 1.0)
            date_pert = x_turin_const @ beta_hat_pert
            simulated_dates.append(date_pert)

        self.simulated_dates_data = np.array(simulated_dates)

    def summarize_data_perturbation(self, alpha=0.05):
        if self.simulated_dates_data is None:
            raise RuntimeError("Appelez run_data_perturbation_monte_carlo() avant.")
        return self._summary_stats(self.simulated_dates_data, alpha)

    def _summary_stats(self, samples, alpha=0.05):
        mean_ = np.mean(samples)
        std_ = np.std(samples, ddof=1)
        lower_q = 100*(alpha/2)
        upper_q = 100*(1 - alpha/2)
        ci_lower, ci_upper = np.percentile(samples, [lower_q, upper_q])
        return {
            'mean': mean_,
            'std': std_,
            'ci': (ci_lower, ci_upper),
            'level': 100*(1-alpha)
        }

    def print_diagnostics(self):
        print("=== Diagnostics de l'Incertitude ===")
        print(f"Test de normalité des résidus (Shapiro-Wilk) : p-value = {self.shapiro_pvalue:.3f}")
