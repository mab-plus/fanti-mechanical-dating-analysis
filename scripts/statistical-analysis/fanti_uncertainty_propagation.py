import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

class FantiUncertaintyPropagation:
    def __init__(self, mult_reg_model, turin_values, n_simulations=10000):
        """
        mult_reg_model : instance of FantiMultipleRegression already fitted.
        turin_values   : dict of (transformed) values for the Shroud.
        n_simulations  : number of Monte Carlo simulations.
        """
        self.model = mult_reg_model
        self.turin_values = turin_values
        self.n_sim = n_simulations

        if self.model.beta is None or self.model.y_pred is None:
            raise ValueError("The mult_reg_model must be fitted (fit()) before.")

        self.X = self.model.X  # already without the column of 1s
        self.y = self.model.y
        self.beta_hat = self.model.beta
        self.residuals = self.model.residuals
        self.n, self.p = self.X.shape

        ss_res = np.sum(self.residuals**2)
        self.s2 = ss_res / (self.n - self.p - 1)  # residual variance
        self.XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.x_turin = self._build_turin_vector()

        # Residual normality test
        self.shapiro_pvalue = stats.shapiro(self.residuals)[1]

        self.simulated_dates_parametric = None
        self.simulated_dates_residual = None
        self.simulated_dates_data = None

    def _build_turin_vector(self):
        """
        Builds the vector [ln(sigma_r), ln(Ei), eta_i] for the Shroud,
        eventually adding a 1 externally.
        """
        x1 = self.turin_values.get('Breaking Strength')  # already in ln?
        x2 = self.turin_values.get('Inverse Young Modulus')
        x3 = self.turin_values.get('Inverse Loss Factor')
        if any(val is None for val in [x1, x2, x3]):
            raise ValueError("turin_values must contain Breaking Strength, Inverse Young Modulus, Inverse Loss Factor.")
        return np.array([x1, x2, x3])

    def run_parametric_monte_carlo(self):
        """
        Generates beta (coeffs) samples according to a multivariate
        normal distribution, then predicts the Shroud date.
        """
        X_const = np.column_stack((np.ones(len(self.X)), self.X))
        cov_beta = self.s2 * np.linalg.inv(X_const.T @ X_const)
        # make the matrix symmetric if needed
        cov_beta = (cov_beta + cov_beta.T) / 2
        min_eig = np.min(np.linalg.eigvals(cov_beta))
        if min_eig < 0:
            cov_beta -= 10*min_eig * np.eye(*cov_beta.shape)

        beta_samples = np.random.multivariate_normal(mean=self.beta_hat, cov=cov_beta, size=self.n_sim)
        # Build x_turin with a 1 for the intercept
        x_turin_const = np.insert(self.x_turin, 0, 1.0)

        # Get the predicted date for each draw
        self.simulated_dates_parametric = np.dot(beta_samples, x_turin_const)

    def summarize_parametric(self, alpha=0.05):
        """
        Summarizes the simulated parametric distribution.
        """
        if self.simulated_dates_parametric is None:
            raise RuntimeError("Call run_parametric_monte_carlo() first.")
        return self._summary_stats(self.simulated_dates_parametric, alpha)

    def run_data_perturbation_monte_carlo(self, x_std=0.01, y_std=20.0, seed=42):
        """
        Extension example: We perturb X and y (Gaussian noise),
        re-estimate the model, and calculate the Shroud date for each simulation.
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

            # Predict the Shroud date
            x_turin_const = np.insert(self.x_turin, 0, 1.0)
            date_pert = x_turin_const @ beta_hat_pert
            simulated_dates.append(date_pert)

        self.simulated_dates_data = np.array(simulated_dates)

    def summarize_data_perturbation(self, alpha=0.05):
        if self.simulated_dates_data is None:
            raise RuntimeError("Call run_data_perturbation_monte_carlo() first.")
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
        print("=== Uncertainty Diagnostics ===")
        print(f"Residual normality test (Shapiro-Wilk): p-value = {self.shapiro_pvalue:.3f}")
