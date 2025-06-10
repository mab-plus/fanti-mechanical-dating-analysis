import numpy as np
from statsmodels.stats.power import TTestIndPower

class PowerAnalysis:
    def __init__(self, n_samples=13, effect_size=0.8, target_power=0.8, alpha=0.05):
        """
        Class to estimate statistical power (T-test)
        and required number of samples.
        n_samples   : current number of samples
        effect_size : assumed effect size (Cohen's d)
        target_power: target power (e.g. 0.8)
        alpha       : Type I error risk
        """
        self.n = n_samples
        self.effect_size = effect_size
        self.target_power = target_power
        self.alpha = alpha
        self.power_analysis = TTestIndPower()

    def analyze(self):
        current_power = self.power_analysis.solve_power(
            effect_size=self.effect_size,
            nobs1=self.n,
            alpha=self.alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        required_n = self.power_analysis.solve_power(
            effect_size=self.effect_size,
            power=self.target_power,
            alpha=self.alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        # For multiple regression, we can approx. lower the effect_size
        required_n_reg = self.power_analysis.solve_power(
            effect_size=self.effect_size / np.sqrt(3),
            power=self.target_power,
            alpha=self.alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        return {
            'current_power': current_power,
            'required_n': required_n,
            'required_n_reg': required_n_reg,
            'power_deficit': self.target_power - current_power
        }

    def print_results(self):
        results = self.analyze()
        print("=== Power Analysis ===")
        print(f"Current power (n={self.n}): {results['current_power']:.3f}")
        print(f"Sample size required to reach {self.target_power*100}%: {results['required_n']:.0f}")
        print(f"Power deficit: {results['power_deficit']:.3f}")
        print(f"Minimum number (~) of samples for multiple regression: {results['required_n_reg']:.0f}")
