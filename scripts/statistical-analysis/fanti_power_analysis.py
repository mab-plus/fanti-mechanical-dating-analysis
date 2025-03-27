import numpy as np
from statsmodels.stats.power import TTestIndPower

class PowerAnalysis:
    def __init__(self, n_samples=13, effect_size=0.8, target_power=0.8, alpha=0.05):
        """
        Classe pour estimer la puissance statistique (test T)
        et le nbre d'échantillons requis.
        n_samples   : nombre d'échantillons actuel
        effect_size : taille d'effet supposée (Cohen's d)
        target_power: puissance visée (ex. 0.8)
        alpha       : risque d'erreur de Type I
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

        # Pour la régression multiple, on peut approx. baisser l'effect_size
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
        print("=== Analyse de Puissance ===")
        print(f"Puissance actuelle (n={self.n}) : {results['current_power']:.3f}")
        print(f"Taille d'échantillon requise pour atteindre {self.target_power*100}% : {results['required_n']:.0f}")
        print(f"Déficit de puissance : {results['power_deficit']:.3f}")
        print(f"Nombre minimal (~) d'échantillons pour la régression multiple : {results['required_n_reg']:.0f}")
