# Imports standards
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import zscore

# Imports sklearn
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning

# Imports locaux
from fanti_experiment import FantiExperiment
from fanti_multiple_regression import FantiMultipleRegression
from fanti_uncertainty_propagation import FantiUncertaintyPropagation
from fanti_power_analysis import PowerAnalysis
from fanti_cross_validation import CrossValidation
from fanti_crossed_analysis import FantiCrossedAnalysis

# --------------------------------------------------------------
# 1) AJOUT POSSIBLE : Gestion simple des outliers (exemple)
# --------------------------------------------------------------

def detect_outliers_zscore(X, y, threshold=2.5):
    """
    Exemple simple de détection d'outliers basé sur le z-score des résidus
    d'une régression linéaire multiple (OLS).
    Return : masques (bool) pour outliers / inliers
    """
    # Ajustement OLS rapide
    # On ajoute la colonne de 1 pour l'intercept :
    X_ones = np.column_stack((np.ones(len(X)), X))
    beta, _, _, _ = np.linalg.lstsq(X_ones, y, rcond=None)
    y_pred = X_ones @ beta
    residuals = y - y_pred

    # Calcul du z-score des résidus
    zs = zscore(residuals)
    outliers_mask = np.abs(zs) > threshold
    inliers_mask = ~outliers_mask
    return outliers_mask, inliers_mask

def detect_outliers_ransac(X, y):
    """
    Exemple d’utilisation de RANSAC pour trouver les inliers/outliers.
    Retourne deux masques booléens inliers/outliers.
    """
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask
    return outlier_mask, inlier_mask

# --------------------------------------------------------------
# 2) AJOUT POSSIBLE : Validation croisée pour la régression multiple
# --------------------------------------------------------------

def multiple_regression_cross_val(X, y, n_splits=3):
    """
    Effectue une validation croisée KFold sur la régression multiple OLS
    et renvoie la moyenne du R² (ou MSE) sur les folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Ajustement OLS
        X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))
        beta, _, _, _ = np.linalg.lstsq(X_train_ones, y_train, rcond=None)

        # Prédiction
        X_test_ones = np.column_stack((np.ones(len(X_test)), X_test))
        y_pred = X_test_ones @ beta

        # On calcule le R²
        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2) + 1e-10
        r2 = 1 - ss_res/ss_tot
        scores.append(r2)

    return np.mean(scores), np.std(scores)

# --------------------------------------------------------------
# 3) AJOUT POSSIBLE : Monte Carlo élargi (perturber aussi X,y)
# --------------------------------------------------------------

def monte_carlo_data_perturbation(X, y,
                                  n_sim=5000,
                                  x_std=0.01,  # Ecart-type 'fictif' pour les variables X
                                  y_std=20.0,  # Ecart-type 'fictif' pour la date en y
                                  random_state=42):
    """
    Génère des échantillons simulés en perturbant X et y autour des valeurs
    observées, puis calcule la distribution des dates prédites (ou du R²).
    x_std, y_std : contrôlent l'ampleur des perturbations.
    Retourne la liste des R² simulés, par ex.
    """
    rng = np.random.default_rng(seed=random_state)
    n = len(y)
    r2_list = []

    for _ in range(n_sim):
        # On crée des X' et y' perturbés
        X_pert = X + rng.normal(loc=0.0, scale=x_std, size=X.shape)
        y_pert = y + rng.normal(loc=0.0, scale=y_std, size=n)

        # Fit OLS
        X_ones = np.column_stack((np.ones(len(X_pert)), X_pert))
        beta, _, _, _ = np.linalg.lstsq(X_ones, y_pert, rcond=None)

        # Prédiction & R²
        y_pred = X_ones @ beta
        ss_res = np.sum((y_pert - y_pred)**2)
        ss_tot = np.sum((y_pert - np.mean(y_pert))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        r2_list.append(r2)

    return r2_list

# --------------------------------------------------------------
# Fichier principal pour orchestrer les analyses
# --------------------------------------------------------------

def main():
    # --- 1) Données issues de l'article de Fanti (Table 2) ---
    samples = {
        'B':   {'date': 2000, 'sigma_r': 1076, 'Ef': 24.8, 'Ei': 32.2, 'eta_d': 4.8,  'eta_i': 1.6},
        'DII': {'date': 1000, 'sigma_r': 678,  'Ef': 19.0, 'Ei': 23.3, 'eta_d': 5.3,  'eta_i': 3.3},
        'D':   {'date': 575,  'sigma_r': 63.2,  'Ef': 4.20, 'Ei': 5.36, 'eta_d': 7.4,  'eta_i': 5.2},
        'FII': {'date': 65,   'sigma_r': 150,   'Ef': 7.38, 'Ei': 9.67, 'eta_d': 7.9,  'eta_i': 3.7},
        'NII': {'date': -250, 'sigma_r': 119,   'Ef': 4.55, 'Ei': 6.88, 'eta_d': 8.0,  'eta_i': 4.6},
        'E':   {'date': -400, 'sigma_r': 140,   'Ef': 4.34, 'Ei': 2.98, 'eta_d': 8.5,  'eta_i': 3.3}
        # ... etc., selon vos besoins ...
    }

    # On construit X et y pour la régression multiple, par ex.
    # Variables: ln(sigma_r), ln(Ei), eta_i (exemple standard)
    # On ignore or on skip si la donnée n'est pas dispo
    data_list = []
    for s in samples.values():
        # On filtre les données manquantes si nécessaire
        data_list.append([
            np.log(s['sigma_r']),
            np.log(s['Ei']),
            s['eta_i']
        ])
    X = np.array(data_list)
    y = np.array([s['date'] for s in samples.values()])

    # --- 2) Exemple d'exclusion d'outliers
    outliers_mask, inliers_mask = detect_outliers_ransac(X, y)
    print(f"Nombre d'outliers détectés (RANSAC) : {np.sum(outliers_mask)}")
    X_in = X[inliers_mask]
    y_in = y[inliers_mask]

# --- 3) Régression multiple + impression des résultats
    mult_reg = FantiMultipleRegression(samples)
    # On n'a pas modifié la classe, mais si vous voulez la restreindre aux inliers :
    mult_reg.X = np.column_stack((np.ones(len(X_in)), X_in[:,0], X_in[:,1], X_in[:,2]))
    mult_reg.y = y_in

    mult_reg.fit()
    mult_reg.print_results()
    mult_reg.plot_fit()

# --- 4) Validation croisée sur la régression multiple
    mean_r2_cv, std_r2_cv = multiple_regression_cross_val(X_in, y_in, n_splits=2)
    print(f"R² moyen en CV (inliers): {mean_r2_cv:.3f} ± {std_r2_cv:.3f}")

# --- 5) Analyse de puissance sur l'échantillon réduit
    power = PowerAnalysis(n_samples=len(y_in))
    power.print_results()

# --- 6) Exemple d'analyse Monte Carlo élargie
    # Perturber X et y autour de x_std=0.05, y_std=50 par ex. (à adapter)
    r2_distrib = monte_carlo_data_perturbation(X_in, y_in, n_sim=1000,
                                               x_std=0.05, y_std=50.0)
    print(f"R² moyen via Monte Carlo data-perturbation : {np.mean(r2_distrib):.3f}")


# --- 7) Analyse croisée additionnelle ---
    print("\n=== Analyse de validation croisée approfondie ===")
    try:
        # Analyse avec FantiCrossedAnalysis sur les données non filtrées
        print("\nRésultats sur données complètes :")
        crossed_analysis_full = FantiCrossedAnalysis(X, y, k_folds=min(5, len(X)-1))
        results_full = crossed_analysis_full.run_complete_analysis()
        crossed_analysis_full.print_results()

        # Analyse avec FantiCrossedAnalysis sur les données filtrées
        if len(X_in) >= 3:  # Vérifie qu'il reste assez de données
            print("\nRésultats sur données filtrées (RANSAC) :")
            crossed_analysis_in = FantiCrossedAnalysis(X_in, y_in, k_folds=min(5, len(X_in)-1))
            results_in = crossed_analysis_in.run_complete_analysis()
            crossed_analysis_in.print_results()
        else:
            print("\nTrop peu d'échantillons après filtrage pour l'analyse croisée approfondie")

    except Exception as e:
        print(f"Erreur lors de l'analyse croisée approfondie : {str(e)}")

    # Petite illustration de distribution
    plt.hist(r2_distrib, bins=30, alpha=0.6, color='b')
    plt.title("Distribution du R² en perturbant X et y (Monte Carlo)")
    plt.xlabel("R²")
    plt.ylabel("Fréquence")
    # plt.show()
    plt.savefig("figure3.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main()
