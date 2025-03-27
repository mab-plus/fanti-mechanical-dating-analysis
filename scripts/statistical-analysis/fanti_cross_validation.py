import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error

class CrossValidation:
    def __init__(self, X, y, k_folds=5):
        """
        Classe générique pour la validation croisée (LOO et K-Fold).
        X, y : array numpy complets pour la régression multiple (ou simple).
        k_folds : nb de plis pour K-Fold
        """
        self.X = X
        self.y = y
        self.k_folds = k_folds

    def leave_one_out_cv(self):
        """
        Validation croisée Leave-One-Out en ajustant un modèle OLS
        sur (X_train, y_train).
        Retourne la MSE moyenne, l’écart-type et la stabilité des beta si besoin.
        """
        loo = LeaveOneOut()
        mse_scores = []
        beta_values = []

        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Ajout d'une colonne de 1 pour l'intercept
            X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))
            beta, _, _, _ = np.linalg.lstsq(X_train_ones, y_train, rcond=None)
            beta_values.append(beta)

            X_test_ones = np.column_stack((np.ones(len(X_test)), X_test))
            y_pred = X_test_ones @ beta
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        beta_values = np.array(beta_values)
        beta_stability = np.std(beta_values, axis=0)

        return {
            'mean_cv_error': np.mean(mse_scores),
            'std_cv_error': np.std(mse_scores),
            'cv_scores': mse_scores,
            'beta_stability': beta_stability
        }

    def k_fold_cv(self):
        """
        Validation croisée K-Fold en ajustant un modèle OLS
        sur (X_train, y_train).
        Retourne la MSE moyenne et son écart-type.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        mse_scores = []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))
            beta, _, _, _ = np.linalg.lstsq(X_train_ones, y_train, rcond=None)

            X_test_ones = np.column_stack((np.ones(len(X_test)), X_test))
            y_pred = X_test_ones @ beta

            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

        return {
            'mean_cv_error': np.mean(mse_scores),
            'std_cv_error': np.std(mse_scores),
            'cv_scores': mse_scores
        }

    def print_results(self, loo_results, kfold_results):
        """
        Fonction pour imprimer de façon compacte les résultats de LOO et K-Fold.
        """
        print("=== Validation Croisée Leave-One-Out ===")
        print(f"Erreur moyenne (MSE) : {loo_results['mean_cv_error']:.2f}")
        print(f"Écart-type           : {loo_results['std_cv_error']:.2f}")
        print("Stabilité des coefficients (std) :")
        for i, std in enumerate(loo_results['beta_stability']):
            print(f"  β{i}: {std:.5f}")

        print("\n=== Validation Croisée K-Fold ===")
        print(f"Erreur moyenne (MSE) : {kfold_results['mean_cv_error']:.2f}")
        print(f"Écart-type           : {kfold_results['std_cv_error']:.2f}")
        print("Scores MSE individuels :", kfold_results['cv_scores'])
