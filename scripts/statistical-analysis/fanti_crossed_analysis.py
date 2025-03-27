import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class FantiCrossedAnalysis:
    def __init__(self, X, y, k_folds=5, test_size=0.2, random_state=42):
        """
        Analyse croisée complète du modèle de Fanti.

        Parameters:
        -----------
        X : array-like
            Variables prédictives (ln(σr), ln(Ei), ηi)
        y : array-like
            Dates à prédire
        k_folds : int
            Nombre de plis pour k-fold CV
        test_size : float
            Proportion de données pour le test set
        random_state : int
            Pour la reproductibilité
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.k_folds = k_folds
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

        # Résultats stockés
        self.kfold_scores = None
        self.loo_scores = None
        self.holdout_score = None

    def _fit_and_score(self, X_train, X_test, y_train, y_test):
        """Helper pour ajuster et évaluer le modèle"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Ajout intercept
        X_train_int = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_test_int = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

        # Régression
        beta = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]
        y_pred = X_test_int @ beta

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return {'rmse': rmse, 'r2': r2, 'beta': beta}

    def k_fold_analysis(self):
        """K-fold cross-validation"""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            fold_scores = self._fit_and_score(X_train, X_test, y_train, y_test)
            scores.append(fold_scores)

        self.kfold_scores = scores
        return scores

    def leave_one_out_analysis(self):
        """Leave-one-out cross-validation"""
        loo = LeaveOneOut()
        scores = []

        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            fold_scores = self._fit_and_score(X_train, X_test, y_train, y_test)
            scores.append(fold_scores)

        self.loo_scores = scores
        return scores

    def holdout_analysis(self):
        """Validation sur échantillon test indépendant"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.holdout_score = self._fit_and_score(X_train, X_test, y_train, y_test)
        return self.holdout_score

    def run_complete_analysis(self):
        """Exécute les trois types de validation croisée"""
        kfold = self.k_fold_analysis()
        loo = self.leave_one_out_analysis()
        holdout = self.holdout_analysis()

        return {
            'k_fold': kfold,
            'loo': loo,
            'holdout': holdout
        }

    def print_results(self):
        """Affiche un résumé des résultats"""
        if not all([self.kfold_scores, self.loo_scores, self.holdout_score]):
            self.run_complete_analysis()

        print("=== Résultats de la Validation Croisée ===")

        print("\nK-Fold CV:")
        rmse_kf = np.mean([s['rmse'] for s in self.kfold_scores])
        r2_kf = np.mean([s['r2'] for s in self.kfold_scores])
        print(f"RMSE: {rmse_kf:.1f} ± {np.std([s['rmse'] for s in self.kfold_scores]):.1f}")
        print(f"R²: {r2_kf:.3f} ± {np.std([s['r2'] for s in self.kfold_scores]):.3f}")

        print("\nLeave-One-Out CV:")
        rmse_loo = np.mean([s['rmse'] for s in self.loo_scores])
        r2_loo = np.mean([s['r2'] for s in self.loo_scores])
        print(f"RMSE: {rmse_loo:.1f} ± {np.std([s['rmse'] for s in self.loo_scores]):.1f}")
        print(f"R²: {r2_loo:.3f} ± {np.std([s['r2'] for s in self.loo_scores]):.3f}")

        print("\nHoldout Validation:")
        print(f"RMSE: {self.holdout_score['rmse']:.1f}")
        print(f"R²: {self.holdout_score['r2']:.3f}")
