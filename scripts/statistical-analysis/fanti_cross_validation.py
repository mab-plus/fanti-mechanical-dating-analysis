import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error

class CrossValidation:
    def __init__(self, X, y, k_folds=5):
        """
        Generic class for cross-validation (LOO and K-Fold).
        X, y : complete numpy arrays for multiple (or simple) regression.
        k_folds : number of folds for K-Fold
        """
        self.X = X
        self.y = y
        self.k_folds = k_folds

    def leave_one_out_cv(self):
        """
        Leave-One-Out cross-validation fitting an OLS model
        on (X_train, y_train).
        Returns average MSE, standard deviation and beta stability if needed.
        """
        loo = LeaveOneOut()
        mse_scores = []
        beta_values = []

        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Add column of 1s for intercept
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
        K-Fold cross-validation fitting an OLS model
        on (X_train, y_train).
        Returns average MSE and its standard deviation.
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
        Function to compactly print LOO and K-Fold results.
        """
        print("=== Leave-One-Out Cross-Validation ===")
        print(f"Mean error (MSE): {loo_results['mean_cv_error']:.2f}")
        print(f"Standard deviation: {loo_results['std_cv_error']:.2f}")
        print("Coefficient stability (std):")
        for i, std in enumerate(loo_results['beta_stability']):
            print(f"  β{i}: {std:.5f}")

        print("\n=== K-Fold Cross-Validation ===")
        print(f"Mean error (MSE): {kfold_results['mean_cv_error']:.2f}")
        print(f"Standard deviation: {kfold_results['std_cv_error']:.2f}")
        print("Individual MSE scores:", kfold_results['cv_scores'])
