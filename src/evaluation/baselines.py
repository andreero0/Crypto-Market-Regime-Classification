"""Baseline models for comparison with LSTM"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class BaselineModels:
    """Simple baseline models for performance comparison"""

    @staticmethod
    def random_classifier(y_test, num_classes=3, random_state=42):
        """
        Random predictions baseline.

        Args:
            y_test (np.array): True labels
            num_classes (int): Number of classes
            random_state (int): Random seed

        Returns:
            float: Accuracy score
        """
        np.random.seed(random_state)
        y_pred = np.random.randint(0, num_classes, size=len(y_test))
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    @staticmethod
    def majority_class_classifier(y_train, y_test):
        """
        Always predict majority class baseline.

        Args:
            y_train (np.array): Training labels
            y_test (np.array): Test labels

        Returns:
            tuple: (accuracy, predictions)
        """
        majority_class = np.bincount(y_train).argmax()
        y_pred = np.full(len(y_test), majority_class)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    @staticmethod
    def random_forest_baseline(X_train, y_train, X_test, y_test,
                               n_estimators=100, random_state=42):
        """
        Random Forest baseline.

        Flattens sequence data for traditional ML algorithm.

        Args:
            X_train (np.array): Training sequences (samples, timesteps, features)
            y_train (np.array): Training labels
            X_test (np.array): Test sequences
            y_test (np.array): Test labels
            n_estimators (int): Number of trees
            random_state (int): Random seed

        Returns:
            tuple: (accuracy, predictions, model)
        """
        # Flatten sequences for RF
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        print(f"Training Random Forest with {n_estimators} trees...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train_flat, y_train)

        y_pred = rf.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_pred, rf

    @staticmethod
    def logistic_regression_baseline(X_train, y_train, X_test, y_test,
                                     max_iter=1000, random_state=42):
        """
        Logistic Regression baseline.

        Flattens sequence data for traditional ML algorithm.

        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training labels
            X_test (np.array): Test sequences
            y_test (np.array): Test labels
            max_iter (int): Maximum iterations
            random_state (int): Random seed

        Returns:
            tuple: (accuracy, predictions, model)
        """
        # Flatten sequences
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        print(f"Training Logistic Regression...")
        lr = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1
        )
        lr.fit(X_train_flat, y_train)

        y_pred = lr.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_pred, lr

    @classmethod
    def evaluate_all_baselines(cls, X_train, y_train, X_test, y_test):
        """
        Evaluate all baseline models.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_test: Test sequences
            y_test: Test labels

        Returns:
            dict: Results for all baselines
        """
        results = {}

        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION")
        print("="*60 + "\n")

        # Random classifier
        print("1. Random Classifier")
        acc_random, _ = cls.random_classifier(y_test)
        results['random'] = acc_random
        print(f"   Accuracy: {acc_random:.4f}\n")

        # Majority class
        print("2. Majority Class Classifier")
        acc_majority, _ = cls.majority_class_classifier(y_train, y_test)
        results['majority_class'] = acc_majority
        print(f"   Accuracy: {acc_majority:.4f}\n")

        # Random Forest
        print("3. Random Forest")
        acc_rf, _, _ = cls.random_forest_baseline(X_train, y_train, X_test, y_test)
        results['random_forest'] = acc_rf
        print(f"   Accuracy: {acc_rf:.4f}\n")

        # Logistic Regression
        print("4. Logistic Regression")
        acc_lr, _, _ = cls.logistic_regression_baseline(X_train, y_train, X_test, y_test)
        results['logistic_regression'] = acc_lr
        print(f"   Accuracy: {acc_lr:.4f}\n")

        print("="*60)
        print("BASELINE SUMMARY")
        print("="*60)
        for model_name, acc in results.items():
            print(f"{model_name:.<30} {acc:.4f}")
        print("="*60 + "\n")

        return results
