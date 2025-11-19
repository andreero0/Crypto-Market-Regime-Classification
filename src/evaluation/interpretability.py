"""Model interpretability using SHAP values"""

import shap
import matplotlib.pyplot as plt
import numpy as np


class ModelInterpreter:
    """SHAP-based model interpretation for LSTM"""

    def __init__(self, model, feature_names):
        """
        Initialize interpreter.

        Args:
            model: Trained Keras model
            feature_names (list): List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def create_explainer(self, X_background, max_background=100):
        """
        Create SHAP explainer with background samples.

        Args:
            X_background (np.array): Background data for SHAP
            max_background (int): Maximum background samples (for speed)

        Returns:
            shap.DeepExplainer: SHAP explainer object
        """
        print(f"\nCreating SHAP explainer with {max_background} background samples...")

        # Use subset of data as background
        background = X_background[:max_background]

        # Create deep explainer for neural networks
        self.explainer = shap.DeepExplainer(self.model, background)

        print("✓ SHAP explainer created")
        return self.explainer

    def calculate_shap_values(self, X_test, max_samples=100):
        """
        Calculate SHAP values for test samples.

        Args:
            X_test (np.array): Test data
            max_samples (int): Maximum samples to explain (computational limit)

        Returns:
            list: SHAP values for each class
        """
        if self.explainer is None:
            raise ValueError("Must create explainer first using create_explainer()")

        print(f"\nCalculating SHAP values for {max_samples} samples...")
        print("This may take a few minutes...")

        # Calculate SHAP values
        test_subset = X_test[:max_samples]
        self.shap_values = self.explainer.shap_values(test_subset)

        print("✓ SHAP values calculated")
        return self.shap_values

    def plot_summary(self, X_test, max_samples=100, class_names=None,
                    save_dir='results/figures'):
        """
        Plot SHAP summary for each class.

        Args:
            X_test (np.array): Test data
            max_samples (int): Number of samples
            class_names (list): Class names
            save_dir (str): Directory to save plots
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")

        if class_names is None:
            class_names = ['bearish', 'bullish', 'neutral']

        print(f"\nGenerating SHAP summary plots...")

        # Flatten sequences for visualization
        X_test_flat = X_test[:max_samples].reshape(X_test[:max_samples].shape[0], -1)

        # Create feature names for flattened temporal data
        timesteps = X_test.shape[1]
        temporal_features = []
        for t in range(timesteps):
            for f in self.feature_names:
                temporal_features.append(f"{f}_t-{timesteps-t-1}")

        # Plot for each class
        for class_idx, class_name in enumerate(class_names):
            print(f"  Plotting {class_name} class...")

            # Flatten SHAP values for this class
            shap_values_flat = self.shap_values[class_idx].reshape(
                self.shap_values[class_idx].shape[0], -1
            )

            # Create figure
            plt.figure(figsize=(14, 10))
            shap.summary_plot(
                shap_values_flat,
                X_test_flat,
                feature_names=temporal_features,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Feature Importance - {class_name.capitalize()} Class',
                     fontsize=16, pad=20)
            plt.tight_layout()

            # Save figure
            save_path = f'{save_dir}/shap_summary_{class_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {save_path}")

            plt.close()

        print("✓ SHAP summary plots generated\n")

    def plot_feature_importance(self, class_idx=1, class_name='bullish',
                                save_path=None):
        """
        Plot aggregated feature importance.

        Args:
            class_idx (int): Class index to plot
            class_name (str): Class name
            save_path (str): Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")

        print(f"\nPlotting feature importance for {class_name} class...")

        # Get SHAP values for this class
        shap_class = self.shap_values[class_idx]

        # Average importance across timesteps
        # Shape: (samples, timesteps, features) -> (features,)
        feature_importance = np.abs(shap_class).mean(axis=(0, 1))

        # Sort by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_n = min(20, len(self.feature_names))
        top_idx = sorted_idx[:top_n]

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), feature_importance[top_idx])
        plt.yticks(range(top_n), [self.feature_names[i] for i in top_idx])
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title(f'Top {top_n} Features - {class_name.capitalize()} Class',
                 fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        plt.show()
        print("✓ Feature importance plot generated\n")

    def analyze_sample(self, X_sample, sample_idx=0, class_names=None,
                      save_path=None):
        """
        Analyze a single prediction with SHAP force plot.

        Args:
            X_sample (np.array): Sample to analyze (1, timesteps, features)
            sample_idx (int): Sample index for labeling
            class_names (list): Class names
            save_path (str): Path to save plot
        """
        if self.explainer is None:
            raise ValueError("Must create explainer first")

        if class_names is None:
            class_names = ['bearish', 'bullish', 'neutral']

        print(f"\nAnalyzing sample {sample_idx}...")

        # Calculate SHAP for this sample
        shap_sample = self.explainer.shap_values(X_sample)

        # Make prediction
        pred_proba = self.model.predict(X_sample, verbose=0)[0]
        pred_class = np.argmax(pred_proba)

        print(f"  Predicted: {class_names[pred_class]} ({pred_proba[pred_class]:.3f})")
        print(f"  Probabilities: {dict(zip(class_names, pred_proba))}")

        # Force plot for predicted class
        shap.force_plot(
            self.explainer.expected_value[pred_class],
            shap_sample[pred_class][0].flatten(),
            X_sample[0].flatten(),
            matplotlib=True,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        plt.show()
        print("✓ Sample analysis complete\n")


def run_full_interpretability_analysis(model, X_train, X_test, feature_names,
                                       class_names=None, max_background=100,
                                       max_samples=100, save_dir='results/figures'):
    """
    Run complete SHAP interpretability analysis.

    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data
        feature_names: List of feature names
        class_names: List of class names
        max_background: Max background samples
        max_samples: Max test samples to analyze
        save_dir: Directory to save plots

    Returns:
        ModelInterpreter: Interpreter with calculated SHAP values
    """
    print("\n" + "="*70)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("="*70)

    # Create interpreter
    interpreter = ModelInterpreter(model, feature_names)

    # Create explainer
    interpreter.create_explainer(X_train, max_background=max_background)

    # Calculate SHAP values
    interpreter.calculate_shap_values(X_test, max_samples=max_samples)

    # Generate plots
    interpreter.plot_summary(X_test, max_samples=max_samples,
                            class_names=class_names, save_dir=save_dir)

    # Plot feature importance for each class
    if class_names is None:
        class_names = ['bearish', 'bullish', 'neutral']

    for idx, class_name in enumerate(class_names):
        save_path = f'{save_dir}/shap_importance_{class_name}.png'
        interpreter.plot_feature_importance(
            class_idx=idx,
            class_name=class_name,
            save_path=save_path
        )

    print("="*70)
    print("✓ SHAP analysis complete")
    print("="*70 + "\n")

    return interpreter
