"""Comprehensive evaluation metrics"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation with enhanced metrics"""

    def __init__(self, class_names=None):
        """
        Initialize evaluator.

        Args:
            class_names (list): List of class names (default: ['bearish', 'bullish', 'neutral'])
        """
        self.class_names = class_names or ['bearish', 'bullish', 'neutral']

    def evaluate_all_metrics(self, y_true, y_pred):
        """
        Calculate all evaluation metrics.

        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels

        Returns:
            dict: Dictionary containing all metrics
        """
        results = {}

        # Overall accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names)),
            zero_division=0
        )

        results['per_class'] = {
            self.class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(self.class_names))
        }

        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        results['weighted_avg'] = {
            'precision': float(precision_w),
            'recall': float(recall_w),
            'f1_score': float(f1_w)
        }

        # Macro averages
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        results['macro_avg'] = {
            'precision': float(precision_m),
            'recall': float(recall_m),
            'f1_score': float(f1_m)
        }

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(
            y_true, y_pred, labels=range(len(self.class_names))
        )

        return results

    def print_results(self, results):
        """
        Print formatted evaluation results.

        Args:
            results (dict): Results from evaluate_all_metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)

        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

        print("\n" + "-"*70)
        print("PER-CLASS PERFORMANCE")
        print("-"*70)
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-"*70)

        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            print(f"{class_name:<12} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} "
                  f"{metrics['support']:<12}")

        print("\n" + "-"*70)
        print("WEIGHTED AVERAGES")
        print("-"*70)
        print(f"Precision: {results['weighted_avg']['precision']:.4f}")
        print(f"Recall:    {results['weighted_avg']['recall']:.4f}")
        print(f"F1-Score:  {results['weighted_avg']['f1_score']:.4f}")

        print("\n" + "-"*70)
        print("MACRO AVERAGES")
        print("-"*70)
        print(f"Precision: {results['macro_avg']['precision']:.4f}")
        print(f"Recall:    {results['macro_avg']['recall']:.4f}")
        print(f"F1-Score:  {results['macro_avg']['f1_score']:.4f}")

        print("\n" + "="*70 + "\n")

    def plot_confusion_matrix(self, cm, save_path=None, title='Confusion Matrix'):
        """
        Plot confusion matrix heatmap.

        Args:
            cm (np.array): Confusion matrix
            save_path (str): Path to save figure (optional)
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )

        plt.title(title, fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_per_class_metrics(self, results, save_path=None):
        """
        Plot per-class precision, recall, and F1-score.

        Args:
            results (dict): Results from evaluate_all_metrics
            save_path (str): Path to save figure (optional)
        """
        metrics_names = ['precision', 'recall', 'f1_score']
        metrics_data = {
            metric: [results['per_class'][cls][metric] for cls in self.class_names]
            for metric in metrics_names
        }

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, metrics_data['precision'], width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, metrics_data['recall'], width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_data['f1_score'], width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to: {save_path}")

        plt.show()

    def compare_with_baselines(self, lstm_accuracy, baseline_results):
        """
        Compare LSTM performance with baseline models.

        Args:
            lstm_accuracy (float): LSTM model accuracy
            baseline_results (dict): Baseline model results
        """
        print("\n" + "="*70)
        print("LSTM vs BASELINE COMPARISON")
        print("="*70)

        all_results = {'LSTM': lstm_accuracy, **baseline_results}

        # Sort by accuracy
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Model':<30} {'Accuracy':<15} {'Improvement vs Random':<20}")
        print("-"*70)

        random_acc = baseline_results.get('random', 0.33)

        for model_name, acc in sorted_results:
            improvement = ((acc - random_acc) / random_acc) * 100
            print(f"{model_name:<30} {acc:.4f} ({acc*100:.2f}%)    {improvement:>+6.2f}%")

        print("-"*70)

        # Calculate improvement over best baseline
        baseline_accs = [acc for name, acc in baseline_results.items()]
        best_baseline_acc = max(baseline_accs)
        best_baseline_name = [name for name, acc in baseline_results.items()
                              if acc == best_baseline_acc][0]

        lstm_improvement = ((lstm_accuracy - best_baseline_acc) / best_baseline_acc) * 100

        print(f"\nLSTM improvement over best baseline ({best_baseline_name}):")
        print(f"  {lstm_improvement:+.2f}%")

        if lstm_improvement > 0:
            print(f"  ✓ LSTM outperforms all baselines")
        else:
            print(f"  ⚠️ LSTM does not outperform best baseline")

        print("="*70 + "\n")
