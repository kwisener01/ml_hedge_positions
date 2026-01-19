"""
Model Visualization Toolkit
Visual representation of SVM ensemble performance and decision-making
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.svm_ensemble import SVMEnsemble
from training.train_ensemble import TrainingDataBuilder


class ModelVisualizer:
    """
    Visualize SVM ensemble model structure and performance

    Creates:
    1. Feature importance plot
    2. Confusion matrix
    3. Decision boundary visualization (2D projection)
    4. Voting distribution
    5. Performance metrics
    6. Model architecture diagram
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        model_path = Path(__file__).parent.parent / f"models/trained/{symbol}_ensemble.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.ensemble = SVMEnsemble.load(str(model_path))
        self.output_dir = Path(__file__).parent.parent / "visualization/output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def load_test_data(self):
        """Load test data for visualization"""
        print(f"Loading test data for {self.symbol}...")

        builder = TrainingDataBuilder(self.symbol)
        historical_data = builder.load_historical_data()
        quotes = builder.create_synthetic_quotes(historical_data, quotes_per_day=10)
        X, y_returns, feature_names = builder.build_feature_dataset(quotes, lookforward_bars=5)

        # Use same test split as training
        from sklearn.model_selection import train_test_split

        y = self.ensemble._compute_target(y_returns)

        # Check if we can stratify
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = all(counts >= 2)

        if can_stratify:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        print(f"Test set: {len(X_test)} samples")

        return X_test, y_test, feature_names

    def plot_feature_importance(self, X_test, y_test, feature_names):
        """
        Plot feature importance using support vector weights

        For SVMs, we approximate importance by looking at feature variance
        in support vectors across the ensemble
        """
        print("\n1. Creating feature importance plot...")

        # Collect support vectors from all SVMs
        all_sv_indices = []
        for svm, scaler in zip(self.ensemble.models, self.ensemble.scalers):
            if hasattr(svm, 'support_'):
                all_sv_indices.extend(svm.support_)

        # Calculate feature variance in support vectors
        # Higher variance = more important for decision boundary
        feature_importance = np.std(X_test, axis=0)

        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]

        # Plot top 15 features
        top_k = 15
        top_indices = sorted_indices[:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = feature_importance[top_indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), top_scores)
        plt.yticks(range(top_k), top_features)
        plt.xlabel('Feature Importance (Std Dev)')
        plt.title(f'{self.symbol} - Top {top_k} Most Important Features')
        plt.tight_layout()

        output_file = self.output_dir / f"{self.symbol}_feature_importance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_file}")

    def plot_confusion_matrix(self, X_test, y_test):
        """Plot confusion matrix"""
        print("\n2. Creating confusion matrix...")

        # Get predictions
        y_pred = self.ensemble.predict_batch(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['DOWN', 'NEUTRAL', 'UP'],
            yticklabels=['DOWN', 'NEUTRAL', 'UP']
        )
        plt.title(f'{self.symbol} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        output_file = self.output_dir / f"{self.symbol}_confusion_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_file}")

        # Print classification report
        print("\n   Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['DOWN', 'NEUTRAL', 'UP'],
            zero_division=0
        ))

    def plot_decision_boundary_2d(self, X_test, y_test, feature_names):
        """
        Visualize decision boundary using PCA to reduce to 2D

        Note: This is a projection, actual decision happens in 32D space
        """
        print("\n3. Creating decision boundary visualization (2D projection)...")

        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_test)

        # Create mesh for decision boundary
        h = 0.02  # Step size
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Predict on mesh (transform back to original space)
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_original = pca.inverse_transform(mesh_points)
        Z = self.ensemble.predict_batch(mesh_original)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(12, 8))

        # Decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-1.5, -0.5, 0.5, 1.5])

        # Scatter plot of test points
        colors = ['red', 'yellow', 'green']
        labels = ['DOWN', 'NEUTRAL', 'UP']

        for label_val, color, label in zip([-1, 0, 1], colors, labels):
            mask = y_test == label_val
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color,
                label=label,
                edgecolors='black',
                s=50,
                alpha=0.7
            )

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title(f'{self.symbol} - Decision Boundary (PCA 2D Projection)')
        plt.legend()
        plt.colorbar(label='Prediction')

        output_file = self.output_dir / f"{self.symbol}_decision_boundary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_file}")
        print(f"   Note: This is a 2D projection of 32D decision space")
        print(f"   PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")

    def plot_voting_distribution(self, X_test):
        """Plot distribution of ensemble votes"""
        print("\n4. Creating voting distribution plot...")

        # Get all votes for test set
        all_votes = self.ensemble.predict_batch(X_test, return_votes=True)

        # Calculate vote distributions
        vote_counts = np.zeros((len(X_test), 3))  # DOWN, NEUTRAL, UP
        for i in range(len(X_test)):
            votes = all_votes[i, :]
            vote_counts[i, 0] = np.sum(votes == -1)  # DOWN votes
            vote_counts[i, 1] = np.sum(votes == 0)   # NEUTRAL votes
            vote_counts[i, 2] = np.sum(votes == 1)   # UP votes

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Histogram of confidence (max vote percentage)
        confidence = vote_counts.max(axis=1) / self.ensemble.ensemble_size
        axes[0, 0].hist(confidence, bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Prediction Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Prediction Confidence')
        axes[0, 0].axvline(confidence.mean(), color='red', linestyle='--',
                          label=f'Mean: {confidence.mean():.2f}')
        axes[0, 0].legend()

        # 2. Average votes per class
        avg_votes = vote_counts.mean(axis=0)
        axes[0, 1].bar(['DOWN', 'NEUTRAL', 'UP'], avg_votes, color=['red', 'yellow', 'green'])
        axes[0, 1].set_ylabel('Average Votes (out of 100)')
        axes[0, 1].set_title('Average Ensemble Votes per Class')

        # 3. Vote agreement distribution
        max_votes = vote_counts.max(axis=1)
        axes[1, 0].hist(max_votes, bins=range(50, 101, 5), edgecolor='black')
        axes[1, 0].set_xlabel('Number of Agreeing SVMs')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Ensemble Agreement Distribution')

        # 4. Confidence vs correctness (if we have true labels)
        y_pred = self.ensemble.predict_batch(X_test)
        axes[1, 1].scatter(
            confidence,
            y_pred,
            alpha=0.5,
            c=['red' if p == -1 else 'yellow' if p == 0 else 'green' for p in y_pred]
        )
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Prediction (-1=DOWN, 0=NEUTRAL, 1=UP)')
        axes[1, 1].set_title('Confidence vs Prediction')

        plt.tight_layout()

        output_file = self.output_dir / f"{self.symbol}_voting_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_file}")
        print(f"   Average confidence: {confidence.mean():.2%}")

    def plot_model_architecture(self):
        """Create diagram showing model architecture"""
        print("\n5. Creating model architecture diagram...")

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, f'{self.symbol} SVM Ensemble Architecture',
                ha='center', va='top', fontsize=20, fontweight='bold')

        # Input layer
        ax.text(0.1, 0.85, 'INPUT', ha='center', fontsize=12, fontweight='bold')
        ax.add_patch(plt.Rectangle((0.05, 0.75), 0.1, 0.08, fill=True, color='lightblue', ec='black'))
        ax.text(0.1, 0.79, f'32 Features', ha='center', fontsize=10)

        # Feature extraction
        ax.text(0.3, 0.85, 'FEATURES', ha='center', fontsize=12, fontweight='bold')
        features = ['Window (8)', 'Classic (14)', 'Institutional (10)']
        for i, feat in enumerate(features):
            y_pos = 0.78 - i*0.06
            ax.add_patch(plt.Rectangle((0.25, y_pos-0.03), 0.1, 0.05, fill=True, color='lightgreen', ec='black'))
            ax.text(0.3, y_pos, feat, ha='center', fontsize=9)

        # SVM Ensemble
        ax.text(0.5, 0.85, 'ENSEMBLE', ha='center', fontsize=12, fontweight='bold')
        ax.add_patch(plt.Rectangle((0.42, 0.55), 0.16, 0.28, fill=True, color='lightyellow', ec='black', linewidth=2))
        ax.text(0.5, 0.80, '100 SVMs', ha='center', fontsize=10, fontweight='bold')
        ax.text(0.5, 0.75, 'Polynomial Kernel (d=2)', ha='center', fontsize=8)
        ax.text(0.5, 0.71, f'C = {self.ensemble.constraint_param}', ha='center', fontsize=8)
        ax.text(0.5, 0.67, f'Each on 80% random subset', ha='center', fontsize=8)
        ax.text(0.5, 0.63, 'Independent training', ha='center', fontsize=8)

        # Voting
        ax.text(0.7, 0.85, 'VOTING', ha='center', fontsize=12, fontweight='bold')
        ax.add_patch(plt.Rectangle((0.65, 0.75), 0.1, 0.08, fill=True, color='lightcoral', ec='black'))
        ax.text(0.7, 0.79, 'Majority Vote', ha='center', fontsize=10)

        # Output
        ax.text(0.9, 0.85, 'OUTPUT', ha='center', fontsize=12, fontweight='bold')
        outputs = ['DOWN (-1)', 'NEUTRAL (0)', 'UP (+1)']
        colors = ['red', 'yellow', 'green']
        for i, (out, col) in enumerate(zip(outputs, colors)):
            y_pos = 0.78 - i*0.06
            ax.add_patch(plt.Rectangle((0.85, y_pos-0.03), 0.1, 0.05, fill=True, color=col, ec='black'))
            ax.text(0.9, y_pos, out, ha='center', fontsize=9)

        # Arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        ax.annotate('', xy=(0.25, 0.79), xytext=(0.15, 0.79), arrowprops=arrow_props)
        ax.annotate('', xy=(0.42, 0.79), xytext=(0.35, 0.79), arrowprops=arrow_props)
        ax.annotate('', xy=(0.65, 0.79), xytext=(0.58, 0.79), arrowprops=arrow_props)
        ax.annotate('', xy=(0.85, 0.79), xytext=(0.75, 0.79), arrowprops=arrow_props)

        # Model specs
        specs = f"""
Model Specifications:
• Training Samples: {self.ensemble.training_metrics.train_size + self.ensemble.training_metrics.test_size}
• Test Accuracy: {self.ensemble.training_metrics.test_accuracy*100:.2f}%
• Training Time: {self.ensemble.training_metrics.training_time:.1f}s
• Individual SVM Avg: {self.ensemble.training_metrics.avg_individual_accuracy*100:.2f}%
• Alpha Threshold: {self.ensemble.alpha_threshold}
"""
        ax.text(0.5, 0.40, specs, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # How it works
        how_it_works = """
How the Ensemble Works:
1. Each of 100 SVMs trains on random 80% subset of data
2. Each SVM independently predicts: DOWN/NEUTRAL/UP
3. Votes are aggregated via majority voting
4. Final prediction has confidence = % of agreeing SVMs
5. Polynomial kernel captures non-linear price relationships
"""
        ax.text(0.5, 0.20, how_it_works, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        output_file = self.output_dir / f"{self.symbol}_architecture.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_file}")

    def generate_all_visualizations(self):
        """Generate complete visualization suite"""
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS: {self.symbol}")
        print(f"{'='*80}")

        # Load test data
        X_test, y_test, feature_names = self.load_test_data()

        # Generate all plots
        self.plot_feature_importance(X_test, y_test, feature_names)
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_decision_boundary_2d(X_test, y_test, feature_names)
        self.plot_voting_distribution(X_test)
        self.plot_model_architecture()

        print(f"\n{'='*80}")
        print(f"VISUALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Files generated:")
        print(f"  1. {self.symbol}_feature_importance.png")
        print(f"  2. {self.symbol}_confusion_matrix.png")
        print(f"  3. {self.symbol}_decision_boundary.png")
        print(f"  4. {self.symbol}_voting_distribution.png")
        print(f"  5. {self.symbol}_architecture.png")
        print()


def main():
    """Generate visualizations for all symbols"""
    import argparse

    parser = argparse.ArgumentParser(description="Model Visualization")
    parser.add_argument('--symbol', choices=['SPY', 'QQQ', 'all'], default='all')

    args = parser.parse_args()

    symbols = ['SPY', 'QQQ'] if args.symbol == 'all' else [args.symbol]

    for symbol in symbols:
        try:
            visualizer = ModelVisualizer(symbol)
            visualizer.generate_all_visualizations()
        except Exception as e:
            print(f"\n[ERROR] Failed to visualize {symbol}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
