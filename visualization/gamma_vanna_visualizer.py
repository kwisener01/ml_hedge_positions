"""
Gamma & Vanna Visualization
Shows gamma exposure profiles, model predictions vs gamma levels, and feature importance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
from datetime import datetime

from institutional.gamma_exposure import GammaVannaCalculator
from models.svm_ensemble import SVMEnsemble
from data.tradier_client import TradierClient
from training.train_ensemble import TrainingDataBuilder


class GammaVannaVisualizer:
    """Visualize gamma/vanna exposure and model behavior"""

    def __init__(self):
        self.gamma_calc = GammaVannaCalculator()
        self.client = TradierClient()

    def plot_gamma_vanna_profile(self, symbol: str, save_path: str = None):
        """
        Plot gamma and vanna exposure across all strikes
        Shows where dealers have hedging pressure
        """
        print(f"\nGenerating gamma/vanna profile for {symbol}...")

        # Get current price
        quote = self.client.get_quote(symbol)
        spot_price = quote.last or quote.mid_price

        # Get expirations
        expirations = self.client.get_expirations(symbol)
        if not expirations:
            print("[ERROR] No expirations available")
            return

        front_expiry = expirations[0]

        # Get options chain
        chain_df = self.client.get_options_chain(symbol, front_expiry)
        if chain_df.empty:
            print("[ERROR] No options chain data")
            return

        # Convert expiration to datetime
        try:
            expiry_dt = datetime.strptime(front_expiry, '%Y-%m-%d')
        except:
            from datetime import timedelta
            expiry_dt = datetime.now() + timedelta(days=30)

        # Calculate gamma/vanna for all strikes
        strike_data, total_gamma, total_vanna = self.gamma_calc.calculate_gex_vanna(
            symbol, spot_price, chain_df, expiry_dt
        )

        # Prepare data for plotting
        strikes = sorted(strike_data.keys())
        call_gamma = [strike_data[s]['call_gamma'] for s in strikes]
        put_gamma = [strike_data[s]['put_gamma'] for s in strikes]
        total_gamma_by_strike = [c + p for c, p in zip(call_gamma, put_gamma)]

        call_vanna = [strike_data[s]['call_vanna'] for s in strikes]
        put_vanna = [strike_data[s]['put_vanna'] for s in strikes]
        total_vanna_by_strike = [c + p for c, p in zip(call_vanna, put_vanna)]

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Overall title
        fig.suptitle(f'{symbol} Gamma & Vanna Exposure Profile\nSpot: ${spot_price:.2f} | Expiration: {front_expiry}',
                     fontsize=16, fontweight='bold')

        # 1. Total Gamma Exposure by Strike
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(strikes, total_gamma_by_strike, color=['red' if g < 0 else 'green' for g in total_gamma_by_strike],
                alpha=0.7, width=1)
        ax1.axvline(spot_price, color='black', linestyle='--', linewidth=2, label=f'Spot: ${spot_price:.2f}')
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Strike Price', fontsize=12)
        ax1.set_ylabel('Net Gamma Exposure ($B)', fontsize=12)
        ax1.set_title('Net Dealer Gamma Exposure\n(Negative = Dealers will AMPLIFY moves | Positive = Dealers will DAMPEN moves)',
                      fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Find zero gamma
        cumulative = np.cumsum(total_gamma_by_strike)
        zero_idx = None
        zero_strike = None
        for i in range(len(cumulative)-1):
            if cumulative[i] <= 0 <= cumulative[i+1]:
                zero_idx = i
                zero_strike = strikes[zero_idx]
                break

        if zero_idx is not None and zero_strike is not None:
            ax1.axvline(zero_strike, color='orange', linestyle=':', linewidth=2,
                       label=f'Zero Gamma: ${zero_strike:.2f}')
            ax1.legend()

        # 2. Call vs Put Gamma
        ax2 = fig.add_subplot(gs[1, 0])
        width = 0.4
        x = np.arange(len(strikes))
        ax2.bar(x - width/2, call_gamma, width, label='Call Gamma', color='green', alpha=0.7)
        ax2.bar(x + width/2, put_gamma, width, label='Put Gamma', color='red', alpha=0.7)
        ax2.axvline(np.where(np.array(strikes) >= spot_price)[0][0], color='black',
                   linestyle='--', linewidth=1.5, label='Spot')
        ax2.set_xlabel('Strike Index', fontsize=10)
        ax2.set_ylabel('Gamma Exposure ($B)', fontsize=10)
        ax2.set_title('Call vs Put Gamma Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Total Vanna Exposure by Strike
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.bar(strikes, total_vanna_by_strike, color=['red' if v < 0 else 'blue' for v in total_vanna_by_strike],
                alpha=0.7, width=1)
        ax3.axvline(spot_price, color='black', linestyle='--', linewidth=2, label=f'Spot: ${spot_price:.2f}')
        ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Strike Price', fontsize=10)
        ax3.set_ylabel('Net Vanna Exposure ($M)', fontsize=10)
        ax3.set_title('Net Dealer Vanna Exposure\n(Sensitivity to Volatility Changes)',
                      fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Gamma Concentration Around Spot
        ax4 = fig.add_subplot(gs[2, 0])
        # Focus on ±10% around spot
        spot_idx = np.where(np.array(strikes) >= spot_price)[0][0]
        range_low = max(0, spot_idx - 20)
        range_high = min(len(strikes), spot_idx + 20)

        focus_strikes = strikes[range_low:range_high]
        focus_gamma = total_gamma_by_strike[range_low:range_high]

        ax4.plot(focus_strikes, focus_gamma, marker='o', linewidth=2, markersize=4, color='purple')
        ax4.axvline(spot_price, color='black', linestyle='--', linewidth=2, label=f'Spot: ${spot_price:.2f}')
        ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax4.fill_between(focus_strikes, 0, focus_gamma, alpha=0.3,
                        color=['red' if g < 0 else 'green' for g in focus_gamma])
        ax4.set_xlabel('Strike Price', fontsize=10)
        ax4.set_ylabel('Gamma Exposure ($B)', fontsize=10)
        ax4.set_title('Gamma Concentration Near Spot (±10%)', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Summary Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        total_call_gamma = sum(call_gamma)
        total_put_gamma = sum(put_gamma)
        total_call_vanna = sum(call_vanna)
        total_put_vanna = sum(put_vanna)

        vanna_bias = total_call_vanna / (abs(total_call_vanna) + abs(total_put_vanna)) if (abs(total_call_vanna) + abs(total_put_vanna)) > 0 else 0

        gamma_at_spot = total_gamma_by_strike[spot_idx] if spot_idx < len(total_gamma_by_strike) else 0

        summary_text = f"""
GAMMA & VANNA SUMMARY
{'='*40}

Total Gamma Exposure: ${total_gamma:.2f}B
  Call Gamma: ${total_call_gamma:.2f}B
  Put Gamma: ${total_put_gamma:.2f}B

Gamma at Spot: ${gamma_at_spot:.2f}B
  {'[HIGH PRESSURE]' if abs(gamma_at_spot) > 50 else '[MODERATE]' if abs(gamma_at_spot) > 20 else '[LOW]'}

Total Vanna: ${total_vanna:.2f}M
  Call Vanna: ${total_call_vanna:.2f}M
  Put Vanna: ${total_put_vanna:.2f}M
  Bias: {vanna_bias:.1%}

Dealer Behavior:
  {' AMPLIFY moves' if total_gamma < 0 else ' DAMPEN moves'}
  {' (Net Short Gamma)' if total_gamma < 0 else ' (Net Long Gamma)'}

Positioning:
  {' PUT HEAVY' if vanna_bias < -0.1 else ' CALL HEAVY' if vanna_bias > 0.1 else ' BALANCED'}
  {' - Defensive' if vanna_bias < -0.1 else ' - Bullish' if vanna_bias > 0.1 else ''}

Zero Gamma Level:
  ${zero_strike:.2f if zero_strike is not None else 'Not Found'}
  {f'({(zero_strike - spot_price) / spot_price * 100:+.1f}% from spot)' if zero_strike is not None else ''}
"""

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
        else:
            plt.show()

        return fig

    def plot_model_vs_gamma(self, symbol: str, save_path: str = None):
        """
        Plot model predictions vs gamma exposure levels
        Shows how model behavior changes with gamma
        """
        print(f"\nGenerating model vs gamma analysis for {symbol}...")

        # Load model
        model_path = Path(__file__).parent.parent / f"models/trained/{symbol}_ensemble.pkl"
        ensemble = SVMEnsemble.load(str(model_path))

        # Build dataset
        builder = TrainingDataBuilder(symbol)
        daily_data = builder.load_historical_data()
        quotes = builder.create_synthetic_quotes(daily_data)
        X, y_returns, feature_names = builder.build_feature_dataset(quotes)

        # Get predictions
        y_pred = ensemble.predict_batch(X)
        y_true = ensemble._compute_target(y_returns)

        # Find gamma feature indices
        gamma_features = {}
        for i, name in enumerate(feature_names):
            if 'gamma' in name.lower() or 'vanna' in name.lower():
                gamma_features[name] = i

        print(f"Found gamma/vanna features: {list(gamma_features.keys())}")

        # Create visualization
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(f'{symbol} Model Predictions vs Gamma/Vanna Levels\nTest Accuracy: {ensemble.training_metrics.test_accuracy*100:.2f}%',
                     fontsize=16, fontweight='bold')

        # 1. Predictions vs Total Gamma
        if 'inst_total_gamma' in gamma_features:
            ax1 = fig.add_subplot(gs[0, :])
            gamma_idx = gamma_features['inst_total_gamma']
            gamma_values = X[:, gamma_idx]

            # Create bins
            bins = np.percentile(gamma_values, [0, 25, 50, 75, 100])
            bin_labels = ['Very Neg', 'Neg', 'Near Zero', 'Pos']

            accuracies = []
            up_pcts = []
            down_pcts = []
            counts = []

            for i in range(len(bins)-1):
                mask = (gamma_values >= bins[i]) & (gamma_values < bins[i+1])
                if np.sum(mask) > 0:
                    acc = np.mean(y_pred[mask] == y_true[mask])
                    up_pct = np.sum(y_pred[mask] == 1) / np.sum(mask) * 100
                    down_pct = np.sum(y_pred[mask] == -1) / np.sum(mask) * 100

                    accuracies.append(acc * 100)
                    up_pcts.append(up_pct)
                    down_pcts.append(down_pct)
                    counts.append(np.sum(mask))
                else:
                    accuracies.append(0)
                    up_pcts.append(0)
                    down_pcts.append(0)
                    counts.append(0)

            x = np.arange(len(bin_labels))
            width = 0.25

            ax1.bar(x - width, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
            ax1.bar(x, up_pcts, width, label='UP Predictions', color='green', alpha=0.7)
            ax1.bar(x + width, down_pcts, width, label='DOWN Predictions', color='red', alpha=0.7)

            ax1.set_xlabel('Total Gamma Regime', fontsize=12)
            ax1.set_ylabel('Percentage', fontsize=12)
            ax1.set_title('Model Behavior vs Gamma Regime\n(How predictions change with dealer positioning)',
                         fontsize=11, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(bin_labels)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Add sample counts
            for i, count in enumerate(counts):
                ax1.text(i, max(accuracies + up_pcts + down_pcts) + 2, f'n={count}',
                        ha='center', va='bottom', fontsize=9)

        # 2. Predictions vs Gamma at Spot
        if 'inst_gamma_at_spot' in gamma_features:
            ax2 = fig.add_subplot(gs[1, 0])
            gamma_spot_idx = gamma_features['inst_gamma_at_spot']
            gamma_spot_values = X[:, gamma_spot_idx]

            # Scatter plot
            colors = ['green' if p == 1 else 'red' if p == -1 else 'gray' for p in y_pred]
            correct = y_pred == y_true

            ax2.scatter(gamma_spot_values[correct], np.arange(np.sum(correct)),
                       c=[colors[i] for i, c in enumerate(correct) if c],
                       alpha=0.3, s=20, label='Correct')
            ax2.scatter(gamma_spot_values[~correct], np.arange(np.sum(~correct)),
                       c='black', alpha=0.3, s=20, marker='x', label='Wrong')

            ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
            ax2.set_xlabel('Gamma at Spot ($B)', fontsize=10)
            ax2.set_ylabel('Sample Index', fontsize=10)
            ax2.set_title('Predictions vs Gamma Concentration\n(Green=UP, Red=DOWN predictions)',
                         fontsize=10, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Predictions vs Vanna Bias
        if 'inst_vanna_bias' in gamma_features:
            ax3 = fig.add_subplot(gs[1, 1])
            vanna_idx = gamma_features['inst_vanna_bias']
            vanna_values = X[:, vanna_idx]

            # Create bins: Put Heavy, Balanced, Call Heavy
            put_heavy = vanna_values < -0.1
            balanced = (vanna_values >= -0.1) & (vanna_values <= 0.1)
            call_heavy = vanna_values > 0.1

            regimes = ['Put Heavy', 'Balanced', 'Call Heavy']
            masks = [put_heavy, balanced, call_heavy]

            accs = []
            ups = []
            downs = []
            cnts = []

            for mask in masks:
                if np.sum(mask) > 0:
                    accs.append(np.mean(y_pred[mask] == y_true[mask]) * 100)
                    ups.append(np.sum(y_pred[mask] == 1) / np.sum(mask) * 100)
                    downs.append(np.sum(y_pred[mask] == -1) / np.sum(mask) * 100)
                    cnts.append(np.sum(mask))
                else:
                    accs.append(0)
                    ups.append(0)
                    downs.append(0)
                    cnts.append(0)

            x = np.arange(len(regimes))
            width = 0.25

            ax3.bar(x - width, accs, width, label='Accuracy', color='blue', alpha=0.7)
            ax3.bar(x, ups, width, label='UP Predictions', color='green', alpha=0.7)
            ax3.bar(x + width, downs, width, label='DOWN Predictions', color='red', alpha=0.7)

            ax3.set_xlabel('Vanna Bias Regime', fontsize=10)
            ax3.set_ylabel('Percentage', fontsize=10)
            ax3.set_title('Model Behavior vs Vanna Bias\n(Options positioning influence)',
                         fontsize=10, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(regimes)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            for i, count in enumerate(cnts):
                ax3.text(i, max(accs + ups + downs) + 2, f'n={count}',
                        ha='center', va='bottom', fontsize=9)

        # 4. Feature Importance for Gamma/Vanna features
        ax4 = fig.add_subplot(gs[2, :])

        # Get feature importance (using coefficient of variation from individual SVMs)
        feature_stds = []
        feature_means = []

        for feat_idx in range(X.shape[1]):
            if feat_idx in gamma_features.values():
                # Calculate importance as variation in this feature
                feat_std = np.std(X[:, feat_idx])
                feat_mean = np.abs(np.mean(X[:, feat_idx]))
                feature_stds.append(feat_std)
                feature_means.append(feat_mean)

        gamma_feature_names = [name for name in gamma_features.keys()]

        # Normalize
        if feature_stds:
            importance = np.array(feature_stds) / (np.sum(feature_stds) if np.sum(feature_stds) > 0 else 1)

            ax4.barh(gamma_feature_names, importance, color='purple', alpha=0.7)
            ax4.set_xlabel('Relative Importance (Feature Variation)', fontsize=10)
            ax4.set_title('Gamma/Vanna Feature Importance\n(How much each feature varies in the data)',
                         fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
        else:
            plt.show()

        return fig


def main():
    """Generate all gamma/vanna visualizations"""
    visualizer = GammaVannaVisualizer()

    output_dir = Path(__file__).parent.parent / "visualization/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ["SPY", "QQQ"]:
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS: {symbol}")
        print(f"{'='*80}")

        # 1. Gamma/Vanna Profile
        profile_path = output_dir / f"{symbol}_gamma_vanna_profile.png"
        try:
            visualizer.plot_gamma_vanna_profile(symbol, str(profile_path))
        except Exception as e:
            print(f"[ERROR] Failed to generate profile: {e}")
            import traceback
            traceback.print_exc()

        # 2. Model vs Gamma
        model_path = output_dir / f"{symbol}_model_vs_gamma.png"
        try:
            visualizer.plot_model_vs_gamma(symbol, str(model_path))
        except Exception as e:
            print(f"[ERROR] Failed to generate model analysis: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"[OK] All visualizations saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
