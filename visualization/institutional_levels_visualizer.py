"""
HP & MHP Institutional Levels Visualization
Shows where institutional levels are relative to price and how model responds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
from datetime import datetime

from institutional.hedge_pressure import HedgePressureCalculator
from institutional.monthly_hp import MonthlyHPCalculator
from institutional.half_gap import HalfGapCalculator
from models.svm_ensemble import SVMEnsemble
from data.tradier_client import TradierClient
from training.train_ensemble import TrainingDataBuilder


class InstitutionalLevelsVisualizer:
    """Visualize HP, MHP, and HG levels with model predictions"""

    def __init__(self):
        self.client = TradierClient()
        self.hp_calc = HedgePressureCalculator()
        self.mhp_calc = MonthlyHPCalculator()
        self.hg_calc = HalfGapCalculator()

    def plot_institutional_levels(self, symbol: str, save_path: str = None):
        """
        Plot HP, MHP, and HG levels relative to current price
        Shows model predictions near these levels
        """
        print(f"\nGenerating institutional levels visualization for {symbol}...")

        # Get current price
        quote = self.client.get_quote(symbol)
        spot_price = quote.last or quote.mid_price

        # Get data for institutional calculations
        chains = self.client.get_multiple_chains(symbol, expiration_count=4)

        # Get 90 days of history
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        history = self.client.get_history(
            symbol,
            interval="daily",
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        # Calculate institutional levels
        hp_result = None
        mhp_result = None
        hg_result = None

        if chains:
            nearest_exp = sorted(chains.keys())[0]
            hp_result = self.hp_calc.calculate(chains[nearest_exp], spot_price, symbol)
            mhp_result = self.mhp_calc.calculate(chains, spot_price, symbol)

        if not history.empty:
            hg_result = self.hg_calc.calculate(history, spot_price, symbol)

        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

        fig.suptitle(f'{symbol} Institutional Levels & Model Response\nSpot Price: ${spot_price:.2f}',
                     fontsize=16, fontweight='bold')

        # 1. All Levels on One Chart (Price Chart Style)
        ax1 = fig.add_subplot(gs[0, :])

        # Plot price history (last 30 days)
        recent_history = history.tail(30)
        ax1.plot(recent_history.index, recent_history['close'], color='black',
                linewidth=2, label='Price', marker='o', markersize=3)

        # Current spot price
        ax1.axhline(spot_price, color='blue', linestyle='-', linewidth=2.5, label=f'Spot: ${spot_price:.2f}')

        # HP Levels
        if hp_result:
            if hp_result.key_support:
                ax1.axhline(hp_result.key_support, color='green', linestyle='--',
                           linewidth=2, label=f'HP Support: ${hp_result.key_support:.2f}')
            if hp_result.key_resistance:
                ax1.axhline(hp_result.key_resistance, color='red', linestyle='--',
                           linewidth=2, label=f'HP Resistance: ${hp_result.key_resistance:.2f}')

        # MHP Levels
        if mhp_result:
            if mhp_result.primary_support:
                ax1.axhline(mhp_result.primary_support, color='darkgreen', linestyle=':',
                           linewidth=2, label=f'MHP Support: ${mhp_result.primary_support:.2f}')
            if mhp_result.primary_resistance:
                ax1.axhline(mhp_result.primary_resistance, color='darkred', linestyle=':',
                           linewidth=2, label=f'MHP Resistance: ${mhp_result.primary_resistance:.2f}')

        # HG Levels
        if hg_result:
            if hg_result.nearest_hg_above:
                ax1.axhline(hg_result.nearest_hg_above, color='orange', linestyle='-.',
                           linewidth=1.5, label=f'HG Above: ${hg_result.nearest_hg_above:.2f}')
            if hg_result.nearest_hg_below:
                ax1.axhline(hg_result.nearest_hg_below, color='purple', linestyle='-.',
                           linewidth=1.5, label=f'HG Below: ${hg_result.nearest_hg_below:.2f}')

        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title('30-Day Price Chart with Institutional Levels', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. Level Distances from Spot
        ax2 = fig.add_subplot(gs[1, 0])

        levels = []
        distances = []
        colors = []
        names = []

        if hp_result:
            if hp_result.key_support:
                dist = (hp_result.key_support - spot_price) / spot_price * 100
                levels.append(hp_result.key_support)
                distances.append(dist)
                colors.append('green')
                names.append('HP Support')

            if hp_result.key_resistance:
                dist = (hp_result.key_resistance - spot_price) / spot_price * 100
                levels.append(hp_result.key_resistance)
                distances.append(dist)
                colors.append('red')
                names.append('HP Resist')

        if mhp_result:
            if mhp_result.primary_support:
                dist = (mhp_result.primary_support - spot_price) / spot_price * 100
                levels.append(mhp_result.primary_support)
                distances.append(dist)
                colors.append('darkgreen')
                names.append('MHP Support')

            if mhp_result.primary_resistance:
                dist = (mhp_result.primary_resistance - spot_price) / spot_price * 100
                levels.append(mhp_result.primary_resistance)
                distances.append(dist)
                colors.append('darkred')
                names.append('MHP Resist')

        if hg_result:
            if hg_result.nearest_hg_above:
                dist = (hg_result.nearest_hg_above - spot_price) / spot_price * 100
                levels.append(hg_result.nearest_hg_above)
                distances.append(dist)
                colors.append('orange')
                names.append('HG Above')

            if hg_result.nearest_hg_below:
                dist = (hg_result.nearest_hg_below - spot_price) / spot_price * 100
                levels.append(hg_result.nearest_hg_below)
                distances.append(dist)
                colors.append('purple')
                names.append('HG Below')

        if distances:
            ax2.barh(names, distances, color=colors, alpha=0.7)
            ax2.axvline(0, color='black', linestyle='-', linewidth=2)
            ax2.axvline(-0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5%')
            ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(-2, color='gray', linestyle=':', alpha=0.5, label='±2%')
            ax2.axvline(2, color='gray', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Distance from Spot (%)', fontsize=10)
            ax2.set_title('How Far Are Institutional Levels?\n(Negative = Below Spot, Positive = Above Spot)',
                         fontsize=10, fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='x')

        # 3. HP Net Pressure Summary
        ax3 = fig.add_subplot(gs[1, 1])

        if hp_result:
            hp_net = hp_result.net_hp
            num_levels = len(hp_result.levels)

            # Simple display of net HP
            categories = ['Net HP']
            values = [hp_net]
            bar_colors = ['green' if hp_net > 0 else 'red']

            ax3.bar(categories, values, color=bar_colors, alpha=0.7, width=0.4)
            ax3.axhline(0, color='black', linestyle='-', linewidth=2)
            ax3.set_ylabel('Hedge Pressure Score', fontsize=11)
            ax3.set_title(f'Net Hedge Pressure: {hp_net:+.3f}\n{hp_result.dominant_direction} ({num_levels} levels detected)',
                         fontsize=11, fontweight='bold')
            ax3.set_ylim([min(hp_net * 1.5, -1), max(hp_net * 1.5, 1)])
            ax3.grid(True, alpha=0.3, axis='y')

            # Add interpretation text
            interpretation = "Bullish" if hp_net > 0.1 else "Bearish" if hp_net < -0.1 else "Neutral"
            ax3.text(0, hp_net * 0.5, interpretation, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

        # 4. Model Predictions Near Levels
        ax4 = fig.add_subplot(gs[2, :])

        # Load model and get predictions at different price levels
        model_path = Path(__file__).parent.parent / f"models/trained/{symbol}_ensemble.pkl"
        if model_path.exists():
            ensemble = SVMEnsemble.load(str(model_path))

            # Build dataset
            builder = TrainingDataBuilder(symbol)
            daily_data = builder.load_historical_data()
            quotes = builder.create_synthetic_quotes(daily_data)
            X, y_returns, feature_names = builder.build_feature_dataset(quotes)

            # Get predictions
            y_pred = ensemble.predict_batch(X)

            # Find support/resistance distance features
            hp_support_idx = None
            hp_resist_idx = None
            mhp_support_idx = None
            mhp_resist_idx = None

            for i, name in enumerate(feature_names):
                if name == 'inst_hp_support_dist':
                    hp_support_idx = i
                elif name == 'inst_hp_resist_dist':
                    hp_resist_idx = i
                elif name == 'inst_mhp_support_dist':
                    mhp_support_idx = i
                elif name == 'inst_mhp_resist_dist':
                    mhp_resist_idx = i

            # Analyze predictions by distance to HP support
            if hp_support_idx is not None:
                hp_support_dist = X[:, hp_support_idx]

                # Create distance bins
                very_close = hp_support_dist < 0.001  # <0.1%
                close = (hp_support_dist >= 0.001) & (hp_support_dist < 0.005)  # 0.1-0.5%
                near = (hp_support_dist >= 0.005) & (hp_support_dist < 0.01)  # 0.5-1%
                medium = (hp_support_dist >= 0.01) & (hp_support_dist < 0.02)  # 1-2%
                far = hp_support_dist >= 0.02  # >2%

                distance_labels = ['Very Close\n(<0.1%)', 'Close\n(0.1-0.5%)', 'Near\n(0.5-1%)',
                                  'Medium\n(1-2%)', 'Far\n(>2%)']
                masks = [very_close, close, near, medium, far]

                up_pcts = []
                down_pcts = []
                neutral_pcts = []
                counts = []

                for mask in masks:
                    if np.sum(mask) > 0:
                        up_pct = np.sum(y_pred[mask] == 1) / np.sum(mask) * 100
                        down_pct = np.sum(y_pred[mask] == -1) / np.sum(mask) * 100
                        neutral_pct = np.sum(y_pred[mask] == 0) / np.sum(mask) * 100

                        up_pcts.append(up_pct)
                        down_pcts.append(down_pct)
                        neutral_pcts.append(neutral_pct)
                        counts.append(np.sum(mask))
                    else:
                        up_pcts.append(0)
                        down_pcts.append(0)
                        neutral_pcts.append(0)
                        counts.append(0)

                x = np.arange(len(distance_labels))
                width = 0.25

                ax4.bar(x - width, up_pcts, width, label='UP Predictions', color='green', alpha=0.7)
                ax4.bar(x, neutral_pcts, width, label='NEUTRAL', color='gray', alpha=0.7)
                ax4.bar(x + width, down_pcts, width, label='DOWN Predictions', color='red', alpha=0.7)

                ax4.set_xlabel('Distance to HP Support', fontsize=11)
                ax4.set_ylabel('Prediction Distribution (%)', fontsize=11)
                ax4.set_title('Model Predictions vs Distance to HP Support\n(Expectation: More UP predictions when close to support)',
                             fontsize=11, fontweight='bold')
                ax4.set_xticks(x)
                ax4.set_xticklabels(distance_labels, fontsize=9)
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')

                # Add sample counts
                for i, count in enumerate(counts):
                    ax4.text(i, max(up_pcts + down_pcts + neutral_pcts) + 2,
                            f'n={count}', ha='center', va='bottom', fontsize=8)

        # 5. MHP Summary Stats
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.axis('off')

        if mhp_result:
            support_val = f"${mhp_result.primary_support:.2f}" if mhp_result.primary_support else "None"
            support_dist = f"{(mhp_result.primary_support - spot_price) / spot_price * 100:.2f}%" if mhp_result.primary_support else "N/A"
            resist_val = f"${mhp_result.primary_resistance:.2f}" if mhp_result.primary_resistance else "None"
            resist_dist = f"{(mhp_result.primary_resistance - spot_price) / spot_price * 100:.2f}%" if mhp_result.primary_resistance else "N/A"

            gamma_zone = f"${mhp_result.gamma_flip_zone[0]:.2f} - ${mhp_result.gamma_flip_zone[1]:.2f}" if mhp_result.gamma_flip_zone else "Not defined"
            in_zone = "INSIDE" if mhp_result.gamma_flip_zone and mhp_result.gamma_flip_zone[0] <= spot_price <= mhp_result.gamma_flip_zone[1] else "OUTSIDE"

            mhp_summary = f"""
MHP (MONTHLY HEDGE PRESSURE)
{'='*35}

MHP Score: {mhp_result.mhp_score:.3f}
  {"Bullish" if mhp_result.mhp_score > 0.1 else "Bearish" if mhp_result.mhp_score < -0.1 else "Neutral"}

Primary Support:  {support_val}
  Distance: {support_dist}

Primary Resistance: {resist_val}
  Distance: {resist_dist}

Gamma Flip Zone:
  {gamma_zone}
  Currently {in_zone}

Expirations Analyzed: {len(mhp_result.expiration_scores) if hasattr(mhp_result, 'expiration_scores') and mhp_result.expiration_scores else 0}
"""
        else:
            mhp_summary = "\nMHP data not available"

        ax5.text(0.05, 0.95, mhp_summary, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 6. HG Summary Stats
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.axis('off')

        if hg_result:
            above_val = f"${hg_result.nearest_hg_above:.2f}" if hg_result.nearest_hg_above else "None"
            above_dist = f"{(hg_result.nearest_hg_above - spot_price) / spot_price * 100:.2f}%" if hg_result.nearest_hg_above else "N/A"
            below_val = f"${hg_result.nearest_hg_below:.2f}" if hg_result.nearest_hg_below else "None"
            below_dist = f"{(hg_result.nearest_hg_below - spot_price) / spot_price * 100:.2f}%" if hg_result.nearest_hg_below else "N/A"

            all_gaps = hg_result.all_half_gaps if hasattr(hg_result, 'all_half_gaps') and hg_result.all_half_gaps else []
            gap_list = ', '.join([f"${hg:.2f}" for hg in sorted(all_gaps)[:5]]) if all_gaps else "None"

            hg_summary = f"""
HALF GAP (PRICE LEVELS)
{'='*35}

Nearest HG Above: {above_val}
  Distance: {above_dist}

Nearest HG Below:  {below_val}
  Distance: {below_dist}

All HG Levels ({len(all_gaps)} total):
  {gap_list}
  ...

HG levels represent unfilled
price gaps from daily OHLC data.
Price tends to "fill the gap" by
returning to these levels.
"""
        else:
            hg_summary = "\nHG data not available"

        ax6.text(0.05, 0.95, hg_summary, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
        else:
            plt.show()

        return fig


def main():
    """Generate institutional levels visualization"""
    visualizer = InstitutionalLevelsVisualizer()

    output_dir = Path(__file__).parent.parent / "visualization/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ["SPY", "QQQ"]:
        print(f"\n{'='*80}")
        print(f"GENERATING INSTITUTIONAL LEVELS: {symbol}")
        print(f"{'='*80}")

        inst_path = output_dir / f"{symbol}_institutional_levels.png"
        try:
            visualizer.plot_institutional_levels(symbol, str(inst_path))
        except Exception as e:
            print(f"[ERROR] Failed to generate visualization: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"[OK] All visualizations saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
