"""
Module Validation Script
Tests Feature Pipeline and Institutional Layer with live and historical data
"""
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder
from institutional.hedge_pressure import HedgePressureCalculator
from institutional.monthly_hp import MonthlyHPCalculator
from institutional.half_gap import HalfGapCalculator


class ValidationReport:
    """Collect and format validation results"""

    def __init__(self):
        self.sections = []

    def add_section(self, title: str, content: str):
        """Add a section to the report"""
        self.sections.append({
            'title': title,
            'content': content,
            'timestamp': datetime.now()
        })

    def save(self, filename: str = "validation_report.txt"):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODULE VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for section in self.sections:
                f.write("\n" + "="*80 + "\n")
                f.write(f"{section['title']}\n")
                f.write("="*80 + "\n")
                f.write(section['content'])
                f.write("\n")

        print(f"\n[OK] Validation report saved to {filename}")


def validate_feature_pipeline(client: TradierClient, report: ValidationReport):
    """Test Feature Pipeline with live data"""
    print("\n" + "="*80)
    print("VALIDATING FEATURE PIPELINE")
    print("="*80)

    results = []

    for symbol in ["SPY", "QQQ"]:
        print(f"\nTesting {symbol}...")

        # Initialize components
        cleaner = DataCleaner()
        window_builder = EventWindowBuilder()
        feature_builder = FeatureMatrixBuilder(client)

        # Collect quotes for one window
        quotes = []
        print(f"Collecting {model_config.window_size} quotes...")

        for i in range(model_config.window_size):
            quote = client.get_quote(symbol)
            if quote:
                quotes.append(quote)
                print(f"  [{i+1}/{model_config.window_size}] {symbol}: ${quote.mid_price:.2f} | Spread: {quote.spread_pct*100:.3f}%")
            time.sleep(1)

        # Test cleaner
        cleaned = cleaner.clean_quotes(quotes)
        clean_rate = len(cleaned) / len(quotes) if quotes else 0

        # Build window
        window = None
        for q in cleaned:
            window = window_builder.add_event(q)
            if window:
                break

        if window:
            # Build features
            feature_array, feature_dict = feature_builder.build_feature_vector(
                symbol, window, cleaned[-1]
            )

            results.append({
                'symbol': symbol,
                'quotes_collected': len(quotes),
                'quotes_cleaned': len(cleaned),
                'clean_rate': clean_rate,
                'window_built': True,
                'feature_count': len(feature_dict),
                'feature_array_shape': feature_array.shape,
                'sample_features': {k: v for k, v in list(feature_dict.items())[:5]}
            })
        else:
            results.append({
                'symbol': symbol,
                'quotes_collected': len(quotes),
                'quotes_cleaned': len(cleaned),
                'clean_rate': clean_rate,
                'window_built': False,
                'error': 'Failed to build window'
            })

    # Format report
    content = ""
    for r in results:
        content += f"\nSymbol: {r['symbol']}\n"
        content += f"  Quotes Collected: {r['quotes_collected']}\n"
        content += f"  Quotes Cleaned: {r['quotes_cleaned']} ({r['clean_rate']*100:.1f}%)\n"

        if r.get('window_built'):
            content += f"  Window Built: YES\n"
            content += f"  Feature Count: {r['feature_count']}\n"
            content += f"  Feature Array Shape: {r['feature_array_shape']}\n"
            content += f"  Sample Features:\n"
            for k, v in r['sample_features'].items():
                if isinstance(v, float):
                    content += f"    {k}: {v:.6f}\n"
                else:
                    content += f"    {k}: {v}\n"
        else:
            content += f"  Window Built: NO\n"
            content += f"  Error: {r.get('error', 'Unknown')}\n"

    # Overall assessment
    success_rate = sum(1 for r in results if r.get('window_built')) / len(results)
    content += f"\n\nOVERALL FEATURE PIPELINE STATUS:\n"
    content += f"  Success Rate: {success_rate*100:.0f}%\n"
    content += f"  Status: {'PASS' if success_rate == 1.0 else 'PARTIAL' if success_rate > 0 else 'FAIL'}\n"

    report.add_section("Feature Pipeline Validation", content)

    return success_rate == 1.0


def validate_institutional_layer(client: TradierClient, report: ValidationReport):
    """Validate HP/MHP/HG calculations"""
    print("\n" + "="*80)
    print("VALIDATING INSTITUTIONAL LAYER")
    print("="*80)

    results = []

    for symbol in ["SPY", "QQQ"]:
        print(f"\nValidating {symbol}...")

        # Get current quote
        quote = client.get_quote(symbol)
        if not quote:
            print(f"  [ERROR] Failed to get quote for {symbol}")
            continue

        spot = quote.mid_price
        print(f"  Current Price: ${spot:.2f}")

        # Test HP
        print("  Testing Hedge Pressure...")
        hp_calc = HedgePressureCalculator()
        chains = client.get_multiple_chains(symbol, expiration_count=1)

        if chains:
            exp, chain = list(chains.items())[0]
            hp_result = hp_calc.calculate(chain, spot, symbol)

            hp_data = {
                'net_hp': hp_result.net_hp,
                'direction': hp_result.dominant_direction,
                'key_support': hp_result.key_support,
                'key_resistance': hp_result.key_resistance,
                'level_count': len(hp_result.levels),
                'top_level_confidence': hp_result.levels[0].confidence if hp_result.levels else 0
            }
        else:
            hp_data = {'error': 'No chains available'}

        # Test MHP
        print("  Testing Monthly Hedge Pressure...")
        mhp_calc = MonthlyHPCalculator()
        chains_multi = client.get_multiple_chains(symbol, expiration_count=4)

        if chains_multi:
            mhp_result = mhp_calc.calculate(chains_multi, spot, symbol)

            mhp_data = {
                'mhp_score': mhp_result.mhp_score,
                'primary_support': mhp_result.primary_support,
                'primary_resistance': mhp_result.primary_resistance,
                'gamma_flip_zone': mhp_result.gamma_flip_zone,
                'expiration_count': len(mhp_result.expiration_results)
            }
        else:
            mhp_data = {'error': 'No chains available'}

        # Test HG
        print("  Testing Half Gap...")
        hg_calc = HalfGapCalculator()
        history = client.get_history(symbol, interval="daily")

        if not history.empty:
            hg_result = hg_calc.calculate(history, spot, symbol)

            hg_data = {
                'active_half_gaps': len(hg_result.active_half_gaps),
                'nearest_above': hg_result.nearest_hg_above,
                'nearest_below': hg_result.nearest_hg_below,
                'gap_magnet_zone': hg_result.gap_magnet_zone
            }
        else:
            hg_data = {'error': 'No history available'}

        results.append({
            'symbol': symbol,
            'spot_price': spot,
            'hp': hp_data,
            'mhp': mhp_data,
            'hg': hg_data
        })

    # Format report
    content = ""

    for r in results:
        content += f"\n{'='*60}\n"
        content += f"Symbol: {r['symbol']} (Spot: ${r['spot_price']:.2f})\n"
        content += f"{'='*60}\n"

        # HP Results
        content += f"\nHedge Pressure (HP):\n"
        if 'error' not in r['hp']:
            content += f"  Net HP: {r['hp']['net_hp']:.3f} ({r['hp']['direction']})\n"
            content += f"  Key Support: ${r['hp']['key_support']:.2f}\n" if r['hp']['key_support'] else "  Key Support: None\n"
            content += f"  Key Resistance: ${r['hp']['key_resistance']:.2f}\n" if r['hp']['key_resistance'] else "  Key Resistance: None\n"
            content += f"  Levels Identified: {r['hp']['level_count']}\n"
            content += f"  Top Level Confidence: {r['hp']['top_level_confidence']:.2f}\n"

            # Validate HP is reasonable
            hp_valid = (
                abs(r['hp']['net_hp']) <= 1.0 and
                r['hp']['level_count'] > 0 and
                r['hp']['top_level_confidence'] > 0
            )
            content += f"  Validation: {'PASS' if hp_valid else 'QUESTIONABLE'}\n"
        else:
            content += f"  Error: {r['hp']['error']}\n"
            content += f"  Validation: FAIL\n"

        # MHP Results
        content += f"\nMonthly Hedge Pressure (MHP):\n"
        if 'error' not in r['mhp']:
            content += f"  MHP Score: {r['mhp']['mhp_score']:.3f}\n"
            content += f"  Primary Support: ${r['mhp']['primary_support']:.2f}\n" if r['mhp']['primary_support'] else "  Primary Support: None\n"
            content += f"  Primary Resistance: ${r['mhp']['primary_resistance']:.2f}\n" if r['mhp']['primary_resistance'] else "  Primary Resistance: None\n"

            if r['mhp']['gamma_flip_zone']:
                low, high = r['mhp']['gamma_flip_zone']
                content += f"  Gamma Flip Zone: ${low:.2f} - ${high:.2f}\n"
            else:
                content += f"  Gamma Flip Zone: None\n"

            content += f"  Expirations Used: {r['mhp']['expiration_count']}\n"

            # Validate MHP
            mhp_valid = (
                abs(r['mhp']['mhp_score']) <= 1.0 and
                r['mhp']['expiration_count'] >= 2
            )
            content += f"  Validation: {'PASS' if mhp_valid else 'QUESTIONABLE'}\n"
        else:
            content += f"  Error: {r['mhp']['error']}\n"
            content += f"  Validation: FAIL\n"

        # HG Results
        content += f"\nHalf Gap (HG):\n"
        if 'error' not in r['hg']:
            content += f"  Active Half Gaps: {r['hg']['active_half_gaps']}\n"
            content += f"  Nearest Above: ${r['hg']['nearest_above']:.2f}\n" if r['hg']['nearest_above'] else "  Nearest Above: None\n"
            content += f"  Nearest Below: ${r['hg']['nearest_below']:.2f}\n" if r['hg']['nearest_below'] else "  Nearest Below: None\n"

            if r['hg']['gap_magnet_zone']:
                low, high = r['hg']['gap_magnet_zone']
                content += f"  Gap Magnet Zone: ${low:.2f} - ${high:.2f}\n"
            else:
                content += f"  Gap Magnet Zone: None\n"

            content += f"  Validation: PASS\n"
        else:
            content += f"  Error: {r['hg']['error']}\n"
            content += f"  Validation: FAIL\n"

    # Overall assessment
    hp_pass = all('error' not in r['hp'] for r in results)
    mhp_pass = all('error' not in r['mhp'] for r in results)
    hg_pass = all('error' not in r['hg'] for r in results)

    content += f"\n\n{'='*60}\n"
    content += f"OVERALL INSTITUTIONAL LAYER STATUS:\n"
    content += f"{'='*60}\n"
    content += f"  HP Calculator: {'PASS' if hp_pass else 'FAIL'}\n"
    content += f"  MHP Calculator: {'PASS' if mhp_pass else 'FAIL'}\n"
    content += f"  HG Calculator: {'PASS' if hg_pass else 'FAIL'}\n"
    content += f"  Overall: {'PASS' if all([hp_pass, mhp_pass, hg_pass]) else 'PARTIAL' if any([hp_pass, mhp_pass, hg_pass]) else 'FAIL'}\n"

    report.add_section("Institutional Layer Validation", content)

    return all([hp_pass, mhp_pass, hg_pass])


def validate_data_availability(report: ValidationReport):
    """Check downloaded data files"""
    print("\n" + "="*80)
    print("VALIDATING DATA AVAILABILITY")
    print("="*80)

    content = ""

    # Check price history
    price_path = Path("data_local/price_history")
    if price_path.exists():
        price_files = list(price_path.glob("*.csv"))
        content += f"\nPrice History Files: {len(price_files)}\n"

        for f in price_files:
            df = pd.read_csv(f)
            content += f"  - {f.name}: {len(df)} rows\n"
    else:
        content += "\n[WARNING] Price history folder not found\n"

    # Check options chains
    content += f"\nOptions Chains:\n"
    for symbol in ["SPY", "QQQ"]:
        options_path = Path(f"data_local/options_chains/{symbol}")
        if options_path.exists():
            chain_files = list(options_path.glob("*.csv"))
            total_options = sum(len(pd.read_csv(f)) for f in chain_files)
            content += f"  {symbol}: {len(chain_files)} chains, {total_options:,} total options\n"
        else:
            content += f"  {symbol}: [WARNING] Folder not found\n"

    report.add_section("Data Availability", content)


def main():
    """Run full validation suite"""
    print("\n" + "="*80)
    print("MODULE VALIDATION SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check API key
    if not tradier_config.api_key:
        print("[ERROR] TRADIER_API_KEY not set")
        sys.exit(1)

    # Initialize
    client = TradierClient()
    report = ValidationReport()

    # Run validations
    validate_data_availability(report)

    feature_pass = validate_feature_pipeline(client, report)
    print(f"\n[{'PASS' if feature_pass else 'FAIL'}] Feature Pipeline")

    institutional_pass = validate_institutional_layer(client, report)
    print(f"\n[{'PASS' if institutional_pass else 'FAIL'}] Institutional Layer")

    # Final summary
    content = f"\nFeature Pipeline: {'PASS' if feature_pass else 'FAIL'}\n"
    content += f"Institutional Layer: {'PASS' if institutional_pass else 'FAIL'}\n\n"

    if feature_pass and institutional_pass:
        content += "STATUS: READY FOR SVM ENSEMBLE BUILD\n"
        content += "\nNext Steps:\n"
        content += "  1. Build SVM ensemble module\n"
        content += "  2. Train on historical data\n"
        content += "  3. Integrate signal generation\n"
    elif feature_pass or institutional_pass:
        content += "STATUS: PARTIAL - Review failed modules\n"
    else:
        content += "STATUS: NOT READY - Fix critical issues\n"

    report.add_section("Final Summary", content)

    # Save report
    report.save("data_local/validation_report.txt")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nOverall Status: {'PASS' if feature_pass and institutional_pass else 'PARTIAL' if feature_pass or institutional_pass else 'FAIL'}")
    print(f"Report: data_local/validation_report.txt\n")


if __name__ == "__main__":
    main()
