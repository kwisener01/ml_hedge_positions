"""
Main entry point - demonstrates full pipeline
"""
import sys
import time
from datetime import datetime

from config.settings import tradier_config, model_config
from data.tradier_client import TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.feature_matrix import FeatureMatrixBuilder, TargetCalculator
from institutional.hedge_pressure import HedgePressureCalculator
from institutional.monthly_hp import MonthlyHPCalculator
from institutional.half_gap import HalfGapCalculator

def run_institutional_analysis(symbol: str = "SPY"):
    """
    Run complete institutional analysis for a symbol
    """
    print(f"\n{'='*60}")
    print(f"INSTITUTIONAL ANALYSIS: {symbol}")
    print(f"{'='*60}\n")
    
    client = TradierClient()
    
    # Get current quote
    quote = client.get_quote(symbol)
    if not quote:
        print("Failed to get quote")
        return
    
    spot = quote.mid_price
    print(f"Current Price: ${spot:.2f}")
    print(f"Spread: ${quote.spread:.4f} ({quote.spread_pct*100:.3f}%)")
    
    # Get options chains
    print("\nFetching options chains...")
    chains = client.get_multiple_chains(symbol, expiration_count=4)
    print(f"Retrieved {len(chains)} expiration chains")
    
    # Calculate HP
    print("\n--- HEDGE PRESSURE (HP) ---")
    hp_calc = HedgePressureCalculator()
    
    for exp, chain in list(chains.items())[:1]:  # Just nearest
        hp_result = hp_calc.calculate(chain, spot, symbol)
        print(f"\nExpiration: {exp}")
        print(f"Net HP: {hp_result.net_hp:.3f} ({hp_result.dominant_direction})")
        print(f"Key Support: ${hp_result.key_support:.2f}" if hp_result.key_support else "Key Support: None")
        print(f"Key Resistance: ${hp_result.key_resistance:.2f}" if hp_result.key_resistance else "Key Resistance: None")
        
        # Top 5 levels
        print("\nTop HP Levels:")
        for i, level in enumerate(hp_result.levels[:5]):
            print(f"  {i+1}. ${level.strike:.2f} | {level.direction:10} | "
                  f"Conf: {level.confidence:.2f} | OI: {level.open_interest:,}")
    
    # Calculate MHP
    print("\n--- MONTHLY HEDGE PRESSURE (MHP) ---")
    mhp_calc = MonthlyHPCalculator()
    mhp_result = mhp_calc.calculate(chains, spot, symbol)
    
    print(f"MHP Score: {mhp_result.mhp_score:.3f}")
    print(f"Primary Support: ${mhp_result.primary_support:.2f}" if mhp_result.primary_support else "Primary Support: None")
    print(f"Primary Resistance: ${mhp_result.primary_resistance:.2f}" if mhp_result.primary_resistance else "Primary Resistance: None")
    
    if mhp_result.gamma_flip_zone:
        low, high = mhp_result.gamma_flip_zone
        print(f"Gamma Flip Zone: ${low:.2f} - ${high:.2f}")
    
    # Calculate HG
    print("\n--- HALF GAP (HG) ---")
    history = client.get_history(symbol, interval="daily")
    hg_calc = HalfGapCalculator()
    hg_result = hg_calc.calculate(history, spot, symbol)
    
    print(f"Active Half Gaps: {len(hg_result.active_half_gaps)}")
    print(f"Nearest HG Above: ${hg_result.nearest_hg_above:.2f}" if hg_result.nearest_hg_above else "Nearest HG Above: None")
    print(f"Nearest HG Below: ${hg_result.nearest_hg_below:.2f}" if hg_result.nearest_hg_below else "Nearest HG Below: None")
    
    if hg_result.gap_magnet_zone:
        low, high = hg_result.gap_magnet_zone
        print(f"Gap Magnet Zone: ${low:.2f} - ${high:.2f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CONFLUENCE SUMMARY")
    print(f"{'='*60}")
    
    levels_near_price = []
    
    if hp_result.key_support and abs(hp_result.key_support - spot) / spot < 0.02:
        levels_near_price.append(f"HP Support: ${hp_result.key_support:.2f}")
    if hp_result.key_resistance and abs(hp_result.key_resistance - spot) / spot < 0.02:
        levels_near_price.append(f"HP Resistance: ${hp_result.key_resistance:.2f}")
    if mhp_result.primary_support and abs(mhp_result.primary_support - spot) / spot < 0.02:
        levels_near_price.append(f"MHP Support: ${mhp_result.primary_support:.2f}")
    if mhp_result.primary_resistance and abs(mhp_result.primary_resistance - spot) / spot < 0.02:
        levels_near_price.append(f"MHP Resistance: ${mhp_result.primary_resistance:.2f}")
    
    for hg in hg_result.active_half_gaps:
        if abs(hg - spot) / spot < 0.02:
            levels_near_price.append(f"Half Gap: ${hg:.2f}")
    
    if levels_near_price:
        print("Institutional levels within 2% of current price:")
        for level in levels_near_price:
            print(f"  • {level}")
    else:
        print("No institutional levels within 2% of current price")
    
    return hp_result, mhp_result, hg_result


def run_feature_pipeline(symbol: str = "SPY", collect_seconds: int = 30):
    """
    Demonstrate feature pipeline data collection
    """
    print(f"\n{'='*60}")
    print(f"FEATURE PIPELINE: {symbol}")
    print(f"Collecting data for {collect_seconds} seconds...")
    print(f"{'='*60}\n")
    
    client = TradierClient()
    cleaner = DataCleaner()
    window_builder = EventWindowBuilder()
    feature_builder = FeatureMatrixBuilder(client)
    
    quotes_collected = []
    windows_completed = []
    
    start_time = time.time()
    
    while time.time() - start_time < collect_seconds:
        quote = client.get_quote(symbol)
        
        if quote:
            # Clean the quote
            cleaned = cleaner.clean_quotes([quote])
            
            if cleaned:
                quotes_collected.append(cleaned[0])
                
                # Try to form a window
                window = window_builder.add_event(cleaned[0])
                
                if window:
                    windows_completed.append(window)
                    print(f"Window {len(windows_completed)} completed | "
                          f"Mid: ${window[-1].mid_price:.2f} | "
                          f"Spread: ${window[-1].spread:.4f}")
        
        time.sleep(1)
    
    print(f"\n--- COLLECTION SUMMARY ---")
    print(f"Quotes collected: {len(quotes_collected)}")
    print(f"Quotes rejected: {cleaner.stats.total_records - cleaner.stats.valid_records}")
    print(f"Windows completed: {len(windows_completed)}")
    
    if windows_completed and quotes_collected:
        print(f"\n--- SAMPLE FEATURE VECTOR ---")
        
        feature_array, feature_dict = feature_builder.build_feature_vector(
            symbol,
            windows_completed[-1],
            quotes_collected[-1]
        )
        
        print(f"Feature vector shape: {feature_array.shape}")
        print(f"\nFeature values:")
        for key, value in feature_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    return quotes_collected, windows_completed


if __name__ == "__main__":
    # Check for API key
    if not tradier_config.api_key:
        print("ERROR: Set TRADIER_API_KEY environment variable")
        print("  export TRADIER_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Run institutional analysis
    for symbol in ["SPY", "QQQ"]:
        run_institutional_analysis(symbol)
    
    # Run feature pipeline demo
    # run_feature_pipeline("SPY", collect_seconds=60)