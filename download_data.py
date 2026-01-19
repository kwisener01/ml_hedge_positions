"""
Download and save historical market data and options chains
Creates local CSV files for offline analysis
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from config.settings import tradier_config
from data.tradier_client import TradierClient


def create_data_folders():
    """Create folder structure for data storage"""
    folders = [
        "data_local",
        "data_local/price_history",
        "data_local/options_chains",
        "data_local/options_chains/SPY",
        "data_local/options_chains/QQQ"
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created folder: {folder}")


def download_price_history(client: TradierClient, symbol: str, years_back: int = 5):
    """
    Download maximum available historical price data

    Args:
        client: TradierClient instance
        symbol: Ticker symbol
        years_back: How many years of data to download
    """
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} Price History")
    print(f"{'='*60}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    intervals = ["daily", "weekly", "monthly"]

    for interval in intervals:
        print(f"\nFetching {interval} data...")

        df = client.get_history(
            symbol=symbol,
            interval=interval,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )

        if not df.empty:
            filename = f"data_local/price_history/{symbol}_{interval}.csv"
            df.to_csv(filename)
            print(f"  [OK] Saved {len(df)} rows to {filename}")
            print(f"    Date range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"  [SKIP] No data available for {interval}")


def download_options_chains(client: TradierClient, symbol: str, max_expirations: int = 20):
    """
    Download all available options chains

    Args:
        client: TradierClient instance
        symbol: Ticker symbol
        max_expirations: Maximum number of expiration dates to download
    """
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} Options Chains")
    print(f"{'='*60}")

    # Get all available expirations
    print("\nFetching available expiration dates...")
    expirations = client.get_expirations(symbol)

    if not expirations:
        print(f"[SKIP] No options expirations available for {symbol}")
        return

    print(f"Found {len(expirations)} expiration dates")
    print(f"Will download first {min(max_expirations, len(expirations))} chains")

    # Download each chain
    chains_downloaded = 0
    total_options = 0

    for i, expiration in enumerate(expirations[:max_expirations]):
        print(f"\n[{i+1}/{min(max_expirations, len(expirations))}] Fetching chain for {expiration}...")

        try:
            df = client.get_options_chain(symbol, expiration, greeks=True)

            if not df.empty:
                # Save to CSV
                filename = f"data_local/options_chains/{symbol}/{symbol}_{expiration}.csv"
                df.to_csv(filename, index=False)

                chains_downloaded += 1
                total_options += len(df)

                # Show summary stats
                calls = len(df[df["option_type"] == "call"])
                puts = len(df[df["option_type"] == "put"])

                print(f"  [OK] Saved {len(df)} options ({calls} calls, {puts} puts)")
                print(f"    File: {filename}")
            else:
                print(f"  [SKIP] No data for {expiration}")

        except Exception as e:
            print(f"  [ERROR] Error downloading {expiration}: {e}")

    print(f"\n{'='*60}")
    print(f"Summary: {chains_downloaded} chains downloaded, {total_options} total options")
    print(f"{'='*60}")


def create_master_summary():
    """Create a summary file of all downloaded data"""
    print(f"\n{'='*60}")
    print("Creating Master Summary")
    print(f"{'='*60}")

    summary = {
        "download_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price_history_files": [],
        "options_chains_summary": {}
    }

    # Count price history files
    price_folder = Path("data_local/price_history")
    if price_folder.exists():
        for csv_file in price_folder.glob("*.csv"):
            df = pd.read_csv(csv_file)
            summary["price_history_files"].append({
                "file": csv_file.name,
                "rows": len(df),
                "columns": list(df.columns)
            })

    # Count options chains
    for symbol in ["SPY", "QQQ"]:
        options_folder = Path(f"data_local/options_chains/{symbol}")
        if options_folder.exists():
            chain_files = list(options_folder.glob("*.csv"))
            total_options = 0

            for chain_file in chain_files:
                df = pd.read_csv(chain_file)
                total_options += len(df)

            summary["options_chains_summary"][symbol] = {
                "chain_count": len(chain_files),
                "total_options": total_options,
                "expirations": [f.stem.split("_", 1)[1] for f in chain_files]
            }

    # Save summary
    with open("data_local/download_summary.txt", "w") as f:
        f.write(f"Data Download Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {summary['download_timestamp']}\n\n")

        f.write("Price History:\n")
        for item in summary["price_history_files"]:
            f.write(f"  - {item['file']}: {item['rows']} rows\n")

        f.write(f"\nOptions Chains:\n")
        for symbol, data in summary["options_chains_summary"].items():
            f.write(f"  {symbol}:\n")
            f.write(f"    - Chain files: {data['chain_count']}\n")
            f.write(f"    - Total options: {data['total_options']}\n")
            f.write(f"    - Expirations: {', '.join(data['expirations'][:5])}")
            if len(data['expirations']) > 5:
                f.write(f" ... +{len(data['expirations']) - 5} more")
            f.write("\n")

    print("[OK] Summary saved to data_local/download_summary.txt")


def main():
    """Main download orchestrator"""
    print("\n" + "="*60)
    print("MARKET DATA DOWNLOADER")
    print("="*60)

    # Check API key
    if not tradier_config.api_key:
        print("ERROR: Set TRADIER_API_KEY environment variable")
        sys.exit(1)

    print(f"API Key configured: {tradier_config.api_key[:8]}...")

    # Create folders
    print("\n--- Creating Data Folders ---")
    create_data_folders()

    # Initialize client
    client = TradierClient()

    # Download data for each symbol
    symbols = ["SPY", "QQQ"]

    for symbol in symbols:
        # Download price history
        download_price_history(client, symbol, years_back=5)

        # Download options chains
        download_options_chains(client, symbol, max_expirations=20)

    # Create summary
    create_master_summary()

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nData saved to:")
    print("  - data_local/price_history/     (historical prices)")
    print("  - data_local/options_chains/    (options data)")
    print("  - data_local/download_summary.txt (summary)")
    print()


if __name__ == "__main__":
    main()
