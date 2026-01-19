"""
Real Intraday Data Downloader
Downloads actual 1-minute bars from Tradier (not synthetic)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import tradier_config
from data.tradier_client import TradierClient


class IntradayDataDownloader:
    """
    Download real intraday data from Tradier

    Tradier provides:
    - 1-minute bars
    - Up to 30 days of history
    - Real tick-level data (not synthetic)
    """

    def __init__(self):
        self.client = TradierClient()

    def download_intraday_history(
        self,
        symbol: str,
        days_back: int = 30,
        interval: str = '1min'
    ) -> pd.DataFrame:
        """
        Download intraday bars

        Args:
            symbol: Ticker symbol
            days_back: How many days to download (max 30 for Tradier)
            interval: '1min', '5min', or '15min'

        Returns:
            DataFrame with OHLCV data at specified interval
        """
        print(f"\n{'='*80}")
        print(f"DOWNLOADING REAL INTRADAY DATA: {symbol}")
        print(f"{'='*80}")
        print(f"Interval: {interval}")
        print(f"Days: {days_back}")

        # Tradier intraday history parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"Date range: {start_date.date()} to {end_date.date()}")

        all_bars = []

        # Download in chunks (Tradier has rate limits)
        chunk_days = 5  # Download 5 days at a time

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            print(f"\nFetching {current_start.date()} to {current_end.date()}...")

            try:
                # Use Tradier timesales endpoint for intraday
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'start': current_start.strftime('%Y-%m-%d'),
                    'end': current_end.strftime('%Y-%m-%d')
                }

                data = self.client._request("/markets/timesales", params)

                if 'series' in data and data['series']:
                    bars = data['series'].get('data', [])

                    if bars:
                        df_chunk = pd.DataFrame(bars)
                        all_bars.append(df_chunk)
                        print(f"  Retrieved {len(df_chunk)} bars")
                    else:
                        print(f"  No data available")
                else:
                    print(f"  No data available")

            except Exception as e:
                print(f"  Error: {e}")

            current_start = current_end
            time.sleep(1)  # Rate limiting

        if not all_bars:
            print("\n[WARNING] No intraday data retrieved")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_bars, ignore_index=True)

        # Convert timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.set_index('timestamp')
            df = df.sort_index()

        print(f"\n[OK] Downloaded {len(df)} total bars")
        print(f"     Date range: {df.index.min()} to {df.index.max()}")

        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save downloaded data to CSV"""
        output_dir = Path(__file__).parent.parent / "data_local/intraday"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"

        df.to_csv(filename)
        print(f"\n[OK] Saved to {filename}")
        print(f"     Size: {len(df)} bars")

        return filename

    def create_training_windows(
        self,
        df: pd.DataFrame,
        window_size: int = 5,
        lookforward: int = 5
    ) -> pd.DataFrame:
        """
        Create training windows from intraday data

        Returns DataFrame with features and targets aligned
        """
        print(f"\nCreating training windows...")
        print(f"  Window size: {window_size} bars")
        print(f"  Lookforward: {lookforward} bars")

        if len(df) < window_size + lookforward:
            print("[ERROR] Not enough data")
            return pd.DataFrame()

        windows = []

        for i in range(len(df) - window_size - lookforward):
            window = df.iloc[i:i+window_size]
            future_bar = df.iloc[i+window_size+lookforward-1]

            # Calculate forward return
            current_price = window['close'].iloc[-1]
            future_price = future_bar['close']
            forward_return = (future_price - current_price) / current_price

            windows.append({
                'timestamp': window.index[-1],
                'current_price': current_price,
                'future_price': future_price,
                'forward_return': forward_return,
                'window_data': window  # Store for feature extraction
            })

        print(f"[OK] Created {len(windows)} training windows")

        return pd.DataFrame(windows)


def download_all_symbols():
    """Download intraday data for all symbols"""
    print("\n" + "="*80)
    print("REAL INTRADAY DATA DOWNLOADER")
    print("="*80)

    if not tradier_config.api_key:
        print("[ERROR] TRADIER_API_KEY not set")
        return

    downloader = IntradayDataDownloader()

    symbols = ["SPY", "QQQ"]
    intervals = ['1min', '5min', '15min']

    for symbol in symbols:
        for interval in intervals:
            try:
                # Download data
                df = downloader.download_intraday_history(
                    symbol=symbol,
                    days_back=30,
                    interval=interval
                )

                if not df.empty:
                    # Save to CSV
                    downloader.save_to_csv(df, symbol, interval)

                    # Create sample windows
                    windows = downloader.create_training_windows(
                        df, window_size=5, lookforward=5
                    )

                    if not windows.empty:
                        print(f"\n  Sample forward returns:")
                        print(f"    Mean: {windows['forward_return'].mean():.6f}")
                        print(f"    Std: {windows['forward_return'].std():.6f}")
                        print(f"    Range: [{windows['forward_return'].min():.6f}, {windows['forward_return'].max():.6f}]")

                print("\n" + "="*80)
                time.sleep(2)  # Rate limiting between symbols

            except Exception as e:
                print(f"\n[ERROR] Failed to download {symbol} {interval}: {e}")
                import traceback
                traceback.print_exc()

    print("\n[COMPLETE] Intraday data download finished")
    print(f"Data saved to: data_local/intraday/")


if __name__ == "__main__":
    download_all_symbols()
