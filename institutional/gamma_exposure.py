"""
Gamma Exposure and Vanna Calculation
Tracks dealer positioning and hedging requirements
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tradier_client import TradierClient
from config.settings import tradier_config


@dataclass
class GammaExposure:
    """Gamma exposure metrics at current price"""
    spot_price: float
    total_gamma_exposure: float  # Net GEX (billions)
    gamma_at_spot: float  # Gamma concentration at current price
    total_vanna: float  # Net vanna exposure
    vanna_at_spot: float  # Vanna at current price

    # Key levels
    zero_gamma_level: Optional[float]  # Price where gamma flips
    max_gamma_strike: float  # Strike with highest gamma
    max_vanna_strike: float  # Strike with highest vanna

    # Directional bias
    gamma_flip_distance: float  # Distance to zero gamma (%)
    vanna_bias: float  # Positive = calls dominate, negative = puts

    # Call/put breakdown
    call_gamma: float
    put_gamma: float
    call_vanna: float
    put_vanna: float

    timestamp: datetime


class GammaVannaCalculator:
    """
    Calculate gamma exposure and vanna from options chain

    Gamma Exposure (GEX):
    - Measures dealer hedging requirements
    - High positive gamma = dealers stabilize price (support)
    - High negative gamma = dealers amplify moves (resistance)
    - Zero gamma = flip point where dealer flow reverses

    Vanna:
    - Measures sensitivity to volatility changes
    - Positive vanna = price up when vol up
    - Important for understanding dealer rebalancing
    """

    def __init__(self, client: TradierClient = None):
        self.client = client or TradierClient()

    def _black_scholes_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes gamma

        Gamma = N'(d1) / (S * sigma * sqrt(T))
        where N'(x) is standard normal PDF
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # Standard normal PDF
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    def _black_scholes_vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes vanna

        Vanna = -N'(d1) * d2 / sigma
        where d2 = d1 - sigma * sqrt(T)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Vanna
        vanna = -norm.pdf(d1) * d2 / sigma

        return vanna

    def calculate_gex_vanna(
        self,
        symbol: str,
        spot_price: float,
        options_df: pd.DataFrame,
        expiration_date: datetime
    ) -> Tuple[Dict[float, Dict], float, float]:
        """
        Calculate gamma and vanna exposure across strike chain

        Args:
            symbol: Ticker symbol
            spot_price: Current underlying price
            options_df: DataFrame of option contracts
            expiration_date: Expiration date

        Returns:
            (strike_data, total_gamma, total_vanna)
            strike_data: {strike: {'gamma': X, 'vanna': Y, 'call_oi': Z, 'put_oi': W}}
        """
        # Time to expiration (years)
        now = datetime.now()
        T = max((expiration_date - now).days / 365.0, 1/365)  # Minimum 1 day

        # Market parameters
        r = 0.045  # Risk-free rate (~4.5%)

        strike_data = {}

        for _, row in options_df.iterrows():
            strike = row.get('strike', 0)
            option_type = row.get('option_type', '')

            # Open interest and volume
            open_interest = row.get('open_interest', 0)
            volume = row.get('volume', 0)

            if open_interest == 0 or pd.isna(open_interest):
                continue

            # Implied volatility
            iv = row.get('greek_mid_iv', 0.25)  # Greeks are flattened with greek_ prefix

            if pd.isna(iv) or iv <= 0:
                iv = 0.25

            # Calculate gamma and vanna
            gamma = self._black_scholes_gamma(spot_price, strike, T, r, iv)
            vanna = self._black_scholes_vanna(spot_price, strike, T, r, iv)

            # Gamma exposure = gamma * open_interest * 100 shares * spot^2
            # Divided by 10^9 for billions
            gex = gamma * open_interest * 100 * spot_price**2 / 1e9

            # Vanna exposure (similar scaling)
            vanna_exp = vanna * open_interest * 100 * spot_price / 1e6

            # Dealers are SHORT options, so flip sign
            # (Positive OI = dealers short = negative gamma for dealers)
            dealer_gex = -gex
            dealer_vanna = -vanna_exp

            if strike not in strike_data:
                strike_data[strike] = {
                    'call_gamma': 0,
                    'put_gamma': 0,
                    'call_vanna': 0,
                    'put_vanna': 0,
                    'call_oi': 0,
                    'put_oi': 0
                }

            if option_type == 'call':
                strike_data[strike]['call_gamma'] += dealer_gex
                strike_data[strike]['call_vanna'] += dealer_vanna
                strike_data[strike]['call_oi'] += open_interest
            else:  # put
                strike_data[strike]['put_gamma'] += dealer_gex
                strike_data[strike]['put_vanna'] += dealer_vanna
                strike_data[strike]['put_oi'] += open_interest

        # Calculate totals
        total_gamma = sum(
            data['call_gamma'] + data['put_gamma']
            for data in strike_data.values()
        )

        total_vanna = sum(
            data['call_vanna'] + data['put_vanna']
            for data in strike_data.values()
        )

        return strike_data, total_gamma, total_vanna

    def find_zero_gamma_level(
        self,
        strike_data: Dict[float, Dict],
        spot_price: float
    ) -> Optional[float]:
        """
        Find the strike where gamma exposure crosses zero
        This is the "gamma flip" level where dealer behavior reverses
        """
        strikes = sorted(strike_data.keys())

        # Calculate cumulative gamma from low to high strikes
        cumulative_gamma = []
        cumulative_strikes = []

        cum_gex = 0
        for strike in strikes:
            data = strike_data[strike]
            cum_gex += data['call_gamma'] + data['put_gamma']
            cumulative_gamma.append(cum_gex)
            cumulative_strikes.append(strike)

        # Find zero crossing
        for i in range(len(cumulative_gamma) - 1):
            if cumulative_gamma[i] <= 0 <= cumulative_gamma[i + 1]:
                # Interpolate
                s1, s2 = cumulative_strikes[i], cumulative_strikes[i + 1]
                g1, g2 = cumulative_gamma[i], cumulative_gamma[i + 1]

                if g2 != g1:
                    zero_strike = s1 + (s2 - s1) * (-g1) / (g2 - g1)
                    return zero_strike

        return None

    def calculate(self, symbol: str) -> Optional[GammaExposure]:
        """
        Calculate complete gamma and vanna exposure for symbol

        Args:
            symbol: Ticker symbol

        Returns:
            GammaExposure object or None if data unavailable
        """
        # Get current price
        try:
            quote = self.client.get_quote(symbol)
            spot_price = quote.last or quote.mid_price
        except Exception as e:
            print(f"[WARN] Failed to get quote for {symbol}: {e}")
            return None

        # Get options expirations
        try:
            expirations = self.client.get_expirations(symbol)
        except Exception as e:
            print(f"[WARN] Failed to get expirations for {symbol}: {e}")
            return None

        if not expirations:
            return None

        # Use front-month expiration (most liquid)
        front_expiry = expirations[0]

        # Get options chain
        try:
            chain_df = self.client.get_options_chain(symbol, front_expiry)
        except Exception as e:
            print(f"[WARN] Failed to get chain for {symbol}: {e}")
            return None

        if chain_df is None or chain_df.empty:
            return None

        # Convert expiration string to datetime
        try:
            expiry_dt = datetime.strptime(front_expiry, '%Y-%m-%d')
        except:
            expiry_dt = datetime.now() + timedelta(days=30)

        # Calculate GEX and vanna
        strike_data, total_gamma, total_vanna = self.calculate_gex_vanna(
            symbol, spot_price, chain_df, expiry_dt
        )

        if not strike_data:
            return None

        # Find gamma and vanna at current spot
        strikes = sorted(strike_data.keys())
        closest_strike = min(strikes, key=lambda k: abs(k - spot_price))

        spot_data = strike_data[closest_strike]
        gamma_at_spot = spot_data['call_gamma'] + spot_data['put_gamma']
        vanna_at_spot = spot_data['call_vanna'] + spot_data['put_vanna']

        # Find zero gamma level
        zero_gamma = self.find_zero_gamma_level(strike_data, spot_price)

        if zero_gamma:
            gamma_flip_distance = (zero_gamma - spot_price) / spot_price
        else:
            gamma_flip_distance = 0.0

        # Find max gamma and vanna strikes
        max_gamma_strike = max(
            strike_data.items(),
            key=lambda x: abs(x[1]['call_gamma'] + x[1]['put_gamma'])
        )[0]

        max_vanna_strike = max(
            strike_data.items(),
            key=lambda x: abs(x[1]['call_vanna'] + x[1]['put_vanna'])
        )[0]

        # Call/put breakdown
        call_gamma = sum(data['call_gamma'] for data in strike_data.values())
        put_gamma = sum(data['put_gamma'] for data in strike_data.values())
        call_vanna = sum(data['call_vanna'] for data in strike_data.values())
        put_vanna = sum(data['put_vanna'] for data in strike_data.values())

        # Vanna bias (positive = calls dominate)
        vanna_bias = call_vanna / (abs(call_vanna) + abs(put_vanna)) if (abs(call_vanna) + abs(put_vanna)) > 0 else 0

        return GammaExposure(
            spot_price=spot_price,
            total_gamma_exposure=total_gamma,
            gamma_at_spot=gamma_at_spot,
            total_vanna=total_vanna,
            vanna_at_spot=vanna_at_spot,
            zero_gamma_level=zero_gamma,
            max_gamma_strike=max_gamma_strike,
            max_vanna_strike=max_vanna_strike,
            gamma_flip_distance=gamma_flip_distance,
            vanna_bias=vanna_bias,
            call_gamma=call_gamma,
            put_gamma=put_gamma,
            call_vanna=call_vanna,
            put_vanna=put_vanna,
            timestamp=datetime.now()
        )


if __name__ == "__main__":
    """Test gamma and vanna calculation"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    calculator = GammaVannaCalculator()

    for symbol in ["SPY", "QQQ"]:
        print(f"\n{'='*80}")
        print(f"GAMMA & VANNA EXPOSURE: {symbol}")
        print(f"{'='*80}")

        gex = calculator.calculate(symbol)

        if gex:
            print(f"\nSpot Price: ${gex.spot_price:.2f}")
            print(f"\nTotal Gamma Exposure: ${gex.total_gamma_exposure:.2f}B")
            print(f"  Call Gamma: ${gex.call_gamma:.2f}B")
            print(f"  Put Gamma: ${gex.put_gamma:.2f}B")
            print(f"  Gamma at Spot: ${gex.gamma_at_spot:.3f}B")

            print(f"\nTotal Vanna: ${gex.total_vanna:.2f}M")
            print(f"  Call Vanna: ${gex.call_vanna:.2f}M")
            print(f"  Put Vanna: ${gex.put_vanna:.2f}M")
            print(f"  Vanna at Spot: ${gex.vanna_at_spot:.3f}M")
            print(f"  Vanna Bias: {gex.vanna_bias:.2%} (+ = calls dominate)")

            print(f"\nKey Levels:")
            if gex.zero_gamma_level:
                print(f"  Zero Gamma: ${gex.zero_gamma_level:.2f} ({gex.gamma_flip_distance:.2%} away)")
            else:
                print(f"  Zero Gamma: Not found")
            print(f"  Max Gamma Strike: ${gex.max_gamma_strike:.2f}")
            print(f"  Max Vanna Strike: ${gex.max_vanna_strike:.2f}")

            # Interpretation
            print(f"\nInterpretation:")
            if gex.total_gamma_exposure > 0:
                print(f"  [STABILIZING] Dealers are net long gamma -> will dampen moves")
            else:
                print(f"  [VOLATILE] Dealers are net short gamma -> will amplify moves")

            if gex.vanna_bias > 0.1:
                print(f"  [CALL HEAVY] High vanna bias -> bullish positioning")
            elif gex.vanna_bias < -0.1:
                print(f"  [PUT HEAVY] Negative vanna bias -> defensive positioning")

        else:
            print("[ERROR] Failed to calculate gamma/vanna")
