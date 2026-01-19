"""
Complete Feature Matrix Construction
Combines all 22 variables + institutional features
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import sys
sys.path.append('..')
from data.tradier_client import QuoteData, TradierClient
from data.cleaner import DataCleaner, EventWindowBuilder
from features.window_features import WindowFeatureExtractor, WindowFeatures
from features.classic_features import ClassicFeatureExtractor, ClassicFeatures
from features.resilience_arb import ResilienceCalculator, ArbitrageCalculator
from institutional.hedge_pressure import HedgePressureCalculator, HedgePressureResult
from institutional.monthly_hp import MonthlyHPCalculator, MonthlyHPResult
from institutional.half_gap import HalfGapCalculator, HalfGapResult
from institutional.gamma_exposure import GammaVannaCalculator, GammaExposure
from config.settings import model_config

@dataclass
class InstitutionalFeatures:
    """Institutional layer features"""
    hp_net: float                  # Net hedge pressure (-1 to 1)
    hp_support_distance: float     # Distance to nearest HP support
    hp_resistance_distance: float  # Distance to nearest HP resistance
    mhp_score: float               # Monthly HP composite
    mhp_support_distance: float    # Distance to MHP support
    mhp_resistance_distance: float # Distance to MHP resistance
    hg_above_distance: float       # Distance to nearest HG above
    hg_below_distance: float       # Distance to nearest HG below
    in_gamma_flip_zone: int        # Binary: in dealer flip zone
    confluence_score: float        # Number of levels near price
    total_gamma_exposure: float    # Net dealer gamma (billions)
    gamma_at_spot: float           # Gamma concentration at price
    total_vanna: float             # Net dealer vanna
    vanna_bias: float              # Call/put vanna bias (-1 to 1)
    gamma_flip_distance: float     # Distance to zero gamma level (%)
    # LOB Resilience features (5)
    order_flow_resilience: float   # Speed of order replacement (0-1)
    bid_ask_resilience: float      # How quickly spread tightens after trades
    volume_resilience: float       # Volume consistency (stability metric)
    price_impact_decay: float      # How fast price impact fades (0-1)
    immediacy_ratio: float         # Fill rate / ADV proxy
    # LOB Arbitrage features (5)
    v3_crossing_return: float      # (Last_Bid - First_Ask) / First_Ask
    bid_ask_spread_pct: float      # Current bid-ask spread %
    spread_z_score: float          # Z-score of spread
    effective_spread: float        # Effective spread after trades
    microstructure_edge: float     # Combined arb signal

    def to_array(self, include_lob: bool = True) -> np.ndarray:
        """
        Convert features to numpy array

        Args:
            include_lob: If True, include LOB features (47 total)
                        If False, exclude LOB features (37 total)
        """
        base_features = np.array([
            self.hp_net, self.hp_support_distance, self.hp_resistance_distance,
            self.mhp_score, self.mhp_support_distance, self.mhp_resistance_distance,
            self.hg_above_distance, self.hg_below_distance,
            self.in_gamma_flip_zone, self.confluence_score,
            self.total_gamma_exposure, self.gamma_at_spot, self.total_vanna,
            self.vanna_bias, self.gamma_flip_distance
        ])

        if include_lob:
            lob_features = np.array([
                self.order_flow_resilience, self.bid_ask_resilience, self.volume_resilience,
                self.price_impact_decay, self.immediacy_ratio,
                self.v3_crossing_return, self.bid_ask_spread_pct, self.spread_z_score,
                self.effective_spread, self.microstructure_edge
            ])
            return np.concatenate([base_features, lob_features])
        else:
            return base_features

    def to_dict(self) -> Dict:
        return {
            "inst_hp_net": self.hp_net,
            "inst_hp_support_dist": self.hp_support_distance,
            "inst_hp_resist_dist": self.hp_resistance_distance,
            "inst_mhp_score": self.mhp_score,
            "inst_mhp_support_dist": self.mhp_support_distance,
            "inst_mhp_resist_dist": self.mhp_resistance_distance,
            "inst_hg_above_dist": self.hg_above_distance,
            "inst_hg_below_dist": self.hg_below_distance,
            "inst_gamma_flip": self.in_gamma_flip_zone,
            "inst_confluence": self.confluence_score,
            "inst_total_gamma": self.total_gamma_exposure,
            "inst_gamma_at_spot": self.gamma_at_spot,
            "inst_total_vanna": self.total_vanna,
            "inst_vanna_bias": self.vanna_bias,
            "inst_gamma_flip_dist": self.gamma_flip_distance,
            "lob_order_flow_res": self.order_flow_resilience,
            "lob_bid_ask_res": self.bid_ask_resilience,
            "lob_volume_res": self.volume_resilience,
            "lob_price_impact_decay": self.price_impact_decay,
            "lob_immediacy": self.immediacy_ratio,
            "lob_v3_crossing": self.v3_crossing_return,
            "lob_spread_pct": self.bid_ask_spread_pct,
            "lob_spread_z": self.spread_z_score,
            "lob_eff_spread": self.effective_spread,
            "lob_micro_edge": self.microstructure_edge
        }

class FeatureMatrixBuilder:
    """
    Builds complete feature matrix for SVM training

    Symbol-specific feature sets:
    - SPY: 47 features (includes LOB microstructure features)
    - QQQ: 37 features (excludes LOB features due to redundancy)

    Feature breakdown:
    - Window features (V1-V10): 10
    - Classic features (V11-V22): 12
    - Institutional features: 15 (HP, MHP, HG, Gamma, Vanna)
    - LOB Resilience features: 5 (SPY ONLY)
    - LOB Arbitrage features: 5 (SPY ONLY)
    """

    def __init__(self, client: Optional[TradierClient] = None, enable_lob_for_all: bool = False):
        """
        Args:
            client: TradierClient instance
            enable_lob_for_all: If True, enable LOB features for all symbols
                               If False, only enable for SPY (default)
        """
        self.client = client or TradierClient()
        self.cleaner = DataCleaner()
        self.window_extractor = WindowFeatureExtractor()
        self.classic_extractor = ClassicFeatureExtractor()
        self.hp_calc = HedgePressureCalculator()
        self.mhp_calc = MonthlyHPCalculator()
        self.hg_calc = HalfGapCalculator()
        self.gamma_calc = GammaVannaCalculator(self.client)
        self.resilience_calc = ResilienceCalculator()
        self.arb_calc = ArbitrageCalculator(k=5)  # k=5 event window for V3
        self.enable_lob_for_all = enable_lob_for_all

        # Cached institutional data (update less frequently)
        self._hp_cache: Dict[str, HedgePressureResult] = {}
        self._mhp_cache: Dict[str, MonthlyHPResult] = {}
        self._hg_cache: Dict[str, HalfGapResult] = {}
        self._gamma_cache: Dict[str, GammaExposure] = {}
        self._cache_time: Optional[datetime] = None
        
    def build_feature_vector(
        self,
        symbol: str,
        window: List[QuoteData],
        current_quote: QuoteData
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build complete feature vector for a single observation

        Symbol-specific behavior:
        - SPY: 47 features (includes LOB)
        - QQQ: 37 features (excludes LOB)

        Returns:
            Tuple of (feature_array, feature_dict)
        """
        # Determine if LOB features should be included
        include_lob = symbol == "SPY" or self.enable_lob_for_all

        # Extract V1-V10 (window features)
        window_features = self.window_extractor.extract(window)

        # Extract V11-V22 (classic features)
        classic_features = self.classic_extractor.extract(current_quote, window)

        # Get institutional features (only calculate LOB for SPY)
        inst_features = self._get_institutional_features(
            symbol, current_quote.mid_price, window
        )

        # Combine all features
        # For QQQ, inst_features.to_array(include_lob=False) returns only 15 features
        # For SPY, inst_features.to_array(include_lob=True) returns all 25 features
        feature_array = np.concatenate([
            window_features.to_array(),
            classic_features.to_array(),
            inst_features.to_array(include_lob=include_lob)
        ])

        feature_dict = {
            **window_features.to_dict(),
            **classic_features.to_dict(),
            **inst_features.to_dict()
        }

        return feature_array, feature_dict
    
    def _get_institutional_features(
        self,
        symbol: str,
        spot_price: float,
        window: List[QuoteData]
    ) -> InstitutionalFeatures:
        """
        Get institutional layer features including LOB resilience and arbitrage

        Uses caching to avoid excessive API calls for institutional data
        LOB features calculated fresh from quote window
        """
        # Update cache every 5 minutes
        if self._should_refresh_cache():
            self._refresh_institutional_cache(symbol, spot_price)

        hp = self._hp_cache.get(symbol)
        mhp = self._mhp_cache.get(symbol)
        hg = self._hg_cache.get(symbol)

        # Calculate distance features
        hp_support_dist = self._calc_distance(
            spot_price, hp.key_support if hp else None
        )
        hp_resist_dist = self._calc_distance(
            spot_price, hp.key_resistance if hp else None, above=True
        )

        mhp_support_dist = self._calc_distance(
            spot_price, mhp.primary_support if mhp else None
        )
        mhp_resist_dist = self._calc_distance(
            spot_price, mhp.primary_resistance if mhp else None, above=True
        )

        hg_above_dist = self._calc_distance(
            spot_price, hg.nearest_hg_above if hg else None, above=True
        )
        hg_below_dist = self._calc_distance(
            spot_price, hg.nearest_hg_below if hg else None
        )

        # Check gamma flip zone
        in_flip_zone = 0
        if mhp and mhp.gamma_flip_zone:
            low, high = mhp.gamma_flip_zone
            if low <= spot_price <= high:
                in_flip_zone = 1

        # Calculate confluence score
        confluence = self._calculate_confluence(spot_price, hp, mhp, hg)

        # Get gamma/vanna exposure
        gex = self._gamma_cache.get(symbol)

        # Calculate LOB features (SPY only, or if explicitly enabled for all symbols)
        use_lob = symbol == "SPY" or self.enable_lob_for_all

        if use_lob:
            # Calculate LOB resilience metrics
            try:
                resilience = self.resilience_calc.calculate(window)
            except Exception as e:
                print(f"Error calculating resilience: {e}")
                # Default resilience values
                resilience = type('obj', (object,), {
                    'order_flow_resilience': 0.5,
                    'bid_ask_resilience': 0.5,
                    'volume_resilience': 0.5,
                    'price_impact_decay': 0.5,
                    'immediacy_ratio': 0.5
                })()

            # Calculate LOB arbitrage indicators
            try:
                arb = self.arb_calc.calculate(window)
            except Exception as e:
                print(f"Error calculating arbitrage: {e}")
                # Default arb values
                arb = type('obj', (object,), {
                    'v3_crossing_return': 0.0,
                    'bid_ask_spread_pct': 0.0,
                    'spread_z_score': 0.0,
                    'effective_spread': 0.0,
                    'microstructure_edge': 0.0
                })()
        else:
            # QQQ and others: Don't calculate, just use zeros
            # These won't be included in feature array anyway
            resilience = type('obj', (object,), {
                'order_flow_resilience': 0.0,
                'bid_ask_resilience': 0.0,
                'volume_resilience': 0.0,
                'price_impact_decay': 0.0,
                'immediacy_ratio': 0.0
            })()
            arb = type('obj', (object,), {
                'v3_crossing_return': 0.0,
                'bid_ask_spread_pct': 0.0,
                'spread_z_score': 0.0,
                'effective_spread': 0.0,
                'microstructure_edge': 0.0
            })()

        return InstitutionalFeatures(
            hp_net=hp.net_hp if hp else 0,
            hp_support_distance=hp_support_dist,
            hp_resistance_distance=hp_resist_dist,
            mhp_score=mhp.mhp_score if mhp else 0,
            mhp_support_distance=mhp_support_dist,
            mhp_resistance_distance=mhp_resist_dist,
            hg_above_distance=hg_above_dist,
            hg_below_distance=hg_below_dist,
            in_gamma_flip_zone=in_flip_zone,
            confluence_score=confluence,
            total_gamma_exposure=gex.total_gamma_exposure if gex else 0,
            gamma_at_spot=gex.gamma_at_spot if gex else 0,
            total_vanna=gex.total_vanna if gex else 0,
            vanna_bias=gex.vanna_bias if gex else 0,
            gamma_flip_distance=gex.gamma_flip_distance if gex else 0,
            # LOB Resilience features
            order_flow_resilience=resilience.order_flow_resilience,
            bid_ask_resilience=resilience.bid_ask_resilience,
            volume_resilience=resilience.volume_resilience,
            price_impact_decay=resilience.price_impact_decay,
            immediacy_ratio=resilience.immediacy_ratio,
            # LOB Arbitrage features
            v3_crossing_return=arb.v3_crossing_return,
            bid_ask_spread_pct=arb.bid_ask_spread_pct,
            spread_z_score=arb.spread_z_score,
            effective_spread=arb.effective_spread,
            microstructure_edge=arb.microstructure_edge
        )
    
    def _should_refresh_cache(self) -> bool:
        """Check if institutional cache needs refresh"""
        if self._cache_time is None:
            return True
        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed > 300  # 5 minute cache
    
    def _refresh_institutional_cache(self, symbol: str, spot_price: float):
        """Refresh all institutional calculations"""
        print(f"Refreshing institutional cache for {symbol}...")
        
        try:
            # Get options chains
            chains = self.client.get_multiple_chains(symbol, expiration_count=4)
            
            # Get price history for HG
            history = self.client.get_history(symbol, interval="daily")
            
            # Calculate HP (nearest expiration)
            if chains:
                nearest_exp = sorted(chains.keys())[0]
                chain = chains[nearest_exp]
                self._hp_cache[symbol] = self.hp_calc.calculate(
                    chain, spot_price, symbol
                )
            
            # Calculate MHP (all expirations)
            if chains:
                self._mhp_cache[symbol] = self.mhp_calc.calculate(
                    chains, spot_price, symbol
                )
            
            # Calculate HG
            if not history.empty:
                self._hg_cache[symbol] = self.hg_calc.calculate(
                    history, spot_price, symbol
                )

            # Calculate Gamma/Vanna exposure
            gex = self.gamma_calc.calculate(symbol)
            if gex:
                self._gamma_cache[symbol] = gex

            self._cache_time = datetime.now()
            
        except Exception as e:
            print(f"Error refreshing institutional cache: {e}")
    
    def _calc_distance(
        self,
        spot: float,
        level: Optional[float],
        above: bool = False
    ) -> float:
        """Calculate percentage distance to a level"""
        if level is None:
            return 1.0  # Max distance if no level
        
        distance = (level - spot) / spot
        
        if above:
            return max(distance, 0)
        else:
            return max(-distance, 0)
    
    def _calculate_confluence(
        self,
        spot: float,
        hp: Optional[HedgePressureResult],
        mhp: Optional[MonthlyHPResult],
        hg: Optional[HalfGapResult]
    ) -> float:
        """
        Calculate confluence score - how many institutional levels
        are within 1% of current price
        """
        confluence = 0
        threshold = spot * 0.01  # 1% range
        
        # Check HP levels
        if hp and hp.key_support and abs(hp.key_support - spot) < threshold:
            confluence += 1
        if hp and hp.key_resistance and abs(hp.key_resistance - spot) < threshold:
            confluence += 1
        
        # Check MHP levels
        if mhp and mhp.primary_support and abs(mhp.primary_support - spot) < threshold:
            confluence += 1
        if mhp and mhp.primary_resistance and abs(mhp.primary_resistance - spot) < threshold:
            confluence += 1
        
        # Check HG levels
        if hg:
            for hg_level in hg.active_half_gaps:
                if abs(hg_level - spot) < threshold:
                    confluence += 1
                    break  # Only count once
        
        return confluence / 5.0  # Normalize to 0-1


class TargetCalculator:
    """
    Calculates prediction targets per specification
    
    Target: Direction of average mid-price over next k events
    Classes: 'up', 'down', 'stationary'
    """
    
    def __init__(self, alpha: float = None):
        self.alpha = alpha or model_config.alpha_threshold
    
    def calculate_target(
        self,
        current_mid: float,
        future_window: List[QuoteData]
    ) -> str:
        """
        Calculate target class for a single observation
        
        Args:
            current_mid: Current mid-price
            future_window: Next k quotes
        """
        if not future_window:
            return "stationary"
        
        # Average mid-price of future window
        future_mids = [q.mid_price for q in future_window]
        future_avg = np.mean(future_mids)
        
        # Calculate percentage change
        pct_change = (future_avg - current_mid) / current_mid
        
        if pct_change > self.alpha:
            return "up"
        elif pct_change < -self.alpha:
            return "down"
        else:
            return "stationary"
    
    def encode_target(self, target: str) -> int:
        """Encode string target to integer"""
        mapping = {"down": 0, "stationary": 1, "up": 2}
        return mapping.get(target, 1)
    
    def decode_target(self, encoded: int) -> str:
        """Decode integer to string target"""
        mapping = {0: "down", 1: "stationary", 2: "up"}
        return mapping.get(encoded, "stationary")