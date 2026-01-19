"""
Strategy Engine - Wraps existing support trading strategy
Imports logic from strategy/support_trading_strategy.py and test_weighted_greek_levels.py
"""
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.support_trading_strategy import BayesianConfidenceScorer


class StrategyEngine:
    """
    Wrapper around support trading strategy logic
    Provides:
    - Strength score calculation (0-100)
    - Bayesian confidence scoring
    - Entry condition checking
    """

    def __init__(self, model_path: str):
        """
        Initialize strategy engine

        Args:
            model_path: Path to XGBoost model pickle file
        """
        self.model = None
        self.selected_features = None

        # Load XGBoost model if file exists
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.selected_features = model_data.get('selected_features', None)
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Model file not found: {model_path}")
            logger.error("Strategy engine initialized without model - predictions will fail")

        # Initialize Bayesian confidence scorer
        self.bayesian_scorer = BayesianConfidenceScorer()

    def calculate_strength_score(self, feature_dict: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Greek level strength score (0-100)
        Based on test_weighted_greek_levels.py algorithm

        Components:
        - Gamma magnitude: 0-20 points
        - Vanna magnitude: 0-15 points
        - HP proximity: 0-10 points
        - MHP proximity: 0-20 points (weighted 2x)
        - HG proximity: 0-25 points
        - Overlap bonus: 0-10 points

        Args:
            feature_dict: Feature dictionary with institutional features

        Returns:
            (total_score, component_breakdown)
        """
        components = {}
        total = 0.0

        # Gamma score (0-20 points)
        gamma = abs(feature_dict.get('inst_total_gamma', 0))
        gamma_score = min(gamma / 0.01, 1.0) * 20
        components['gamma'] = gamma_score
        total += gamma_score

        # Vanna score (0-15 points)
        vanna = abs(feature_dict.get('inst_total_vanna', 0))
        vanna_score = min(vanna / 0.01, 1.0) * 15
        components['vanna'] = vanna_score
        total += vanna_score

        # HP proximity (0-10 points)
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
        hp_min_dist = min(hp_support_dist, hp_resist_dist)

        if hp_min_dist < 0.005:  # Within 0.5%
            hp_score = (0.005 - hp_min_dist) / 0.005 * 10
        else:
            hp_score = 0.0
        components['hp'] = hp_score
        total += hp_score

        # MHP proximity (0-20 points, weighted 2x)
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))
        mhp_min_dist = min(mhp_support_dist, mhp_resist_dist)

        if mhp_min_dist < 0.005:  # Within 0.5%
            mhp_score = (0.005 - mhp_min_dist) / 0.005 * 20
        else:
            mhp_score = 0.0
        components['mhp'] = mhp_score
        total += mhp_score

        # HG proximity (0-25 points, highest weight)
        hg_below_dist = abs(feature_dict.get('inst_hg_below_dist', 1.0))
        hg_above_dist = abs(feature_dict.get('inst_hg_above_dist', 1.0))
        hg_min_dist = min(hg_below_dist, hg_above_dist)

        if hg_min_dist < 0.005:  # Within 0.5%
            hg_score = (0.005 - hg_min_dist) / 0.005 * 25
        else:
            hg_score = 0.0
        components['hg'] = hg_score
        total += hg_score

        # Overlap bonus (0-10 points)
        # Count how many levels are within 0.2% of each other
        levels_close = 0
        if hp_min_dist < 0.002:
            levels_close += 1
        if mhp_min_dist < 0.002:
            levels_close += 1
        if hg_min_dist < 0.002:
            levels_close += 1

        overlap_score = min(levels_close * 5, 10)  # +5 per level, max 10
        components['overlap'] = overlap_score
        total += overlap_score

        return np.clip(total, 0, 100), components

    def identify_level_type(self, feature_dict: Dict) -> str:
        """
        Identify if at support or resistance level

        Args:
            feature_dict: Feature dictionary with institutional features

        Returns:
            'support', 'resistance', or 'none'
        """
        hp_support_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
        hp_resist_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
        mhp_support_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
        mhp_resist_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))

        # Check if closer to support or resistance
        support_min = min(hp_support_dist, mhp_support_dist)
        resistance_min = min(hp_resist_dist, mhp_resist_dist)

        # Need to be within 0.5% to be considered at a level
        threshold = 0.005

        if support_min < threshold and support_min < resistance_min:
            return 'support'
        elif resistance_min < threshold and resistance_min < support_min:
            return 'resistance'
        else:
            return 'none'

    def identify_level_source(self, feature_dict: Dict) -> str:
        """
        Identify which level is closest (HP, MHP, HG)

        Args:
            feature_dict: Feature dictionary

        Returns:
            Level source string (e.g., 'HP Support', 'MHP Resistance')
        """
        level_type = self.identify_level_type(feature_dict)

        if level_type == 'support':
            hp_dist = abs(feature_dict.get('inst_hp_support_dist', 1.0))
            mhp_dist = abs(feature_dict.get('inst_mhp_support_dist', 1.0))
            hg_dist = abs(feature_dict.get('inst_hg_below_dist', 1.0))

            min_dist = min(hp_dist, mhp_dist, hg_dist)

            if min_dist == hp_dist:
                return 'HP Support'
            elif min_dist == mhp_dist:
                return 'MHP Support'
            else:
                return 'HG Support'

        elif level_type == 'resistance':
            hp_dist = abs(feature_dict.get('inst_hp_resist_dist', 1.0))
            mhp_dist = abs(feature_dict.get('inst_mhp_resist_dist', 1.0))
            hg_dist = abs(feature_dict.get('inst_hg_above_dist', 1.0))

            min_dist = min(hp_dist, mhp_dist, hg_dist)

            if min_dist == hp_dist:
                return 'HP Resistance'
            elif min_dist == mhp_dist:
                return 'MHP Resistance'
            else:
                return 'HG Resistance'

        return 'None'

    def check_entry_conditions(
        self,
        feature_array: np.ndarray,
        feature_dict: Dict,
        min_strength: float,
        max_strength: float,
        min_confidence: float
    ) -> Tuple[bool, float, float, str]:
        """
        Check if all entry conditions are met

        Conditions:
        1. Moderate strength (15-30)
        2. Support level only
        3. High Bayesian confidence (> 0.55)
        4. Model predicts UP

        Args:
            feature_array: Feature array for model
            feature_dict: Feature dictionary for analysis
            min_strength: Minimum strength score
            max_strength: Maximum strength score
            min_confidence: Minimum Bayesian confidence

        Returns:
            (should_enter, confidence, strength_score, level_type)
        """
        # Calculate strength
        strength_score, components = self.calculate_strength_score(feature_dict)

        # Condition 1: Moderate strength
        if not (min_strength <= strength_score <= max_strength):
            return False, 0.0, strength_score, 'none'

        # Condition 2: Support level only
        level_type = self.identify_level_type(feature_dict)
        if level_type != 'support':
            return False, 0.0, strength_score, level_type

        # Check if model is loaded
        if self.model is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Cannot make predictions - model not loaded")
            return False, 0.0, strength_score, level_type

        # Get model prediction
        proba = self.model.predict_proba(feature_array.reshape(1, -1))[0]
        prob_up = proba[1]

        # Calculate Bayesian confidence
        confidence = self.bayesian_scorer.calculate_confidence(
            model_probability=prob_up,
            strength_score=strength_score,
            level_type=level_type
        )

        # Condition 3: High confidence
        if confidence < min_confidence:
            return False, confidence, strength_score, level_type

        # Condition 4: Model predicts UP
        if prob_up <= 0.5:
            return False, confidence, strength_score, level_type

        # All conditions met!
        return True, confidence, strength_score, level_type
