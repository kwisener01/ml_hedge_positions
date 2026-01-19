"""
Feature Ablation Study
Tests different feature combinations to find optimal set for each symbol
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_ensemble import train_pipeline
from datetime import datetime


def run_feature_ablation(symbol: str, trials_per_config: int = 3):
    """
    Test different feature configurations

    Configurations tested:
    1. Base only (window + classic): 22 features
    2. Base + Institutional (HP/MHP/HG): 32 features
    3. Base + Institutional + Gamma/Vanna: 37 features
    4. Base + Institutional + Gamma/Vanna + LOB: 47 features
    5. Base + Institutional + LOB (no Gamma/Vanna): 42 features
    """

    print("\n" + "="*80)
    print(f"FEATURE ABLATION STUDY: {symbol}")
    print("="*80)
    print(f"Trials per configuration: {trials_per_config}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    configurations = [
        {
            "name": "Base Only (V1-V22)",
            "features": 22,
            "description": "Window + Classic features only",
            "code": "base_only"
        },
        {
            "name": "Base + Institutional",
            "features": 32,
            "description": "Add HP/MHP/HG (no Gamma/Vanna/LOB)",
            "code": "base_inst"
        },
        {
            "name": "Base + Inst + Gamma/Vanna",
            "features": 37,
            "description": "Add Gamma/Vanna exposure (no LOB)",
            "code": "base_inst_gamma"
        },
        {
            "name": "Base + Inst + Gamma + LOB",
            "features": 47,
            "description": "All features (current SPY config)",
            "code": "full"
        },
        {
            "name": "Base + Inst + LOB (no Gamma)",
            "features": 42,
            "description": "LOB without Gamma/Vanna",
            "code": "base_inst_lob"
        }
    ]

    results = {}

    for config in configurations:
        print(f"\n{'-'*80}")
        print(f"Testing: {config['name']} ({config['features']} features)")
        print(f"Description: {config['description']}")
        print(f"{'-'*80}")

        accuracies = []

        for trial in range(trials_per_config):
            print(f"\nTrial {trial + 1}/{trials_per_config}...")

            # This would require modifying train_ensemble to accept feature config
            # For now, we'll output the structure
            # accuracy = train_with_config(symbol, config['code'])
            # accuracies.append(accuracy)

            # Placeholder for now
            print(f"  [Would train {symbol} with {config['features']} features]")

        # results[config['name']] = {
        #     'mean': np.mean(accuracies),
        #     'std': np.std(accuracies),
        #     'trials': accuracies
        # }

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"\nSymbol: {symbol}")
    print(f"Trials per config: {trials_per_config}\n")
    print(f"{'Configuration':<30} {'Features':<10} {'Mean Acc':<12} {'Std Dev':<10}")
    print("-"*80)

    # for config_name, result in results.items():
    #     print(f"{config_name:<30} {config['features']:<10} "
    #           f"{result['mean']:.2%}    {result['std']:.2%}")

    print("\n" + "="*80)


if __name__ == "__main__":
    """
    Run feature ablation for both symbols

    This script structure shows what we WOULD test.
    To implement, we need to:
    1. Modify FeatureMatrixBuilder to support feature masking
    2. Add configuration flags to train_ensemble.py
    3. Run multiple trials and average results
    """

    print("""
    ================================================================================
    FEATURE ABLATION STUDY - IMPLEMENTATION PLAN
    ================================================================================

    This script outlines the feature combinations to test:

    For SPY (currently at 54.38% with 47 features):
    ============================================
    1. Base only (22): Baseline performance
    2. Base + Inst (32): Does institutional help without gamma?
    3. Base + Inst + Gamma (37): Gamma value without LOB
    4. Base + Inst + Gamma + LOB (47): Current best
    5. Base + Inst + LOB (42): LOB value without gamma

    For QQQ (currently at 51.20% with 37 features, declining to 49% with 47):
    ============================================================================
    1. Base only (22): Baseline performance
    2. Base + Inst (32): Leverages 14.4% HP/MHP interaction
    3. Base + Inst + Gamma (37): Current config (51.20%)
    4. Base + Inst + Gamma + LOB (47): Tested, got 49% (overfitting)
    5. Base + Inst + LOB (42): Skip gamma, test LOB directly

    Expected Results:
    ================
    SPY:
    - Config 4 (47 features) should win: LOB helps SPY
    - Config 3 (37 features) second best: Gamma helps
    - Config 2 (32 features) baseline: Institutional helps some

    QQQ:
    - Config 2 (32 features) might win: Just HP/MHP/HG
    - Config 3 (37 features) current: 51.20%
    - Config 4 (47 features) overfits: 49%

    Alternative Approach:
    ====================
    Instead of full ablation study, we could:

    1. QUICK TEST: Train 3 specific configs for QQQ
       - 32 features (no gamma/vanna)
       - 37 features (with gamma/vanna, current)
       - 42 features (with LOB, no gamma)

    2. Pick winner, move to next improvement strategy

    Would you like to:
    A) Implement full ablation study (requires code changes)
    B) Quick test specific QQQ configs
    C) Move to extended synthetic data (10→100 quotes/day)
    """)
