#!/usr/bin/env python
"""Quick test to verify baseline_methods refactoring works correctly."""
from __future__ import annotations

import random
import sys

# Test that all methods are importable and functional
from models.baseline_methods import get_method_instance

def test_all_methods():
    """Test that all 5 methods work with new architecture."""
    methods = ["SVR", "PLSR", "XGB", "RF", "LR"]
    rng = random.Random(42)
    
    print("Testing baseline_methods refactoring...")
    print("=" * 60)
    
    for method_name in methods:
        try:
            # Get method instance
            method = get_method_instance(method_name)
            print(f"\n✓ {method_name:6s} - imported successfully")
            
            # Generate config
            config = method.get_random_config(rng)
            assert isinstance(config, dict), f"Config should be dict, got {type(config)}"
            assert "use_sg" in config or method_name in ["PLSR", "LR"], "Missing common config fields"
            print(f"        - config generated ({len(config)} keys)")
            
            # Instantiate model
            model = method.instantiate_model(config)
            assert hasattr(model, "fit"), f"{method_name} model should have fit() method"
            assert hasattr(model, "predict"), f"{method_name} model should have predict() method"
            print(f"        - model instantiated: {type(model).__name__}")
            
        except Exception as e:
            print(f"\n✗ {method_name:6s} - FAILED: {e}", file=sys.stderr)
            return False
    
    print("\n" + "=" * 60)
    print("✓ All methods passed refactoring test!")
    return True


if __name__ == "__main__":
    success = test_all_methods()
    sys.exit(0 if success else 1)
