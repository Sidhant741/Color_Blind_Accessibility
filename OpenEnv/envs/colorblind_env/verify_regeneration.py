import sys
import json
import os
import time

sys.path.append("/Users/vikram/Desktop/color_blind/OpenEnv/envs/colorblind_env")
from app import run_inference

def test_regeneration():
    print("Testing Regeneration Logic...")
    
    # 1. First run (New Plot)
    print("Step 1: Running New Plot...")
    o1, f1, c1 = run_inference(preset_categories=None, colorblind_types=None)
    
    o1_hex = {k: v.hex for k, v in o1.items()}
    print(f"Original Colors: {o1_hex}")
    print(f"CB Types: {c1}")
    
    # 2. Second run (Regenerate)
    print("\nStep 2: Running Regeneration...")
    o2, f2, c2 = run_inference(preset_categories=o1, colorblind_types=c1)
    
    o2_hex = {k: v.hex for k, v in o2.items()}
    print(f"Original Colors: {o2_hex}")
    print(f"CB Types: {c2}")
    
    # 3. Assertions
    match_colors = (o1_hex == o2_hex)
    match_cb = (c1 == c2)
    different_fix = (f1 != f2)
    
    print("\nResults:")
    print(f"Match Original Colors? {match_colors}")
    print(f"Match CB Types? {match_cb}")
    print(f"Different RL Fix? {different_fix}")
    
    if match_colors and match_cb:
        print("\n✅ SUCCESS: Regeneration preserved the original layout!")
    else:
        print("\n❌ FAILURE: Original layout changed!")
        if not match_colors:
            print(f"Diff Colors: {o1_hex} vs {o2_hex}")
        if not match_cb:
            print(f"Diff CB: {c1} vs {c2}")

if __name__ == "__main__":
    test_regeneration()
