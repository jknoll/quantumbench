#!/usr/bin/env python3
"""
Quick script to test JSON parsing with the improved error handling
"""
import os
import json
from generate_dataset import DatasetGenerator

def test_json_parsing():
    """Test a minimal API call to debug JSON parsing"""
    print("🔍 Testing JSON parsing with minimal setup...")
    
    # Create generator with caching disabled to isolate the issue
    generator = DatasetGenerator(use_caching=False)
    
    # Load prompt
    try:
        static_prompt, dynamic_prompt = generator.load_prompt(1)
        print(f"✓ Prompt loaded: {len(static_prompt)} static, {len(dynamic_prompt)} dynamic chars")
    except Exception as e:
        print(f"❌ Failed to load prompt: {e}")
        return
    
    # Try a single generation with detailed logging
    print("\n📡 Attempting single API call...")
    result = generator.generate_dataset_entry(static_prompt, dynamic_prompt, max_retries=0)
    
    print(f"\n📊 Result summary:")
    print(f"  Type: {type(result)}")
    print(f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
    if "error" in result:
        print(f"  Error: {result['error']}")
        if "extracted_json" in result:
            print(f"  Extracted JSON length: {len(result['extracted_json'])}")
        if "original_error" in result:
            print(f"  Original error: {result['original_error']}")
    else:
        print(f"  Success! Generated dataset with ID: {result.get('problem_id', 'N/A')}")

if __name__ == "__main__":
    test_json_parsing()