#!/usr/bin/env python3
"""
Test runner for all quantumbench tests.
Runs all test files and reports overall status.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def main():
    """Run all tests and report results."""
    test_dir = Path(__file__).parent
    test_files = sorted([f for f in test_dir.glob("test_*.py") if f.name != "run_all_tests.py"])
    
    print("🧪 QUANTUMBENCH TEST SUITE")
    print(f"Found {len(test_files)} test files")
    
    results = {}
    passed = 0
    
    for test_file in test_files:
        success = run_test_file(test_file)
        results[test_file.name] = success
        if success:
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{len(test_files)} tests passed")
    
    if passed == len(test_files):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())