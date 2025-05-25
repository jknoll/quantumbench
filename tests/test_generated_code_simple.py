#!/usr/bin/env python3
"""
Simple test of the generated code by writing it to a file and importing
"""
import json
import tempfile
import sys
import os

def test_generated_code():
    """Load and test the generated dataset code"""
    
    # Load the generated dataset
    dataset_file = "/Users/justinknoll/git/quantum-computing-fine-tune/datasets/claude_sonnet_4_20250514_20250524_105639/1.json"
    
    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        print(f"✅ Loaded dataset: {dataset['problem_id']}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Extract the primary solution code
    primary_solution = dataset['solutions'][0]  # preference_rank 1
    code = primary_solution['code']
    
    print(f"🧪 Testing primary solution code...")
    print(f"📄 Code length: {len(code)} characters")
    
    # Write code to temporary file and execute it
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        print(f"🔄 Executing code from temporary file...")
        
        # Execute the file
        import subprocess
        result = subprocess.run([sys.executable, temp_file], 
                              capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(temp_file)
        
        if result.returncode == 0:
            print("✅ Code executed successfully!")
            print("📊 Output:")
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
            return True
        else:
            print(f"❌ Code execution failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"    {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Code execution timed out (>30 seconds)")
        return False
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        return False

def test_alternative_solution():
    """Test the second solution as well"""
    
    dataset_file = "/Users/justinknoll/git/quantum-computing-fine-tune/datasets/claude_sonnet_4_20250514_20250524_105639/1.json"
    
    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    if len(dataset['solutions']) < 2:
        print("⚠️  Only one solution available")
        return True
    
    # Test second solution
    alt_solution = dataset['solutions'][1]  # preference_rank 2
    code = alt_solution['code']
    
    print(f"\n🧪 Testing alternative solution...")
    print(f"📄 Code length: {len(code)} characters")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        print(f"🔄 Executing alternative solution...")
        
        import subprocess
        result = subprocess.run([sys.executable, temp_file], 
                              capture_output=True, text=True, timeout=30)
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            print("✅ Alternative solution executed successfully!")
            print("📊 Output:")
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
        else:
            print(f"⚠️  Alternative solution failed (expected: {alt_solution.get('why_less_preferred', 'issues noted')})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:5]:  # First 5 lines only
                    print(f"    {line}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Alternative solution test failed: {e}")
        return True  # Not critical

if __name__ == "__main__":
    print("🧪 Testing Generated Qiskit Code Execution")
    print("=" * 50)
    
    success = test_generated_code()
    
    if success:
        test_alternative_solution()
    
    print(f"\n🏁 Primary solution test: {'SUCCESS' if success else 'FAILED'}")