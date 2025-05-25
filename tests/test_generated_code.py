#!/usr/bin/env python3
"""
Test execution of the generated Qiskit code from the dataset
"""
import json
import sys

def test_generated_code():
    """Load and execute the generated dataset code"""
    
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
    if 'solutions' not in dataset or len(dataset['solutions']) == 0:
        print("❌ No solutions found in dataset")
        return False
    
    primary_solution = dataset['solutions'][0]  # preference_rank 1
    code = primary_solution['code']
    
    print(f"🧪 Testing primary solution code...")
    print(f"📄 Code length: {len(code)} characters")
    
    # Execute the code with proper namespace
    try:
        print("🔄 Executing generated code...")
        
        # Create a proper execution environment
        exec_globals = {}
        exec_locals = {}
        
        # Execute with both globals and locals
        exec(code, exec_globals, exec_locals)
        print("✅ Code executed successfully!")
        
        # Check if the expected outputs were produced
        if 'optimal_circuit' in exec_locals:
            print(f"✅ Circuit created successfully")
        if 'iterations' in exec_locals and 'probs' in exec_locals:
            print(f"✅ Analysis completed with {len(exec_locals['probs'])} probability measurements")
            max_prob = max(exec_locals['probs'])
            print(f"✅ Maximum success probability: {max_prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try to identify the specific issue
        if "not defined" in str(e):
            print("   Issue: Import or variable definition problem")
        elif "No module named" in str(e):
            print("   Issue: Missing module dependency")
        
        # Show problematic code section
        lines = code.split('\n')
        print(f"\n📝 Code structure ({len(lines)} lines):")
        for i, line in enumerate(lines[:15], 1):  # Show first 15 lines
            print(f"  {i:2d}: {line}")
        if len(lines) > 15:
            print(f"  ... ({len(lines) - 15} more lines)")
        
        return False

def test_common_mistakes():
    """Test the common mistakes code to verify they demonstrate the intended errors"""
    
    dataset_file = "/Users/justinknoll/git/quantum-computing-fine-tune/datasets/claude_sonnet_4_20250514_20250524_105639/1.json"
    
    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    if 'common_mistakes' not in dataset:
        print("❌ No common mistakes found in dataset")
        return False
    
    print(f"\n🧪 Testing {len(dataset['common_mistakes'])} common mistakes...")
    
    for i, mistake in enumerate(dataset['common_mistakes'], 1):
        print(f"\n--- Common Mistake {i}: {mistake['error_type']} ---")
        code = mistake['incorrect_code']
        
        try:
            print(f"🔄 Executing mistake code...")
            exec(code)
            print(f"✅ Mistake code executed (demonstrates: {mistake['explanation'][:60]}...)")
        except Exception as e:
            print(f"⚠️  Mistake code failed to execute: {e}")
            print(f"   This might be expected for some demonstration purposes")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Generated Qiskit Code Execution")
    print("=" * 50)
    
    success = test_generated_code()
    
    if success:
        print("\n" + "=" * 50)
        test_common_mistakes()
    
    print(f"\n🏁 Test completed: {'SUCCESS' if success else 'FAILED'}")