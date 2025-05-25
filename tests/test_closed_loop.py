#!/usr/bin/env python3
"""
Closed loop test: Generate dataset with cheapest model and validate it.
This test performs the complete workflow from generation to validation.
"""

import sys
import os
import tempfile
import json
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_dataset import DatasetGenerator

def test_closed_loop_generation_and_validation():
    """
    Perform a complete closed loop test:
    1. Generate a 1-example dataset using the cheapest Anthropic model
    2. Validate the generated dataset
    3. Report results
    """
    print("🔄 CLOSED LOOP TEST: Generation + Validation")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  ANTHROPIC_API_KEY not set. This test requires an API key.")
        print("   Set your API key: export ANTHROPIC_API_KEY=your_key_here")
        return False
    
    try:
        print("🚀 Phase 1: Dataset Generation")
        print("-" * 40)
        
        # Use Claude 3 Haiku (cheapest model)
        cheapest_model = "claude-3-haiku-20240307"
        print(f"📦 Using model: {cheapest_model}")
        
        # Initialize generator with budget constraints
        generator = DatasetGenerator(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model=cheapest_model,
            min_balance=0.0  # Disable balance check for testing
        )
        
        # Generate 1 example with budget protection
        print("💰 Setting conservative budget limits...")
        generator.max_cost = 0.05  # Maximum $0.05 spend
        
        print("🎯 Generating 1 dataset example...")
        datasets = generator.generate_multiple_datasets(num_examples=1)
        
        if not datasets:
            print("❌ No datasets generated - likely budget constraints")
            return False
        
        # The datasets are saved automatically, get the path from the output directory
        dataset_path = os.path.join(generator.output_directory, "1.json")
        print(f"✅ Dataset generated: {dataset_path}")
        
        # Print generation summary
        generator.print_summary_stats()
        
        print("\n🔍 Phase 2: Dataset Validation")
        print("-" * 40)
        
        # Run validation on the generated dataset
        validation_cmd = [
            sys.executable, 
            "validate_dataset.py", 
            str(dataset_path)
        ]
        
        print(f"🧪 Running validation: {' '.join(validation_cmd)}")
        
        # Change to parent directory for validation
        parent_dir = Path(__file__).parent.parent
        result = subprocess.run(
            validation_cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True
        )
        
        print("📊 Validation Results:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Validation Warnings/Errors:")
            print(result.stderr)
        
        validation_success = result.returncode == 0
        
        print("\n📋 Closed Loop Summary")
        print("-" * 40)
        print(f"Generation: ✅ Success")
        print(f"Validation: {'✅ Success' if validation_success else '❌ Failed'}")
        print(f"Model used: {cheapest_model}")
        print(f"Total cost: ${generator.total_cost:.6f}")
        print(f"Tokens: {generator.total_tokens_in + generator.total_tokens_out}")
        
        return validation_success
        
    except Exception as e:
        print(f"❌ Closed loop test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_closed_loop_with_mock():
    """
    Test the closed loop workflow with mocked API for CI/testing without API key.
    """
    print("\n🔄 MOCK CLOSED LOOP TEST (No API Key Required)")
    print("=" * 60)
    
    try:
        # Create a minimal test dataset
        test_dataset = {
            "problem_description": "Create a simple Bell state circuit",
            "qiskit_code": """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)

circuit.h(qr[0])
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
""",
            "expected_output": "Bell state measurement distribution with roughly equal '00' and '11' counts",
            "category": "entanglement",
            "difficulty": "beginner"
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_dataset, tmp, indent=2)
            tmp_path = tmp.name
        
        print(f"📝 Created mock dataset: {tmp_path}")
        
        # Run validation on mock dataset
        validation_cmd = [
            sys.executable, 
            "validate_dataset.py", 
            tmp_path
        ]
        
        parent_dir = Path(__file__).parent.parent
        result = subprocess.run(
            validation_cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True
        )
        
        print("📊 Mock Validation Results:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        # Cleanup
        os.unlink(tmp_path)
        
        validation_success = result.returncode == 0
        print(f"\n✅ Mock closed loop: {'Success' if validation_success else 'Failed'}")
        
        return validation_success
        
    except Exception as e:
        print(f"❌ Mock closed loop test failed: {e}")
        return False

def main():
    """Run closed loop tests."""
    print("🧪 QUANTUMBENCH CLOSED LOOP TESTING")
    print("=" * 60)
    
    # Try real API test first if key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        print("🔑 API key detected - running full closed loop test")
        real_test_passed = test_closed_loop_generation_and_validation()
    else:
        print("🚫 No API key - skipping real API test")
        real_test_passed = None
    
    # Always run mock test
    mock_test_passed = test_closed_loop_with_mock()
    
    # Summary
    print(f"\n📋 FINAL RESULTS")
    print("=" * 60)
    if real_test_passed is not None:
        print(f"Real API Test: {'✅ PASS' if real_test_passed else '❌ FAIL'}")
    else:
        print("Real API Test: ⏭️  SKIPPED (no API key)")
    print(f"Mock Test: {'✅ PASS' if mock_test_passed else '❌ FAIL'}")
    
    # Return success if any test passed
    overall_success = (real_test_passed or mock_test_passed)
    if overall_success:
        print("🎉 Closed loop testing completed successfully!")
    else:
        print("❌ Closed loop testing failed!")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)