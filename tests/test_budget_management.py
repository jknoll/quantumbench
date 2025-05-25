#!/usr/bin/env python3
"""
Test script to demonstrate budget management functionality without requiring API key.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_dataset import DatasetGenerator

class MockDatasetGenerator(DatasetGenerator):
    """Mock version of DatasetGenerator for testing budget management."""
    
    def __init__(self, min_balance=0.50, max_cost=None):
        # Initialize without API client for testing
        self.model = "claude-sonnet-4-20250514"
        self.min_balance = min_balance
        self.max_cost = max_cost
        self.start_balance = None
        self.end_balance = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost = 0.0
        self.generation_stopped_by_budget = False
        self.requested_examples = 0
        self.completed_examples = 0
    
    def simulate_generation_cost(self, example_num):
        """Simulate the cost of generating one example."""
        # Simulate varying costs per example
        base_cost = 0.05  # $0.05 per example
        variation = 0.01 * example_num  # Slightly increasing cost
        simulated_cost = base_cost + variation
        
        self.total_cost += simulated_cost
        self.total_tokens_in += 5000  # Simulate token usage
        self.total_tokens_out += 2000
        
        print(f"  Simulated generation cost: ${simulated_cost:.4f}")
        print(f"  Total cost so far: ${self.total_cost:.4f}")
        
        return simulated_cost

def test_budget_scenarios():
    """Test various budget management scenarios."""
    
    print("🧪 TESTING BUDGET MANAGEMENT FUNCTIONALITY")
    print("=" * 60)
    
    # Test 1: No budget constraints
    print("\n📋 TEST 1: No Budget Constraints")
    print("-" * 40)
    generator = MockDatasetGenerator()
    
    for i in range(3):
        can_continue, message = generator.check_budget_constraints()
        print(f"Example {i+1}: {message}")
        if can_continue:
            generator.simulate_generation_cost(i+1)
            generator.completed_examples += 1
        else:
            generator.generation_stopped_by_budget = True
            break
    
    generator.print_summary_stats()
    
    # Test 2: Maximum cost limit
    print("\n📋 TEST 2: Maximum Cost Limit ($0.10)")
    print("-" * 40)
    generator = MockDatasetGenerator(max_cost=0.10)
    generator.requested_examples = 5
    
    for i in range(5):
        can_continue, message = generator.check_budget_constraints()
        print(f"Example {i+1}: {message}")
        if can_continue:
            generator.simulate_generation_cost(i+1)
            generator.completed_examples += 1
        else:
            print(f"🛑 STOPPING: {message}")
            generator.generation_stopped_by_budget = True
            break
    
    generator.print_summary_stats()
    
    # Test 3: Minimum balance threshold
    print("\n📋 TEST 3: Minimum Balance Threshold")
    print("-" * 40)
    generator = MockDatasetGenerator(min_balance=2.0)  # High threshold
    generator.requested_examples = 3
    
    # Simulate already having spent some money
    generator.total_cost = 8.5  # Simulate $8.50 already spent
    
    for i in range(3):
        can_continue, message = generator.check_budget_constraints()
        print(f"Example {i+1}: {message}")
        if can_continue:
            generator.simulate_generation_cost(i+1)
            generator.completed_examples += 1
        else:
            print(f"🛑 STOPPING: {message}")
            generator.generation_stopped_by_budget = True
            break
    
    generator.print_summary_stats()
    
    # Test 4: Both constraints
    print("\n📋 TEST 4: Both Max Cost and Min Balance")
    print("-" * 40)
    generator = MockDatasetGenerator(min_balance=1.0, max_cost=0.15)
    generator.requested_examples = 4
    
    for i in range(4):
        can_continue, message = generator.check_budget_constraints()
        print(f"Example {i+1}: {message}")
        if can_continue:
            generator.simulate_generation_cost(i+1)
            generator.completed_examples += 1
        else:
            print(f"🛑 STOPPING: {message}")
            generator.generation_stopped_by_budget = True
            break
    
    generator.print_summary_stats()

def test_command_line_args():
    """Test command line argument validation."""
    print("\n🧪 TESTING COMMAND LINE VALIDATION")
    print("=" * 60)
    
    test_cases = [
        ("Valid case", ["--max-cost", "0.50", "--min-balance", "0.25"]),
        ("Invalid negative min-balance", ["--min-balance", "-0.50"]),
        ("Invalid zero max-cost", ["--max-cost", "0"]),
        ("Valid small max-cost", ["--max-cost", "0.01"]),
    ]
    
    for description, args in test_cases:
        print(f"\n📋 {description}: {' '.join(args)}")
        # Would test actual argument parsing here in a real test
        print("  (Command line validation implemented in main())")

if __name__ == "__main__":
    test_budget_scenarios()
    test_command_line_args()
    
    print("\n✅ Budget management testing completed!")
    print("\nTo test with real API:")
    print("  export ANTHROPIC_API_KEY=your_key")
    print("  python generate_dataset.py --max-cost 0.10 --num-examples 3")