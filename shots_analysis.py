#!/usr/bin/env python3
"""
Statistical Analysis for Optimal Shot Count in Quantum Circuit Validation

This script analyzes the statistical requirements for distinguishing between
correct and incorrect quantum algorithms based on measurement outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List
import json

def binomial_confidence_interval(n_success: int, n_trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate binomial confidence interval for success probability."""
    if n_trials == 0:
        return 0.0, 0.0
    
    p_hat = n_success / n_trials
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval (more accurate for extreme probabilities)
    denominator = 1 + z_score**2 / n_trials
    center = (p_hat + z_score**2 / (2 * n_trials)) / denominator
    half_width = z_score * np.sqrt(p_hat * (1 - p_hat) / n_trials + z_score**2 / (4 * n_trials**2)) / denominator
    
    lower = max(0, center - half_width)
    upper = min(1, center + half_width)
    
    return lower, upper

def calculate_statistical_power(n_shots: int, p_correct: float = 0.95, p_random: float = 0.25, 
                              alpha: float = 0.05) -> float:
    """
    Calculate statistical power to distinguish correct algorithm from random results.
    
    Args:
        n_shots: Number of measurement shots
        p_correct: Expected probability for correct algorithm (e.g., 0.95 for |11⟩ in Grover's)
        p_random: Expected probability for random/broken algorithm (e.g., 0.25 for uniform)
        alpha: Significance level (Type I error rate)
    
    Returns:
        Statistical power (1 - Type II error rate)
    """
    # Under null hypothesis (random), threshold for rejection
    threshold = stats.binom.ppf(1 - alpha, n_shots, p_random)
    
    # Under alternative hypothesis (correct algorithm), probability of exceeding threshold
    power = 1 - stats.binom.cdf(threshold, n_shots, p_correct)
    
    return power

def analyze_shot_requirements():
    """Analyze shot requirements for different confidence levels and effect sizes."""
    
    print("STATISTICAL ANALYSIS: Optimal Shot Count for Quantum Circuit Validation")
    print("=" * 80)
    
    # Define scenarios
    scenarios = [
        {"name": "Grover's |11⟩ (Perfect vs Random)", "p_correct": 0.95, "p_random": 0.25},
        {"name": "Grover's |11⟩ (Good vs Random)", "p_correct": 0.80, "p_random": 0.25},
        {"name": "General Marked State (Perfect vs Random)", "p_correct": 0.90, "p_random": 0.25},
        {"name": "Noisy Algorithm Detection", "p_correct": 0.70, "p_random": 0.25},
    ]
    
    shot_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    print("\nSTATISTICAL POWER ANALYSIS")
    print("-" * 50)
    print(f"{'Scenario':<35} {'Shots':<8} {'Power':<8} {'95% CI Width':<12}")
    print("-" * 50)
    
    recommendations = {}
    
    for scenario in scenarios:
        name = scenario["name"]
        p_correct = scenario["p_correct"]
        p_random = scenario["p_random"]
        
        print(f"\n{name}:")
        
        for shots in shot_counts:
            power = calculate_statistical_power(shots, p_correct, p_random)
            
            # Calculate confidence interval width for the correct algorithm
            expected_successes = int(shots * p_correct)
            lower, upper = binomial_confidence_interval(expected_successes, shots)
            ci_width = upper - lower
            
            print(f"{'':35} {shots:<8} {power:<8.3f} {ci_width:<12.3f}")
            
            # Store recommendation when power >= 0.90 (90% power)
            if name not in recommendations and power >= 0.90:
                recommendations[name] = shots
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    for scenario, min_shots in recommendations.items():
        print(f"{scenario:<35}: {min_shots:>6} shots (90% power)")
    
    # Analyze current validation results
    print("\n" + "=" * 80)
    print("CURRENT VALIDATION ANALYSIS")
    print("=" * 80)
    
    # Load the latest validation results
    try:
        import glob
        json_files = glob.glob("dataset_validations/dataset_validation_*.json")
        if json_files:
            latest_file = max(json_files)
            with open(latest_file, 'r') as f:
                validation_data = json.load(f)
            
            print(f"Analyzing results from: {latest_file}")
            
            solutions = validation_data.get("detailed_results", {}).get("solutions", [])
            
            for i, solution in enumerate(solutions, 1):
                actual_output = solution.get("actual_output", "")
                rank = solution.get("preference_rank", "Unknown")
                
                print(f"\nSolution {i} (Rank {rank}):")
                print(f"  Output: {actual_output}")
                
                # Parse the counts if possible
                if "11" in actual_output and ":" in actual_output:
                    try:
                        # Extract counts from string like "{'11': 1000}" or "{'01': 274, '11': 230, ...}"
                        import re
                        count_11_match = re.search(r"'11':\s*(\d+)", actual_output)
                        total_match = re.findall(r":\s*(\d+)", actual_output)
                        
                        if count_11_match and total_match:
                            count_11 = int(count_11_match.group(1))
                            total_shots = sum(int(match) for match in total_match)
                            
                            prob_11 = count_11 / total_shots
                            lower, upper = binomial_confidence_interval(count_11, total_shots)
                            
                            print(f"  |11⟩ probability: {prob_11:.3f}")
                            print(f"  95% CI: [{lower:.3f}, {upper:.3f}]")
                            print(f"  CI width: {upper - lower:.3f}")
                            
                            # Determine if this looks like correct Grover's algorithm
                            if prob_11 > 0.8:
                                verdict = "✓ Likely correct Grover's algorithm"
                            elif 0.4 > prob_11 > 0.1:
                                verdict = "? Possibly incorrect/noisy algorithm"  
                            else:
                                verdict = "✗ Likely random/broken algorithm"
                            
                            print(f"  Assessment: {verdict}")
                            
                            # Check if current shot count is sufficient
                            if upper - lower < 0.1:  # CI width < 10%
                                print(f"  Shot count: ✓ Sufficient precision ({total_shots} shots)")
                            else:
                                needed_shots = int((1.96**2 * prob_11 * (1 - prob_11)) / (0.05**2))  # For 5% margin of error
                                print(f"  Shot count: ⚠ Could use more precision (current: {total_shots}, suggested: {needed_shots})")
                    
                    except Exception as e:
                        print(f"  Could not parse output: {e}")
                else:
                    print(f"  Could not analyze output format")
    
    except Exception as e:
        print(f"Could not load validation results: {e}")
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print("For reliable quantum algorithm validation:")
    print("• Minimum 500 shots for 90% statistical power")
    print("• Recommended 1000 shots for good precision (±3% margin)")
    print("• Use 2000+ shots for high-stakes validation")
    print("• Consider multiple runs for critical algorithms")
    print("\nConfidence intervals help distinguish:")
    print("• Perfect algorithms: P(target_state) > 0.8")
    print("• Good algorithms: P(target_state) > 0.6") 
    print("• Random/broken: P(target_state) ≈ 0.25 (for 4-state system)")

if __name__ == "__main__":
    analyze_shot_requirements()