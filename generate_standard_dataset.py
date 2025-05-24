#!/usr/bin/env python3
"""
Generate test datasets for standard vs extended thinking comparison
"""

import json
from datetime import datetime
from generate_dataset import DatasetGenerator

class MockDatasetGenerator(DatasetGenerator):
    """Mock generator that creates test datasets without API calls."""
    
    def __init__(self, extended_thinking=False):
        self.model = "claude-sonnet-4-20250514"
        self.extended_thinking = extended_thinking
        self.total_cost = 0.0
        
    def generate_mock_dataset(self):
        """Generate a mock dataset for comparison."""
        
        # Standard reasoning trace
        standard_reasoning = {
            "problem_analysis": "For a 2-qubit system with Grover's algorithm, we need to search for |11⟩ state with optimal iteration count.",
            "quantum_physics_reasoning": "Grover's algorithm uses amplitude amplification through oracle and diffusion operators.",
            "algorithm_choice": "Standard Grover's algorithm provides quadratic speedup for unstructured search.",
            "parameter_selection": "Optimal iterations = floor(π/4 * sqrt(N/M)) ≈ 1 for N=4, M=1",
            "implementation_strategy": "Initialize superposition, apply oracle+diffusion, measure."
        }
        
        # Extended reasoning trace
        extended_reasoning = {
            "problem_analysis": {
                "sub_problem_1": "State space analysis: 2-qubit system has 4 computational basis states |00⟩, |01⟩, |10⟩, |11⟩",
                "sub_problem_2": "Target identification: Mark state |11⟩ which corresponds to index 3 in computational basis",
                "sub_problem_3": "Search space size: N = 2^2 = 4 total states, M = 1 marked state",
                "sub_problem_4": "Probability amplification: Need to rotate amplitude vector in 2D subspace spanned by marked/unmarked states",
                "sub_problem_5": "Optimal iteration calculation: θ = arcsin(√(M/N)) = arcsin(1/2), optimal iterations = ⌊π/(4θ)⌋",
                "sub_problem_6": "Success probability: After optimal iterations, P(success) = sin²((2k+1)θ) where k is iteration count",
                "mathematical_foundation": "Grover operator G = -A_s A_s₀ where A_s is marked state reflection and A_s₀ is uniform state reflection"
            },
            "quantum_physics_reasoning": {
                "mathematical_formalism": "Initial state: |ψ₀⟩ = 1/2(|00⟩ + |01⟩ + |10⟩ + |11⟩), Hilbert space: ℂ⁴",
                "state_vector_evolution": "|ψₖ⟩ = Gᵏ|ψ₀⟩ where G = -(2|ψ₀⟩⟨ψ₀| - I)(2|11⟩⟨11| - I)",
                "operator_analysis": "Oracle O|x⟩ = (-1)^f(x)|x⟩ where f(x) = 1 iff x = 11, Diffusion D = 2|ψ₀⟩⟨ψ₀| - I",
                "amplitude_evolution": "α₁₁(k) = sin((2k+1)arcsin(1/2)), α_others(k) = cos((2k+1)arcsin(1/2))/√3",
                "evolution_equations": "After k iterations: α_marked = sin((2k+1)θ), α_unmarked = -cos((2k+1)θ)/√(N-M)"
            },
            "algorithm_choice": {
                "approach_1": "Standard Grover's algorithm - O(√N) complexity, deterministic oracle construction",
                "approach_2": "Amplitude amplification variant - More general but same complexity for this problem",
                "approach_3": "Quantum random walk - O(√N) but with different constant factors and complexity",
                "pros_cons_analysis": "Standard Grover's: Simple, well-studied, optimal. AA: More flexible but overkill. QRW: Interesting but less efficient",
                "final_choice_reasoning": "Standard Grover's chosen for simplicity, optimality, and direct applicability to marked state search"
            },
            "parameter_selection": {
                "mathematical_derivation": "θ = arcsin(√(M/N)) = arcsin(√(1/4)) = arcsin(1/2) = π/6",
                "optimal_iterations": "k_opt = ⌊π/(4θ)⌋ = ⌊π/(4·π/6)⌋ = ⌊3/2⌋ = 1",
                "optimization_criteria": "Maximize P(success) = sin²((2k+1)θ) over integer k",
                "sensitivity_analysis": "k=0: P=0.25, k=1: P=1.0, k=2: P=0.25 - clear optimum at k=1",
                "robustness_assessment": "Algorithm tolerates small oracle phase errors up to ±π/8 with >90% success probability"
            },
            "implementation_strategy": {
                "step_1": "Initialize 2-qubit quantum register and classical register for measurement",
                "step_2": "Apply Hadamard gates to both qubits: H⊗H|00⟩ = 1/2(|00⟩+|01⟩+|10⟩+|11⟩)",
                "step_3": "Implement oracle: Apply controlled-Z gate CZ(q₀,q₁) to flip phase of |11⟩",
                "step_4": "Implement diffusion operator: Apply H⊗H, then X⊗X, then CZ(q₀,q₁), then X⊗X, then H⊗H",
                "step_5": "Repeat oracle+diffusion for optimal iteration count (1 iteration)",
                "step_6": "Apply measurement to both qubits in computational basis",
                "step_7": "Post-process measurement results to extract probabilities",
                "step_8": "Verify success by checking P(|11⟩) > 0.8 threshold",
                "error_handling": "Check for decoherence effects, gate fidelity issues, measurement errors",
                "validation_strategy": "Run multiple shots (≥1000) to achieve statistical confidence in results"
            },
            "complexity_analysis": {
                "time_complexity": "O(√N) = O(√4) = O(1) for this specific problem size",
                "space_complexity": "O(log N) = O(2) qubits required for state representation", 
                "circuit_depth": "Depth = 2H + 1CZ + 4H + 2X + 1CZ + 2X + 1measure = 13 gates deep",
                "gate_count": "Total gates: 6H + 2CZ + 4X + 2measure = 14 gates",
                "scalability": "For n-qubit systems: gate count scales as O(n), depth as O(√2ⁿ)"
            },
            "error_analysis": {
                "decoherence_effects": "T₁ and T₂ times limit coherent evolution, require gates faster than ~10μs",
                "gate_fidelity": "Each gate introduces ~0.1% error, total fidelity ≈ (0.999)¹⁴ ≈ 98.6%",
                "measurement_errors": "Readout fidelity ~99%, affects final probability extraction",
                "mitigation_strategies": "Error correction codes, dynamical decoupling, gate optimization",
                "robustness_bounds": "Algorithm maintains >80% success with gate errors up to 1% per operation"
            },
            "optimization_opportunities": {
                "gate_reduction": "CZ can be decomposed more efficiently on specific hardware architectures",
                "parallel_operations": "Diffusion operator gates can be partially parallelized",
                "hardware_specific": "Use native gate sets (e.g., √X, CPhase) to reduce compilation overhead",
                "circuit_synthesis": "Optimize gate ordering to minimize cross-talk and improve fidelity",
                "adaptive_methods": "Use quantum error mitigation techniques like zero-noise extrapolation"
            }
        }
        
        # Create dataset based on extended thinking mode
        if self.extended_thinking:
            reasoning_trace = extended_reasoning
            dataset_type = "extended_thinking"
        else:
            reasoning_trace = standard_reasoning
            dataset_type = "standard"
            
        dataset = {
            "problem_id": f"grover_2qubit_{dataset_type}",
            "problem_description": "Implement Grover's algorithm to search for state |11⟩ in a 2-qubit system.",
            "difficulty": "intermediate", 
            "category": "quantum_algorithms",
            "learning_objectives": ["Grover's algorithm", "Oracle construction", "Amplitude amplification"],
            "prerequisites": ["Basic quantum gates", "Quantum superposition"],
            "reasoning_trace": reasoning_trace,
            "solutions": [{
                "preference_rank": 1,
                "code": "# Standard Grover's implementation code would go here",
                "expected_output": "{'11': ~1000}",
                "output_interpretation": "High probability of |11⟩ confirms successful implementation"
            }],
            "common_mistakes": [{
                "incorrect_code": "# Common mistake code would go here",
                "error_type": "conceptual_misunderstanding",
                "explanation": "Example of typical error",
                "expected_output": "{'00': ~1000}"
            }]
        }
        
        return dataset

def compare_reasoning_length():
    """Compare reasoning trace lengths between standard and extended thinking."""
    
    print("🔍 COMPARING STANDARD vs EXTENDED THINKING")
    print("=" * 60)
    
    # Generate standard dataset
    print("\n📋 Generating Standard Reasoning Dataset...")
    standard_gen = MockDatasetGenerator(extended_thinking=False)
    standard_dataset = standard_gen.generate_mock_dataset()
    
    # Generate extended thinking dataset
    print("📋 Generating Extended Thinking Dataset...")
    extended_gen = MockDatasetGenerator(extended_thinking=True)
    extended_dataset = extended_gen.generate_mock_dataset()
    
    # Save datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    standard_file = f"datasets/qiskit_dataset_standard_thinking_{timestamp}.json"
    extended_file = f"datasets/qiskit_dataset_extended_thinking_{timestamp}.json"
    
    import os
    os.makedirs("datasets", exist_ok=True)
    
    with open(standard_file, 'w', encoding='utf-8') as f:
        json.dump(standard_dataset, f, indent=2, ensure_ascii=False)
    
    with open(extended_file, 'w', encoding='utf-8') as f:
        json.dump(extended_dataset, f, indent=2, ensure_ascii=False)
    
    # Analysis
    def analyze_reasoning_trace(reasoning_trace, name):
        if isinstance(reasoning_trace, dict):
            total_chars = 0
            total_sections = len(reasoning_trace)
            
            print(f"\n{name} REASONING ANALYSIS:")
            print("-" * 40)
            
            for section, content in reasoning_trace.items():
                if isinstance(content, str):
                    chars = len(content)
                    words = len(content.split())
                    print(f"  {section}: {chars} chars, {words} words")
                    total_chars += chars
                elif isinstance(content, dict):
                    # Extended thinking has nested structure
                    section_chars = sum(len(str(v)) for v in content.values())
                    section_words = sum(len(str(v).split()) for v in content.values())
                    print(f"  {section}: {section_chars} chars, {section_words} words ({len(content)} subsections)")
                    total_chars += section_chars
            
            print(f"\nTOTAL: {total_chars} characters, {total_sections} main sections")
            return total_chars, total_sections
        
        return 0, 0
    
    # Analyze both datasets
    standard_chars, standard_sections = analyze_reasoning_trace(
        standard_dataset["reasoning_trace"], "STANDARD"
    )
    
    extended_chars, extended_sections = analyze_reasoning_trace(
        extended_dataset["reasoning_trace"], "EXTENDED"
    )
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Standard reasoning:  {standard_chars:,} characters, {standard_sections} sections")
    print(f"Extended reasoning:  {extended_chars:,} characters, {extended_sections} sections")
    
    if extended_chars > 0 and standard_chars > 0:
        ratio = extended_chars / standard_chars
        increase = ((extended_chars - standard_chars) / standard_chars) * 100
        print(f"Size increase:       {ratio:.1f}x larger ({increase:.0f}% increase)")
    
    print(f"\nFiles saved:")
    print(f"  Standard: {standard_file}")
    print(f"  Extended: {extended_file}")
    
    return standard_file, extended_file

if __name__ == "__main__":
    standard_file, extended_file = compare_reasoning_length()
    print(f"\n✅ Comparison complete!")
    print(f"\nTo validate datasets:")
    print(f"  python validate_dataset.py {standard_file}")
    print(f"  python validate_dataset.py {extended_file}")