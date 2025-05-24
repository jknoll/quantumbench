# Qiskit Fine-tuning Dataset Generation Request

I need you to generate a comprehensive dataset for fine-tuning a smaller language model to produce valid, performant Qiskit code with strong reasoning capabilities. This dataset will be used for both supervised fine-tuning and preference-based methods like DPO.

## Dataset Structure Requirements

For each problem, provide a JSON object with the following structure:

```json
{
  "problem_id": "unique_identifier",
  "problem_description": "Clear natural language description of the quantum computing problem",
  "difficulty": "beginner|intermediate|advanced",
  "category": "circuit_construction|quantum_algorithms|optimization|error_mitigation|hardware_interaction|visualization",
  "learning_objectives": ["what concepts this teaches"],
  "prerequisites": ["assumed knowledge"],
  
  "reasoning_trace": {
    "problem_analysis": "Step-by-step breakdown of what the problem is asking",
    "quantum_physics_reasoning": "Relevant quantum mechanical principles and intuition",
    "algorithm_choice": "Why this approach/algorithm is appropriate",
    "parameter_selection": "How to choose parameters (iterations, angles, etc.) with mathematical justification",
    "implementation_strategy": "High-level approach to the implementation"
  },
  
  "solutions": [
    {
      "preference_rank": 1,
      "code": "Complete, executable Qiskit code",
      "code_reasoning": {
        "structure_choice": "Why this code structure was chosen",
        "gate_implementation": "Rationale for specific gate choices and arrangements",
        "optimization_decisions": "Performance and efficiency considerations",
        "measurement_strategy": "Why this measurement approach"
      },
      "expected_output": "What the code produces when executed",
      "output_interpretation": "How to understand and validate the results"
    },
    {
      "preference_rank": 2,
      "code": "Alternative correct but less optimal implementation",
      "code_reasoning": {...},
      "expected_output": "...",
      "output_interpretation": "...",
      "why_less_preferred": "Explanation of why this is ranked lower"
    }
  ],
  
  "common_mistakes": [
    {
      "incorrect_code": "Code with typical errors",
      "error_type": "bug|inefficiency|conceptual_misunderstanding",
      "explanation": "Why this is wrong and how to fix it",
      "debugging_reasoning": "Step-by-step process to identify and correct the error"
    }
  ],
  
  "validation_tests": [
    {
      "test_description": "How to verify the code works correctly",
      "test_code": "Code to validate the implementation",
      "expected_test_result": "What the test should show"
    }
  ],
  
  "extensions": [
    "How this problem could be extended or made more complex",
    "Related problems or variations"
  ]
}
```

## Content Requirements

### Topic Coverage
Generate problems covering these areas with the specified distribution:

**Basic Circuit Construction (15 problems):**
- Single and multi-qubit gates
- Circuit composition and measurement
- Parameterized circuits
- Circuit visualization and analysis

**Quantum Algorithms (25 problems):**
- Grover's algorithm (various sizes and oracle functions)
- Quantum Fourier Transform and applications
- Variational algorithms (VQE, QAOA)
- Quantum phase estimation
- Shor's algorithm components
- Deutsch-Jozsa, Bernstein-Vazirani
- Amplitude amplification variants

**Optimization and Transpilation (10 problems):**
- Circuit optimization techniques
- Backend-specific transpilation
- Gate decomposition strategies
- Noise-aware compilation

**Error Mitigation and Noise (10 problems):**
- Error mitigation techniques
- Noise modeling and simulation
- Quantum error correction basics
- Readout error correction

**Hardware Interaction (10 problems):**
- Backend selection and configuration
- Job submission and monitoring
- Hardware constraint handling
- Real device vs simulator differences

**Advanced Topics (10 problems):**
- Custom gate definitions
- Pulse-level programming basics
- Quantum machine learning applications
- Hybrid classical-quantum algorithms

### Reasoning Depth Requirements

For each problem, include reasoning that demonstrates:

1. **Quantum Physics Understanding**: Show how quantum mechanical principles guide the solution
2. **Mathematical Rigor**: Include relevant calculations, especially for algorithm parameters
3. **Implementation Trade-offs**: Explain choices between different valid approaches
4. **Practical Considerations**: Address real-world constraints like hardware limitations
5. **Error Analysis**: Anticipate common failure modes and debugging strategies

### Code Quality Requirements

- All code must be complete, executable, and properly commented
- Include necessary imports and backend setup
- Use current Qiskit syntax and best practices
- Provide multiple working solutions where applicable
- Include both simulator and hardware-compatible versions when relevant

### Validation Requirements

Each solution should include:
- Clear expected outputs (measurement results, circuit properties, etc.)
- Methods to verify correctness
- Performance benchmarks where applicable (circuit depth, gate count, etc.)

## Output Format

Please generate {num_examples} problem(s) total following the above distribution. Output each problem as a separate JSON object, properly formatted for easy parsing. Include a brief introduction explaining the dataset's structure and intended use.

## Additional Instructions

- Ensure problems progress from basic to advanced within each category
- Include both textbook-standard implementations and more creative/optimized approaches
- Make reasoning traces educational - they should teach quantum computing concepts
- Ensure incorrect examples in common_mistakes represent realistic errors beginners make
- Include sufficient mathematical detail for advanced problems
- Consider both theoretical understanding and practical implementation skills

Generate the complete dataset now, ensuring each problem is fully specified according to the above structure.