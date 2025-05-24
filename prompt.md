# Qiskit Fine-tuning Dataset Generation Request

I need you to generate a comprehensive dataset for fine-tuning a smaller language model to produce valid, performant Qiskit code with strong reasoning capabilities. This dataset will be used for supervised fine-tuning with prompt-completion pairs.

## Dataset Structure Requirements

For each problem, provide a JSON object with the following structure:

```json
{
  "problem_id": "unique_identifier",
  "prompt": "Clear natural language description of the quantum computing problem",
  "difficulty": "beginner|intermediate|advanced",
  "category": "circuit_construction|quantum_algorithms|optimization|error_mitigation|hardware_interaction|visualization",
  "learning_objectives": ["what concepts this teaches"],
  "prerequisites": ["assumed knowledge"],
  
  "reasoning_trace": "Comprehensive step-by-step reasoning that explains the quantum physics concepts, algorithm choice, parameter selection, and implementation strategy needed to solve this problem",
  
  "solution": {
    "code": "Complete, executable Qiskit code",
    "expected_output": "What the code produces when executed",
    "output_interpretation": "How to understand and validate the results"
  },
  
  "extensions": [
    "How this problem could be extended or made more complex",
    "Related problems or variations"
  ]
}
```

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

The reasoning trace should be a comprehensive narrative that demonstrates:

1. **Quantum Physics Understanding**: How quantum mechanical principles guide the solution
2. **Mathematical Rigor**: Relevant calculations, especially for algorithm parameters
3. **Algorithm Choice**: Why this approach is appropriate and how it works
4. **Implementation Strategy**: Step-by-step approach to building the solution
5. **Parameter Selection**: How to choose parameters with mathematical justification

### Code Quality Requirements

- All code must be complete, executable, and properly commented
- Include necessary imports and backend setup
- Use current Qiskit syntax and best practices
- Include both simulator and hardware-compatible versions when relevant

### Validation Requirements

Each solution should include:
- Clear expected outputs (measurement results, circuit properties, etc.)
- Methods to verify correctness
- Performance benchmarks where applicable (circuit depth, gate count, etc.)

## General Instructions

- Ensure problems progress from basic to advanced within each category
- Focus on the most effective and educational implementation approach for each problem
- Make reasoning traces comprehensive and educational - they should teach quantum computing concepts
- Include sufficient mathematical detail for advanced problems
- Consider both theoretical understanding and practical implementation skills
- Ensure each solution represents best practices and optimal approaches

## Extended Thinking Mode Configuration

Extended thinking is currently set to: **{extended_thinking}**

If extended thinking is **enabled (true)**, you must provide significantly more detailed reasoning traces with the following enhancements:

### Enhanced Reasoning Trace Requirements (when extended_thinking = true):
The reasoning trace should be a comprehensive narrative (8-12 paragraphs) that includes:
- **Detailed problem decomposition**: Break down into 5-7 sub-problems with mathematical foundations
- **Quantum physics deep dive**: Mathematical formalism, state vectors, operators, and evolution equations
- **Algorithm analysis**: Compare alternative approaches with detailed pros/cons analysis
- **Mathematical derivations**: Include parameter selection with optimization criteria and sensitivity analysis
- **Implementation strategy**: Provide 8-10 detailed steps with error handling considerations
- **Complexity analysis**: Computational complexity, circuit depth analysis, and scalability considerations
- **Error analysis**: Potential sources of error, mitigation strategies, and robustness assessment
- **Optimization opportunities**: Performance improvements, gate count reduction, and hardware-specific optimizations

### Standard Reasoning Trace (when extended_thinking = false):
Provide a concise but comprehensive reasoning trace (3-5 paragraphs) that covers the quantum physics concepts, algorithm choice, parameter selection, and implementation strategy in a clear, educational narrative.

## Output Format and Final Instructions

Please generate {num_examples} problem(s) total following the distribution specified above. Output each problem as a separate JSON object, properly formatted for easy parsing. Include a brief introduction explaining the dataset's structure and intended use.

Generate the complete dataset now, ensuring each problem is fully specified according to the structure and requirements outlined in this prompt.