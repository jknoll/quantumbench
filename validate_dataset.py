#!/usr/bin/env python3
"""
Dataset Validation Script

This script reads qiskit_dataset.json and executes the code from each entry,
comparing expected vs actual outputs and providing a comprehensive summary.
"""

import json
import re
import sys
import traceback
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from io import StringIO
import contextlib

import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import MCXGate


@dataclass
class ValidationResult:
    """Container for validation results of a single code entry."""
    problem_id: str
    difficulty: str
    category: str
    preference_rank: Optional[int]
    entry_type: str  # 'solution' or 'common_mistake'
    code_executed: bool
    expected_output: str
    actual_output: str
    outputs_match: bool
    rmse_error: Optional[float]
    execution_error: Optional[str]


class DatasetValidator:
    """Validates Qiskit dataset entries by executing code and comparing outputs."""
    
    def __init__(self, dataset_path: str = "qiskit_dataset.json"):
        self.dataset_path = dataset_path
        self.results: List[ValidationResult] = []
        self.is_directory = os.path.isdir(dataset_path)
        self.dataset_files = []
        
        if self.is_directory:
            # Find all JSON files in the directory
            json_files = []
            for filename in os.listdir(dataset_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(dataset_path, filename)
                    json_files.append(filepath)
            
            # Sort by filename for consistent processing order
            self.dataset_files = sorted(json_files)
            print(f"Found {len(self.dataset_files)} JSON files in directory: {dataset_path}")
            for f in self.dataset_files:
                print(f"  - {os.path.basename(f)}")
        else:
            # Single file mode
            self.dataset_files = [dataset_path]
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load and parse the dataset files."""
        if len(self.dataset_files) == 1:
            # Single file mode - use existing logic
            return self._load_single_dataset(self.dataset_files[0])
        else:
            # Multiple files mode - create a collection
            return self._load_multiple_datasets()
    
    def _load_single_dataset(self, dataset_file: str) -> Dict[str, Any]:
        """Load and parse a single dataset file."""
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # If it's a valid single dataset (has problem_id), return it directly
            if 'problem_id' in dataset:
                print(f"✓ Dataset loaded successfully: {os.path.basename(dataset_file)}")
                return dataset
            
            # If it's a collection format (has problems array), return collection metadata
            if 'problems' in dataset and len(dataset['problems']) > 0:
                print(f"✓ Dataset collection loaded successfully: {os.path.basename(dataset_file)} ({len(dataset['problems'])} problems)")
                return dataset
            
            # Check if dataset contains a parsing error with raw_response
            if 'error' in dataset and 'raw_response' in dataset:
                # Try to extract JSON from raw_response
                raw_response = dataset['raw_response']
                json_match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    
                    # Try to fix common JSON issues
                    # Fix the specific issue with the comma delimiter
                    json_str = json_str.replace('"Understanding Grover\'s algorithm components"', '"Understanding Grover\'s algorithm components"')
                    json_str = json_str.replace('    "Oracle construction for specific marked states",', '    "Oracle construction for specific marked states",')
                    
                    # Try to handle the problematic line around character 2124
                    # This seems to be related to newlines in strings
                    lines = json_str.split('\n')
                    for i, line in enumerate(lines):
                        if 'problem_analysis' in line:
                            # Fix the multi-line string issue
                            lines[i] = line.replace('"1. Need to search for state |11⟩ in 2-qubit space\\n2. Search space size N = 2^2 = 4\\n3. Required components: oracle, diffusion operator, iteration circuit\\n4. Optimal iterations = ⌊π/4 * √N⌋ ≈ 1 for N=4"', 
                                                  '"1. Need to search for state |11⟩ in 2-qubit space. 2. Search space size N = 2^2 = 4. 3. Required components: oracle, diffusion operator, iteration circuit. 4. Optimal iterations = ⌊π/4 * √N⌋ ≈ 1 for N=4"')
                    
                    json_str = '\n'.join(lines)
                    
                    try:
                        parsed_dataset = json.loads(json_str)
                        print("✓ Successfully extracted and fixed JSON from raw_response")
                        return parsed_dataset
                    except json.JSONDecodeError as e:
                        print(f"✗ Failed to parse extracted JSON even after fixes: {e}")
                        print(f"Error around character {e.pos}")
                        # Print some context around the error
                        start = max(0, e.pos - 50)
                        end = min(len(json_str), e.pos + 50)
                        print(f"Context: ...{json_str[start:end]}...")
                        return None
                else:
                    print("✗ No JSON found in raw_response")
                    return None
            
            return dataset
        except FileNotFoundError:
            print(f"✗ Dataset file '{dataset_file}' not found")
            return None
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse dataset JSON: {e}")
            return None
    
    def _load_multiple_datasets(self) -> Dict[str, Any]:
        """Load and combine multiple dataset files into a collection."""
        problems = []
        total_loaded = 0
        total_failed = 0
        
        print(f"Loading {len(self.dataset_files)} dataset files...")
        
        for dataset_file in self.dataset_files:
            try:
                dataset = self._load_single_dataset(dataset_file)
                if dataset and 'problem_id' in dataset:
                    # Add source file info to dataset
                    dataset['source_file'] = os.path.basename(dataset_file)
                    problems.append(dataset)
                    total_loaded += 1
                elif dataset and 'problems' in dataset:
                    # Handle collection files - merge their problems
                    for problem in dataset['problems']:
                        problem['source_file'] = os.path.basename(dataset_file)
                        problems.append(problem)
                    total_loaded += len(dataset['problems'])
                else:
                    print(f"⚠️  Skipping {os.path.basename(dataset_file)} - invalid format")
                    total_failed += 1
            except Exception as e:
                print(f"✗ Failed to load {os.path.basename(dataset_file)}: {e}")
                total_failed += 1
        
        print(f"✓ Loaded {total_loaded} problems from {len(self.dataset_files)} files ({total_failed} failed)")
        
        # Create collection format
        collection = {
            "metadata": {
                "collection_type": "directory_validation",
                "source_directory": self.dataset_path,
                "total_files": len(self.dataset_files),
                "loaded_problems": total_loaded,
                "failed_files": total_failed,
                "created_at": datetime.now().isoformat()
            },
            "problems": problems
        }
        
        return collection
    
    def preprocess_code(self, code: str) -> str:
        """Preprocess code to handle import and compatibility issues."""
        import re
        
        # Remove import lines but preserve the functionality in namespace
        # Note: All needed imports are provided in the execution namespace
        processed = re.sub(r'^from qiskit import.*$', '# Import handled by validation framework', code, flags=re.MULTILINE)
        processed = re.sub(r'^import qiskit.*$', '# Import handled by validation framework', processed, flags=re.MULTILINE)
        processed = re.sub(r'^from qiskit_aer import.*$', '# Import handled by validation framework', processed, flags=re.MULTILINE)
        processed = re.sub(r'^import numpy.*$', '# Import handled by validation framework', processed, flags=re.MULTILINE)
        processed = re.sub(r'^from numpy import.*$', '# Import handled by validation framework', processed, flags=re.MULTILINE)
        
        # For solutions that don't explicitly execute, add execution
        if 'create_grover_2qubit_alternative()' in processed and 'execute(' not in processed:
            processed += '\n\n# Execute the circuit\ncircuit = create_grover_2qubit_alternative()\nsimulator = Aer.get_backend(\'qasm_simulator\')\nresult = execute(circuit, simulator, shots=1000).result()\ncounts = result.get_counts()\nprint(counts)'
        
        # Handle legacy code snippets (for backwards compatibility with old datasets)
        # New datasets should have complete, executable common mistakes
        
        # For code that references 'qc' but doesn't define it (legacy support)
        if 'qc.' in processed and 'qc =' not in processed and 'QuantumCircuit(' not in processed and len(processed.strip().split('\n')) < 5:
            # Only add context for very short snippets (legacy datasets)
            if 'qc.ccz(' in processed and '0, 1, 2' in processed:
                processed = 'qc = QuantumCircuit(3, 3)\n' + processed
            else:
                processed = 'qc = QuantumCircuit(2, 2)\n' + processed
        
        return processed
    
    def safe_execute_code(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """Safely execute code and capture output."""
        try:
            # Capture stdout
            old_stdout = sys.stdout
            captured_output = StringIO()
            
            # Helper function for backward compatibility  
            class MockJob:
                def __init__(self, result):
                    self._result = result
                def result(self):
                    return self._result
            
            def execute_circuit(circuit, backend, shots=1000):
                job = backend.run(circuit, shots=shots)
                return MockJob(job.result())
            
            # Pre-process code to handle import issues and __main__ execution
            processed_code = self.preprocess_code(code)
            
            # Handle if __name__ == "__main__": blocks
            # Replace the conditional with direct execution for validation
            import re
            processed_code = re.sub(
                r'if __name__ == ["\']__main__["\']:\s*\n',
                '# Main execution block (modified for validation):\nif True:  # Execute main block\n',
                processed_code,
                flags=re.MULTILINE
            )
            
            # Create a comprehensive namespace for execution
            # This includes all common imports that generated code might use
            namespace = {
                # Core Qiskit
                'QuantumCircuit': QuantumCircuit,
                'QuantumRegister': QuantumRegister, 
                'ClassicalRegister': ClassicalRegister,
                'transpile': transpile,
                
                # Qiskit Aer
                'Aer': Aer,
                'AerSimulator': AerSimulator,
                'execute': execute_circuit,
                
                # Quantum Info
                'Statevector': Statevector,
                
                # Circuit Library
                'MCXGate': MCXGate,
                
                # NumPy
                'np': np,
                'numpy': np,
                
                # Built-in functions
                'print': lambda *args, **kwargs: print(*args, file=captured_output, **kwargs),
                'range': range,
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                
                # Math functions commonly used
                'pi': np.pi,
                'sqrt': np.sqrt,
                'sin': np.sin,
                'cos': np.cos,
                
                # Common circuit variables (for incomplete code snippets)
                'qr': QuantumRegister(3, 'q'),  # Default quantum register
                'cr': ClassicalRegister(3, 'c'),  # Default classical register
                'circuit': None,  # Will be set if circuit is defined
                'qc': None,  # Common variable name for quantum circuit
                'qreg': QuantumRegister(3, 'q'),  # Alternative register naming
                'creg': ClassicalRegister(3, 'c'),  # Alternative register naming
                'qubits': [0, 1, 2],  # Default qubit indices
                
                # Special variables for proper execution
                '__name__': '__main__',  # Ensure __name__ == "__main__" works
            }
            
            with contextlib.redirect_stdout(captured_output):
                # Execute the processed code
                exec(processed_code, namespace)
                
                # Get the captured printed output first (this is what we want to compare)
                captured_text = captured_output.getvalue().strip()
                
                if captured_text:
                    # Use the printed output as the primary result
                    output = captured_text
                elif 'counts' in namespace:
                    # Fallback to namespace variables if no printed output
                    output = str(namespace['counts'])
                elif 'result' in namespace:
                    output = str(namespace['result'])
                else:
                    # Check for other possible result variables as last resort
                    for var_name, var_value in namespace.items():
                        if var_name in ['circuit', 'simulator', 'backend']:
                            continue
                        if hasattr(var_value, 'get_counts') or 'count' in str(type(var_value)).lower():
                            output = f"{var_name}: {str(var_value)}"
                            break
                    else:
                        output = "No output captured"
            
            sys.stdout = old_stdout
            return True, output, None
            
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, "", error_msg
    
    def calculate_rmse(self, expected: str, actual: str) -> Optional[float]:
        """Calculate RMSE between expected and actual outputs when possible."""
        try:
            # Try to extract numerical values from strings
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            actual_nums = re.findall(r'-?\d+\.?\d*', actual)
            
            if len(expected_nums) != len(actual_nums) or len(expected_nums) == 0:
                return None
            
            expected_values = [float(x) for x in expected_nums]
            actual_values = [float(x) for x in actual_nums]
            
            return float(np.sqrt(np.mean((np.array(expected_values) - np.array(actual_values))**2)))
        except:
            return None
    
    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Compare expected and actual outputs with some tolerance for quantum results."""
        if expected == actual:
            return True
        
        # For quantum algorithms, we need to compare the key metrics and results
        # rather than exact text matches due to randomness and formatting differences
        
        # Extract key metrics from both outputs
        expected_metrics = self._extract_metrics(expected)
        actual_metrics = self._extract_metrics(actual)
        
        # Compare circuit statistics (if present)
        if self._compare_circuit_stats(expected_metrics, actual_metrics):
            return True
        
        # Compare quantum measurement results (most important)
        if self._compare_quantum_measurements(expected_metrics, actual_metrics):
            return True
        
        # Compare success probabilities and quantum speedup
        if self._compare_algorithm_performance(expected_metrics, actual_metrics):
            return True
        
        # Legacy comparison patterns for backward compatibility
        if self._legacy_pattern_matching(expected, actual):
            return True
        
        return False
    
    def _extract_metrics(self, output: str) -> Dict[str, Any]:
        """Extract key metrics from output text."""
        metrics = {}
        
        # Extract circuit depth and gate counts
        depth_match = re.search(r'Circuit depth:\s*(\d+)', output)
        if depth_match:
            metrics['circuit_depth'] = int(depth_match.group(1))
        
        # Extract gate counts (handles both dict format and individual counts)
        gate_count_match = re.search(r'Gate count:\s*({[^}]+})', output)
        if gate_count_match:
            try:
                # Parse the gate count dictionary
                gate_dict_str = gate_count_match.group(1)
                # Convert to proper Python dict format
                gate_dict_str = gate_dict_str.replace("'", '"')
                import json
                metrics['gate_counts'] = json.loads(gate_dict_str)
            except:
                pass
        
        # Extract success probability for quantum algorithms
        prob_match = re.search(r'probability:\s*(0\.\d+)', output)
        if prob_match:
            metrics['success_probability'] = float(prob_match.group(1))
        
        # Extract quantum speedup factor
        speedup_match = re.search(r'speedup factor:\s*([\d.]+)x', output)
        if speedup_match:
            metrics['speedup_factor'] = float(speedup_match.group(1))
        
        # Extract measurement counts (for Grover's algorithm specifically)
        # Look for |101⟩ state counts (the target state)
        target_count_match = re.search(r"'101':\s*(\d+)", output)
        if target_count_match:
            metrics['target_state_count'] = int(target_count_match.group(1))
        
        # Extract total shots
        shots_match = re.search(r'(\d+)\s+shots', output)
        if shots_match:
            metrics['total_shots'] = int(shots_match.group(1))
        
        return metrics
    
    def _compare_circuit_stats(self, expected: Dict, actual: Dict) -> bool:
        """Compare circuit statistics with reasonable tolerance."""
        if 'circuit_depth' in expected and 'circuit_depth' in actual:
            # Allow 20% tolerance for circuit depth due to optimization differences
            exp_depth = expected['circuit_depth']
            act_depth = actual['circuit_depth']
            if abs(exp_depth - act_depth) / exp_depth <= 0.2:
                return True
        return False
    
    def _compare_quantum_measurements(self, expected: Dict, actual: Dict) -> bool:
        """Compare quantum measurement results for algorithm correctness."""
        # For Grover's algorithm, check if target state is dominant
        if 'target_state_count' in expected and 'target_state_count' in actual:
            exp_count = expected['target_state_count']
            act_count = actual['target_state_count']
            
            # Both should show the target state as highly probable (>80% of shots)
            # Assuming ~8192 shots, target should have >6500 counts
            if exp_count > 6500 and act_count > 6500:
                return True
            
            # Alternative: compare relative dominance
            if 'total_shots' in expected and 'total_shots' in actual:
                exp_prob = exp_count / expected['total_shots']
                act_prob = act_count / actual['total_shots']
                # Both should show >80% success rate
                if exp_prob > 0.8 and act_prob > 0.8:
                    return True
        
        return False
    
    def _compare_algorithm_performance(self, expected: Dict, actual: Dict) -> bool:
        """Compare algorithm performance metrics."""
        # Compare success probability
        if 'success_probability' in expected and 'success_probability' in actual:
            exp_prob = expected['success_probability']
            act_prob = actual['success_probability']
            # Allow 10% tolerance for quantum algorithms due to statistical variance
            if abs(exp_prob - act_prob) / exp_prob <= 0.1:
                return True
        
        # Compare speedup factor
        if 'speedup_factor' in expected and 'speedup_factor' in actual:
            exp_speedup = expected['speedup_factor']
            act_speedup = actual['speedup_factor']
            # Allow 15% tolerance for speedup calculations
            if abs(exp_speedup - act_speedup) / exp_speedup <= 0.15:
                return True
        
        return False
    
    def _legacy_pattern_matching(self, expected: str, actual: str) -> bool:
        """Legacy pattern matching for backward compatibility."""
        # Handle ~1000 notation in expected output (approximate match)
        if '~1000' in expected and '11' in expected:
            # Look for {'11': exact_number} in actual
            actual_match = re.search(r"'11':\s*(\d+)", actual)
            if actual_match:
                act_count = int(actual_match.group(1))
                # Accept if '11' state is dominant (>80% of shots)
                return act_count > 800
        
        # For quantum measurement results, check if the dominant state matches
        if '11' in expected and '11' in actual:
            # Extract counts if possible
            expected_match = re.search(r"'11':\s*(\d+)", expected)
            actual_match = re.search(r"'11':\s*(\d+)", actual)
            
            if expected_match and actual_match:
                exp_count = int(expected_match.group(1))
                act_count = int(actual_match.group(1))
                # Consider it a match if both show |11⟩ as dominant state (>50% of shots)
                return exp_count > 500 and act_count > 500
        
        # Handle more general ~number patterns
        approx_match = re.search(r"~(\d+)", expected)
        if approx_match:
            target_value = int(approx_match.group(1))
            # Extract all numbers from actual output
            actual_numbers = re.findall(r'\d+', actual)
            if actual_numbers:
                # Check if any number is close to target (within 20%)
                for num_str in actual_numbers:
                    num = int(num_str)
                    if target_value == 0:
                        if num == 0:
                            return True
                    elif abs(num - target_value) / target_value < 0.2:
                        return True
        
        return False
    
    def validate_solutions(self, dataset: Dict[str, Any]) -> None:
        """Validate all solutions in the dataset or dataset collection."""
        # Handle collection format
        if 'problems' in dataset:
            problems = dataset['problems']
            print(f"Validating solutions across {len(problems)} problems in collection")
            for problem in problems:
                self._validate_solutions_for_problem(problem)
            return
        
        # Handle single problem format
        self._validate_solutions_for_problem(dataset)
    
    def _validate_solutions_for_problem(self, dataset: Dict[str, Any]) -> None:
        """Validate solutions for a single problem."""
        # Handle both old format (solutions array) and new format (single solution)
        if 'solution' in dataset:
            # New simplified format
            solution = dataset['solution']
            print(f"Found 1 solution to validate for problem {dataset.get('problem_id', 'unknown')}")
            
            result = ValidationResult(
                problem_id=dataset.get('problem_id', 'unknown'),
                difficulty=dataset.get('difficulty', 'unknown'),
                category=dataset.get('category', 'unknown'),
                preference_rank=1,  # Single solution is always rank 1
                entry_type='solution',
                code_executed=False,
                expected_output=solution.get('expected_output', ''),
                actual_output='',
                outputs_match=False,
                rmse_error=None,
                execution_error=None
            )
            
            # Execute the code
            code = solution.get('code', '')
            if code:
                success, output, error = self.safe_execute_code(code)
                result.code_executed = success
                result.actual_output = output if success else ''
                result.execution_error = error
                
                if success:
                    result.outputs_match = self.compare_outputs(result.expected_output, result.actual_output)
                    if not result.outputs_match:
                        result.rmse_error = self.calculate_rmse(result.expected_output, result.actual_output)
            
            self.results.append(result)
            
        elif 'solutions' in dataset:
            # Old format with multiple solutions
            solutions = dataset['solutions']
            print(f"Found {len(solutions)} solution(s) to validate for problem {dataset.get('problem_id', 'unknown')}")
            
            for solution in tqdm(solutions, desc=f"Validating solutions for {dataset.get('problem_id', 'unknown')}"):
                result = ValidationResult(
                    problem_id=dataset.get('problem_id', 'unknown'),
                    difficulty=dataset.get('difficulty', 'unknown'),
                    category=dataset.get('category', 'unknown'),
                    preference_rank=solution.get('preference_rank'),
                    entry_type='solution',
                    code_executed=False,
                    expected_output=solution.get('expected_output', ''),
                    actual_output='',
                    outputs_match=False,
                    rmse_error=None,
                    execution_error=None
                )
                
                # Execute the code
                code = solution.get('code', '')
                if code:
                    success, output, error = self.safe_execute_code(code)
                    result.code_executed = success
                    result.actual_output = output if success else ''
                    result.execution_error = error
                    
                    if success:
                        result.outputs_match = self.compare_outputs(result.expected_output, result.actual_output)
                        if not result.outputs_match:
                            result.rmse_error = self.calculate_rmse(result.expected_output, result.actual_output)
                
                self.results.append(result)
        else:
            print(f"✗ No solutions found in problem {dataset.get('problem_id', 'unknown')}")
    
    def validate_common_mistakes(self, dataset: Dict[str, Any]) -> None:
        """Validate common mistake examples in the dataset or dataset collection."""
        # Handle collection format
        if 'problems' in dataset:
            problems = dataset['problems']
            print(f"Validating common mistakes across {len(problems)} problems in collection")
            for problem in problems:
                self._validate_common_mistakes_for_problem(problem)
            return
        
        # Handle single problem format
        self._validate_common_mistakes_for_problem(dataset)
    
    def _validate_common_mistakes_for_problem(self, dataset: Dict[str, Any]) -> None:
        """Validate common mistakes for a single problem."""
        if 'common_mistakes' not in dataset:
            # Silently skip - common mistakes are optional in simplified format
            return
        
        mistakes = dataset['common_mistakes']
        print(f"Found {len(mistakes)} common mistake(s) to validate for problem {dataset.get('problem_id', 'unknown')}")
        
        for mistake in tqdm(mistakes, desc=f"Validating common mistakes for {dataset.get('problem_id', 'unknown')}"):
            # Use expected output from mistake if available, otherwise default
            expected_output = mistake.get('expected_output', "Should fail or produce incorrect results")
            
            result = ValidationResult(
                problem_id=dataset.get('problem_id', 'unknown'),
                difficulty=dataset.get('difficulty', 'unknown'),
                category=dataset.get('category', 'unknown'),
                preference_rank=None,
                entry_type='common_mistake',
                code_executed=False,
                expected_output=expected_output,
                actual_output='',
                outputs_match=False,
                rmse_error=None,
                execution_error=None
            )
            
            # Execute the code
            code = mistake.get('incorrect_code', '')
            if code:
                success, output, error = self.safe_execute_code(code)
                result.code_executed = success
                result.actual_output = output if success else ''
                result.execution_error = error
                
                # For common mistakes, check if output matches expected incorrect behavior
                if success and expected_output != "Should fail or produce incorrect results":
                    result.outputs_match = self.compare_outputs(expected_output, result.actual_output)
            
            self.results.append(result)
    
    def generate_summary_table(self) -> str:
        """Generate a summary table of all validation results."""
        if not self.results:
            return "No results to display"
        
        # Separate solutions and common mistakes
        solutions = [r for r in self.results if r.entry_type == 'solution']
        mistakes = [r for r in self.results if r.entry_type == 'common_mistake']
        
        output = []
        
        # Solutions table
        if solutions:
            output.append("SOLUTIONS VALIDATION RESULTS")
            output.append("=" * 50)
            output.append("Expected: All solutions should compile, execute, and produce correct outputs\n")
            
            data = []
            for result in solutions:
                data.append({
                    'Problem ID': result.problem_id,
                    'Preference Rank': result.preference_rank or 'N/A',
                    'Code Executed': '✓' if result.code_executed else '✗',
                    'Outputs Match': '✓' if result.outputs_match else '✗',
                    'RMSE Error': f"{result.rmse_error:.6f}" if result.rmse_error else 'N/A',
                    'Execution Error': result.execution_error or 'None'
                })
            
            df_solutions = pd.DataFrame(data)
            output.append(tabulate(df_solutions, headers='keys', tablefmt='grid', showindex=False))
            
            # Solutions statistics
            total_solutions = len(solutions)
            executed_solutions = sum(1 for r in solutions if r.code_executed)
            correct_solutions = sum(1 for r in solutions if r.outputs_match)
            
            output.append(f"\nSOLUTIONS SUMMARY:")
            output.append(f"Total solutions: {total_solutions}")
            output.append(f"Successfully executed: {executed_solutions}/{total_solutions} ({executed_solutions/total_solutions*100:.1f}%)")
            if executed_solutions > 0:
                output.append(f"Correct outputs: {correct_solutions}/{executed_solutions} ({correct_solutions/executed_solutions*100:.1f}% of executed)")
            else:
                output.append(f"Correct outputs: {correct_solutions}/{executed_solutions} (no successful executions)")
            output.append("")
        
        # Common mistakes table
        if mistakes:
            output.append("COMMON MISTAKES VALIDATION RESULTS")
            output.append("=" * 50)
            output.append("Expected: Should execute but produce incorrect/no results or demonstrate the mistake\n")
            
            data = []
            for result in mistakes:
                data.append({
                    'Problem ID': result.problem_id,
                    'Mistake Type': 'Code snippet',
                    'Code Executed': '✓' if result.code_executed else '✗',
                    'Demonstrates Mistake': '✓' if (result.code_executed and not result.outputs_match) else '✗',
                    'Execution Error': result.execution_error or 'None'
                })
            
            df_mistakes = pd.DataFrame(data)
            output.append(tabulate(df_mistakes, headers='keys', tablefmt='grid', showindex=False))
            
            # Common mistakes statistics
            total_mistakes = len(mistakes)
            executed_mistakes = sum(1 for r in mistakes if r.code_executed)
            demonstrates_mistake = sum(1 for r in mistakes if r.code_executed and not r.outputs_match)
            
            output.append(f"\nCOMMON MISTAKES SUMMARY:")
            output.append(f"Total mistake examples: {total_mistakes}")
            output.append(f"Successfully executed: {executed_mistakes}/{total_mistakes} ({executed_mistakes/total_mistakes*100:.1f}%)")
            if executed_mistakes > 0:
                output.append(f"Demonstrates mistake: {demonstrates_mistake}/{executed_mistakes} ({demonstrates_mistake/executed_mistakes*100:.1f}% of executed)")
            else:
                output.append(f"Demonstrates mistake: {demonstrates_mistake}/{executed_mistakes} (no successful executions)")
        
        return "\n".join(output)
    
    def extract_model_and_timestamp_from_filename(self, dataset_path: str) -> Tuple[str, str]:
        """Extract model name and timestamp from dataset filename or directory."""
        import re
        
        if self.is_directory:
            # For directory validation, extract from directory name or use first file
            dir_name = os.path.basename(dataset_path.rstrip('/'))
            
            # Check if directory name has model/timestamp info
            # Pattern: [model_name]_[timestamp] or similar
            dir_pattern = r'([a-zA-Z_0-9]+)_(\d{8}_\d{6})'
            dir_match = re.search(dir_pattern, dir_name)
            
            if dir_match:
                model_name = dir_match.group(1)
                timestamp = dir_match.group(2)
                print(f"Extracted from directory name: model='{model_name}', timestamp='{timestamp}'")
                return model_name, timestamp
            
            # Try to extract from first valid dataset file
            for dataset_file in self.dataset_files:
                try:
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    
                    if 'problem_id' in dataset:
                        # Extract from first file's content
                        model_name = self._extract_model_from_dataset(dataset)
                        
                        # Use directory creation time or first file time
                        try:
                            dir_stat = os.stat(dataset_path)
                            timestamp = datetime.fromtimestamp(dir_stat.st_mtime).strftime("%Y%m%d_%H%M%S")
                        except:
                            file_stat = os.stat(dataset_file)
                            timestamp = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y%m%d_%H%M%S")
                        
                        print(f"Extracted from directory content: model='{model_name}', timestamp='{timestamp}'")
                        return model_name, timestamp
                        
                except Exception:
                    continue
            
            # Fallback for directory
            return f"directory_{dir_name}", datetime.now().strftime("%Y%m%d_%H%M%S")
        
        else:
            # Single file mode - use existing logic
            return self._extract_from_single_file(dataset_path)
    
    def _extract_model_from_dataset(self, dataset: Dict[str, Any]) -> str:
        """Extract model name from dataset content."""
        # Look for model information in dataset metadata or reasoning
        model_name = "claude_3_5_sonnet"  # Default for basic datasets
        
        # Check reasoning trace for model hints
        if 'reasoning_trace' in dataset:
            reasoning_text = str(dataset['reasoning_trace'])
            if 'claude-sonnet-4' in reasoning_text or 'claude_sonnet_4' in reasoning_text:
                model_name = "claude_sonnet_4_20250514"
        
        # Check metadata if available
        if 'metadata' in dataset and 'model_used' in dataset['metadata']:
            model_name = dataset['metadata']['model_used'].replace('-', '_')
        
        return model_name
    
    def _extract_from_single_file(self, dataset_file: str) -> Tuple[str, str]:
        """Extract model and timestamp from a single dataset file."""
        import re
        
        # Get just the filename without path and the full path
        filename = os.path.basename(dataset_file)
        full_path = os.path.abspath(dataset_file)
        
        # Check if this file is in a timestamped directory (new format)
        parent_dir = os.path.dirname(full_path)
        parent_dir_name = os.path.basename(parent_dir)
        
        # Pattern for timestamped directory: [model_name]_[timestamp]
        dir_pattern = r'([a-zA-Z_0-9]+)_(\d{8}_\d{6})'
        dir_match = re.search(dir_pattern, parent_dir_name)
        
        if dir_match:
            model_name = dir_match.group(1)
            timestamp = dir_match.group(2)
            print(f"Extracted from parent directory '{parent_dir_name}': model='{model_name}', timestamp='{timestamp}'")
            return model_name, timestamp
        
        # Pattern to match legacy format: qiskit_dataset_[model_name]_[timestamp].json
        pattern = r'qiskit_dataset_(.+)_(\d{8}_\d{6})\.json'
        match = re.search(pattern, filename)
        
        if match:
            model_name = match.group(1)
            timestamp = match.group(2)
            print(f"Extracted from filename: model='{model_name}', timestamp='{timestamp}'")
            return model_name, timestamp
        else:
            # Check if it's a basic qiskit_dataset.json file or numbered file (1.json, 2.json, etc.)
            if filename == "qiskit_dataset.json" or re.match(r'\d+\.json$', filename):
                # Try to read the dataset to extract model info
                try:
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    
                    model_name = self._extract_model_from_dataset(dataset)
                    
                    # Use file modification time as timestamp
                    file_stat = os.stat(dataset_file)
                    timestamp = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y%m%d_%H%M%S")
                    
                    print(f"Extracted from dataset content: model='{model_name}', timestamp='{timestamp}'")
                    return model_name, timestamp
                    
                except Exception as e:
                    print(f"Could not read dataset file for metadata extraction: {e}")
            
            # Final fallback to current timestamp
            print(f"Warning: Could not extract model/timestamp from filename '{filename}', using current timestamp")
            return "unknown_model", datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_validation_results(self, summary_table: str) -> Tuple[str, str]:
        """Save validation results to files in dataset_validations directory."""
        try:
            # Create directory if it doesn't exist
            os.makedirs("dataset_validations", exist_ok=True)
            
            # Extract model and timestamp from dataset path
            model_name, timestamp = self.extract_model_and_timestamp_from_filename(self.dataset_path)
            
            # Generate filenames using extracted model and timestamp
            txt_filename = os.path.join("dataset_validations", f"{model_name}_{timestamp}.txt")
            json_filename = os.path.join("dataset_validations", f"{model_name}_{timestamp}.json")
            
            # Save text summary
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("DATASET VALIDATION RESULTS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Validation executed at: {datetime.now().isoformat()}\n")
                f.write(f"Dataset source: {self.dataset_path}\n")
                if self.is_directory:
                    f.write(f"Source type: Directory with {len(self.dataset_files)} JSON files\n")
                else:
                    f.write(f"Source type: Single file\n")
                f.write(f"Dataset model: {model_name}\n")
                f.write(f"Dataset timestamp: {timestamp}\n\n")
                f.write(summary_table)
                
                # Add detailed results
                f.write("\n\nDETAILED VALIDATION RESULTS\n")
                f.write("=" * 50 + "\n")
                for i, result in enumerate(self.results):
                    f.write(f"\n{i+1}. {result.entry_type.upper()} - Rank {result.preference_rank or 'N/A'}\n")
                    f.write(f"   Problem ID: {result.problem_id}\n")
                    f.write(f"   Executed: {'✓' if result.code_executed else '✗'}\n")
                    
                    if result.execution_error:
                        f.write(f"   Error: {result.execution_error}\n")
                    else:
                        f.write(f"   Expected: {result.expected_output}\n")
                        f.write(f"   Actual:   {result.actual_output}\n")
                        f.write(f"   Match: {'✓' if result.outputs_match else '✗'}\n")
                        if result.rmse_error:
                            f.write(f"   RMSE: {result.rmse_error:.6f}\n")
            
            # Separate solutions and common mistakes for JSON
            solutions = [r for r in self.results if r.entry_type == 'solution']
            mistakes = [r for r in self.results if r.entry_type == 'common_mistake']
            
            # Save JSON structured data
            json_data = {
                "metadata": {
                    "validation_executed_at": datetime.now().isoformat(),
                    "dataset_source": self.dataset_path,
                    "source_type": "directory" if self.is_directory else "single_file",
                    "dataset_files": [os.path.basename(f) for f in self.dataset_files] if self.is_directory else [os.path.basename(self.dataset_path)],
                    "dataset_model": model_name,
                    "dataset_timestamp": timestamp,
                    "total_entries": len(self.results)
                },
                "solutions_analysis": {
                    "total_solutions": len(solutions),
                    "successfully_executed": sum(1 for r in solutions if r.code_executed),
                    "correct_outputs": sum(1 for r in solutions if r.outputs_match),
                    "execution_success_rate": sum(1 for r in solutions if r.code_executed) / len(solutions) * 100 if solutions else 0,
                    "correctness_rate": sum(1 for r in solutions if r.outputs_match) / sum(1 for r in solutions if r.code_executed) * 100 if any(r.code_executed for r in solutions) else 0
                },
                "common_mistakes_analysis": {
                    "total_mistake_examples": len(mistakes),
                    "successfully_executed": sum(1 for r in mistakes if r.code_executed),
                    "demonstrates_mistake": sum(1 for r in mistakes if r.code_executed and not r.outputs_match),
                    "execution_success_rate": sum(1 for r in mistakes if r.code_executed) / len(mistakes) * 100 if mistakes else 0,
                    "mistake_demonstration_rate": sum(1 for r in mistakes if r.code_executed and not r.outputs_match) / sum(1 for r in mistakes if r.code_executed) * 100 if any(r.code_executed for r in mistakes) else 0
                },
                "detailed_results": {
                    "solutions": [asdict(result) for result in solutions],
                    "common_mistakes": [asdict(result) for result in mistakes]
                }
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Validation results saved to:")
            print(f"  Text: {txt_filename}")
            print(f"  JSON: {json_filename}")
            
            return txt_filename, json_filename
            
        except Exception as e:
            print(f"Error saving validation results: {e}")
            return None, None
    
    def print_detailed_results(self) -> None:
        """Print detailed results for each validation."""
        print("\n" + "="*80)
        print("DETAILED VALIDATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(self.results):
            print(f"\n{i+1}. {result.entry_type.upper()} - Rank {result.preference_rank or 'N/A'}")
            print(f"   Problem ID: {result.problem_id}")
            print(f"   Executed: {'✓' if result.code_executed else '✗'}")
            
            if result.execution_error:
                print(f"   Error: {result.execution_error}")
            else:
                print(f"   Expected: {result.expected_output}")
                print(f"   Actual:   {result.actual_output}")
                print(f"   Match: {'✓' if result.outputs_match else '✗'}")
                if result.rmse_error:
                    print(f"   RMSE: {result.rmse_error:.6f}")
    
    def run_validation(self) -> bool:
        """Run the complete validation process."""
        print("Starting Dataset Validation")
        print("="*50)
        
        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            return False
        
        if 'problems' in dataset:
            print(f"✓ Dataset collection loaded successfully")
            print(f"  Total problems: {len(dataset['problems'])}")
            if dataset['problems']:
                first_problem = dataset['problems'][0]
                print(f"  Sample problem: {first_problem.get('problem_id', 'unknown')}")
                print(f"  Sample difficulty: {first_problem.get('difficulty', 'unknown')}")
                print(f"  Sample category: {first_problem.get('category', 'unknown')}")
        else:
            print(f"✓ Dataset loaded successfully")
            print(f"  Problem ID: {dataset.get('problem_id', 'unknown')}")
            print(f"  Difficulty: {dataset.get('difficulty', 'unknown')}")
            print(f"  Category: {dataset.get('category', 'unknown')}")
        
        # Validate solutions
        self.validate_solutions(dataset)
        
        # Validate common mistakes
        self.validate_common_mistakes(dataset)
        
        # Generate and print summary
        summary_table = self.generate_summary_table()
        print("\n" + "="*80)
        print("VALIDATION SUMMARY TABLE")
        print("="*80)
        print(summary_table)
        
        # Print detailed results
        self.print_detailed_results()
        
        # Overall statistics by category
        solutions = [r for r in self.results if r.entry_type == 'solution']
        mistakes = [r for r in self.results if r.entry_type == 'common_mistake']
        
        stats_text = f"""
OVERALL VALIDATION STATISTICS
{"="*80}"""

        if solutions:
            total_solutions = len(solutions)
            executed_solutions = sum(1 for r in solutions if r.code_executed)
            correct_solutions = sum(1 for r in solutions if r.outputs_match)
            
            stats_text += f"""
SOLUTIONS:
  Total solutions: {total_solutions}
  Successfully executed: {executed_solutions}/{total_solutions} ({executed_solutions/total_solutions*100:.1f}%)"""
            if executed_solutions > 0:
                stats_text += f"""
  Correct outputs: {correct_solutions}/{executed_solutions} ({correct_solutions/executed_solutions*100:.1f}% of executed)"""
            else:
                stats_text += f"""
  Correct outputs: {correct_solutions}/{executed_solutions} (no successful executions)"""

        if mistakes:
            total_mistakes = len(mistakes)
            executed_mistakes = sum(1 for r in mistakes if r.code_executed)
            demonstrates_mistake = sum(1 for r in mistakes if r.code_executed and not r.outputs_match)
            
            stats_text += f"""

COMMON MISTAKES:
  Total mistake examples: {total_mistakes}
  Successfully executed: {executed_mistakes}/{total_mistakes} ({executed_mistakes/total_mistakes*100:.1f}%)"""
            if executed_mistakes > 0:
                stats_text += f"""
  Demonstrates mistake: {demonstrates_mistake}/{executed_mistakes} ({demonstrates_mistake/executed_mistakes*100:.1f}% of executed)"""
            else:
                stats_text += f"""
  Demonstrates mistake: {demonstrates_mistake}/{executed_mistakes} (no successful executions)"""
        
        stats_text += "\n" + "="*80
        print(stats_text)
        
        # Save results to files
        full_summary = summary_table + stats_text
        self.save_validation_results(full_summary)
        
        return True


def main():
    """Main entry point."""
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "qiskit_dataset.json"
    
    print(f"Dataset Validator - Processing: {dataset_path}")
    if os.path.isdir(dataset_path):
        print(f"Mode: Directory validation")
    else:
        print(f"Mode: Single file validation")
    print("=" * 60)
    
    validator = DatasetValidator(dataset_path)
    success = validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())