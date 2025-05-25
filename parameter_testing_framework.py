#!/usr/bin/env python3
"""
Parameter Testing Framework for Quantum Circuit Validation

This module implements a parameterized testing system that allows for
implementation-independent validation of quantum circuits. It supports
testing any LLM-generated code against reference implementations using
random parameter generation and statistical output comparison.
"""

import json
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import re
import sys
from io import StringIO
import contextlib
import traceback
from collections import Counter

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import scipy.stats as stats


@dataclass
class ParameterSpec:
    """Specification for a parameter that can be varied in testing."""
    name: str
    type: str  # 'integer', 'float', 'discrete', 'boolean'
    range: Union[Tuple[int, int], Tuple[float, float], List[Any]]
    description: str


@dataclass
class TestCase:
    """A single test case with specific parameter values."""
    input_params: Dict[str, Any]
    expected_properties: Dict[str, Any]
    expected_outputs: Dict[str, Any]


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""
    shots: int = 1000
    tolerance: float = 0.05
    min_success_rate: float = 0.8
    statistical_threshold: float = 0.01


@dataclass
class ParameterTestResult:
    """Result of testing with a specific parameter set."""
    test_case: TestCase
    reference_output: Dict[str, Any]
    target_output: Dict[str, Any]
    success: bool
    details: Dict[str, Any]


class QuantumParameterGenerator:
    """Generates test parameters for different quantum algorithm types."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_grover_params(self, n_trials: int = 5) -> List[Dict[str, Any]]:
        """Generate parameters for Grover's algorithm testing."""
        params = []
        
        for _ in range(n_trials):
            n_qubits = random.randint(2, 4)  # 2-4 qubits for manageable testing
            total_states = 2 ** n_qubits
            
            # Generate marked states (1-3 marked states)
            num_marked = random.randint(1, min(3, total_states // 2))
            marked_states = random.sample(range(total_states), num_marked)
            
            # Calculate optimal iterations
            optimal_iterations = int(np.pi * np.sqrt(total_states / num_marked) / 4)
            
            params.append({
                'n_qubits': n_qubits,
                'marked_states': marked_states,
                'optimal_iterations': optimal_iterations,
                'total_states': total_states
            })
        
        return params
    
    def generate_bell_state_params(self, n_trials: int = 4) -> List[Dict[str, Any]]:
        """Generate parameters for Bell state testing."""
        bell_states = [
            {'bell_type': 'phi_plus'},
            {'bell_type': 'phi_minus'},
            {'bell_type': 'psi_plus'},
            {'bell_type': 'psi_minus'}
        ]
        
        return bell_states[:n_trials]
    
    def generate_qft_params(self, n_trials: int = 5) -> List[Dict[str, Any]]:
        """Generate parameters for QFT testing."""
        params = []
        
        for _ in range(n_trials):
            n_qubits = random.randint(2, 4)
            # Generate random input state
            input_state = random.randint(0, 2**n_qubits - 1)
            
            params.append({
                'n_qubits': n_qubits,
                'input_state': input_state,
                'input_binary': format(input_state, f'0{n_qubits}b')
            })
        
        return params


class QuantumCircuitExecutor:
    """Executes quantum circuits with parameters and captures outputs."""
    
    def __init__(self, shots: int = 1000):
        self.shots = shots
        self.simulator = AerSimulator()
    
    def execute_code(self, code: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum code with given parameters and return results."""
        try:
            # Create a modified execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'np': np,
                'QuantumCircuit': QuantumCircuit,
                'QuantumRegister': QuantumRegister,
                'ClassicalRegister': ClassicalRegister,
                'AerSimulator': AerSimulator,
                'transpile': transpile,
                'params': params,  # Make parameters available to the code
                'shots': self.shots,
                'print': lambda *args: None,  # Suppress print statements
            }
            
            # Capture stdout
            captured_output = StringIO()
            
            with contextlib.redirect_stdout(captured_output):
                exec(code, exec_globals)
            
            # Extract results from the execution environment
            results = {}
            
            # Look for common result variables
            if 'counts' in exec_globals:
                results['counts'] = dict(exec_globals['counts'])
            if 'result' in exec_globals:
                results['result'] = exec_globals['result']
            if 'circuit' in exec_globals:
                circuit = exec_globals['circuit']
                results['circuit_depth'] = circuit.depth()
                results['gate_count'] = circuit.count_ops()
            
            results['stdout'] = captured_output.getvalue()
            results['success'] = True
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


class StatisticalComparator:
    """Compares quantum measurement outputs using statistical methods."""
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
    
    def compare_measurement_distributions(self, 
                                        ref_counts: Dict[str, int], 
                                        target_counts: Dict[str, int],
                                        total_shots: int) -> Dict[str, Any]:
        """Compare two measurement count distributions statistically."""
        
        # Convert to probability distributions
        ref_probs = {state: count/total_shots for state, count in ref_counts.items()}
        target_probs = {state: count/total_shots for state, count in target_counts.items()}
        
        # Get all states
        all_states = set(ref_probs.keys()) | set(target_probs.keys())
        
        # Calculate chi-squared test
        observed = []
        expected = []
        
        for state in sorted(all_states):
            ref_prob = ref_probs.get(state, 0)
            target_prob = target_probs.get(state, 0)
            
            observed.append(target_counts.get(state, 0))
            expected.append(ref_prob * total_shots)
        
        # Perform chi-squared test
        if sum(expected) > 0:
            chi2_stat, p_value = stats.chisquare(observed, expected)
        else:
            chi2_stat, p_value = float('inf'), 0.0
        
        # Calculate total variation distance
        tv_distance = 0.5 * sum(abs(ref_probs.get(state, 0) - target_probs.get(state, 0)) 
                               for state in all_states)
        
        # Determine if distributions are statistically similar
        similar = p_value > 0.05 and tv_distance < self.tolerance
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'tv_distance': tv_distance,
            'similar': similar,
            'ref_probs': ref_probs,
            'target_probs': target_probs
        }
    
    def compare_circuit_properties(self, 
                                 ref_props: Dict[str, Any], 
                                 target_props: Dict[str, Any]) -> Dict[str, Any]:
        """Compare circuit properties like depth and gate count."""
        
        comparison = {}
        
        for prop in ['circuit_depth', 'gate_count']:
            if prop in ref_props and prop in target_props:
                ref_val = ref_props[prop]
                target_val = target_props[prop]
                
                if isinstance(ref_val, dict) and isinstance(target_val, dict):
                    # Compare gate count dictionaries
                    ref_total = sum(ref_val.values())
                    target_total = sum(target_val.values())
                    comparison[f'{prop}_total'] = {
                        'reference': ref_total,
                        'target': target_total,
                        'ratio': target_total / max(ref_total, 1),
                        'similar': abs(target_total - ref_total) <= max(5, ref_total * 0.2)
                    }
                else:
                    # Compare scalar values
                    comparison[prop] = {
                        'reference': ref_val,
                        'target': target_val,
                        'ratio': target_val / max(ref_val, 1),
                        'similar': abs(target_val - ref_val) <= max(2, ref_val * 0.2)
                    }
        
        return comparison


class ParameterizedValidator:
    """Main class for parameterized validation of quantum circuits."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.param_generator = QuantumParameterGenerator()
        self.executor = QuantumCircuitExecutor(shots=self.config.shots)
        self.comparator = StatisticalComparator(tolerance=self.config.tolerance)
    
    def validate_algorithm(self, 
                          reference_code: str,
                          target_code: str,
                          algorithm_type: str,
                          n_trials: int = 5) -> Dict[str, Any]:
        """Validate a target implementation against a reference using parameters."""
        
        # Generate test parameters based on algorithm type
        if algorithm_type.lower() == 'grover':
            param_sets = self.param_generator.generate_grover_params(n_trials)
        elif algorithm_type.lower() == 'bell_state':
            param_sets = self.param_generator.generate_bell_state_params(n_trials)
        elif algorithm_type.lower() == 'qft':
            param_sets = self.param_generator.generate_qft_params(n_trials)
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        results = []
        success_count = 0
        
        for i, params in enumerate(param_sets):
            print(f"Testing parameter set {i+1}/{len(param_sets)}: {params}")
            
            # Execute reference implementation
            ref_result = self.executor.execute_code(reference_code, params)
            
            # Execute target implementation
            target_result = self.executor.execute_code(target_code, params)
            
            # Compare results
            if ref_result['success'] and target_result['success']:
                comparison = self._compare_results(ref_result, target_result)
                success = comparison['overall_success']
                if success:
                    success_count += 1
            else:
                comparison = {
                    'overall_success': False,
                    'error': 'Execution failed',
                    'ref_error': ref_result.get('error'),
                    'target_error': target_result.get('error')
                }
                success = False
            
            test_result = ParameterTestResult(
                test_case=TestCase(
                    input_params=params,
                    expected_properties={},
                    expected_outputs={}
                ),
                reference_output=ref_result,
                target_output=target_result,
                success=success,
                details=comparison
            )
            
            results.append(test_result)
        
        # Calculate overall metrics
        success_rate = success_count / len(param_sets)
        
        return {
            'algorithm_type': algorithm_type,
            'total_tests': len(param_sets),
            'successful_tests': success_count,
            'success_rate': success_rate,
            'meets_threshold': success_rate >= self.config.min_success_rate,
            'individual_results': results,
            'summary': self._generate_summary(results)
        }
    
    def _compare_results(self, ref_result: Dict[str, Any], target_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare reference and target results."""
        comparison = {'overall_success': True}
        
        # Compare measurement distributions if available
        if 'counts' in ref_result and 'counts' in target_result:
            dist_comparison = self.comparator.compare_measurement_distributions(
                ref_result['counts'], 
                target_result['counts'],
                self.config.shots
            )
            comparison['distribution'] = dist_comparison
            if not dist_comparison['similar']:
                comparison['overall_success'] = False
        
        # Compare circuit properties if available
        prop_comparison = self.comparator.compare_circuit_properties(
            ref_result, target_result
        )
        if prop_comparison:
            comparison['properties'] = prop_comparison
            # Check if any property comparison failed significantly
            for prop, details in prop_comparison.items():
                if isinstance(details, dict) and 'similar' in details:
                    if not details['similar']:
                        comparison['overall_success'] = False
        
        return comparison
    
    def _generate_summary(self, results: List[ParameterTestResult]) -> Dict[str, Any]:
        """Generate a summary of test results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        
        # Analyze failure reasons
        failure_reasons = []
        for result in results:
            if not result.success:
                if 'error' in result.details:
                    failure_reasons.append(result.details['error'])
                elif 'distribution' in result.details and not result.details['distribution']['similar']:
                    failure_reasons.append('Statistical distribution mismatch')
                elif 'properties' in result.details:
                    failure_reasons.append('Circuit property mismatch')
        
        return {
            'total_tests': total,
            'successful_tests': successful,
            'failed_tests': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'failure_reasons': Counter(failure_reasons)
        }


def create_parameterized_dataset_schema() -> Dict[str, Any]:
    """Define the schema for parameterized datasets."""
    return {
        "problem_id": "string",
        "prompt": "string - Problem description that includes parameter specifications",
        "difficulty": "string",
        "category": "string", 
        "learning_objectives": ["array of strings"],
        "prerequisites": ["array of strings"],
        "reasoning_trace": "string",
        
        # New parameterized testing fields
        "parameter_specs": [
            {
                "name": "string",
                "type": "integer|float|discrete|boolean", 
                "range": "tuple or list of valid values",
                "description": "string"
            }
        ],
        
        "test_cases": [
            {
                "input_params": {"param_name": "value"},
                "expected_properties": {
                    "circuit_depth": {"max": 50},
                    "success_probability": {"min": 0.8}
                },
                "expected_outputs": {
                    "measurement_distribution": {"target_state": {"min": 800, "max": 1000}},
                    "verification": "statistical_comparison"
                }
            }
        ],
        
        "algorithm_type": "string - grover|bell_state|qft|custom",
        "evaluation_method": "string - statistical|exact|custom",
        
        "solution": {
            "code": "string - Reference implementation that accepts params",
            "output_interpretation": "string"
        },
        
        "extensions": ["array of strings"]
    }


if __name__ == "__main__":
    # Example usage
    config = ValidationConfig(shots=1000, tolerance=0.05)
    validator = ParameterizedValidator(config)
    
    # Example reference code for Grover's algorithm
    reference_code = """
# Parameterized Grover's algorithm
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def create_grover_circuit(n_qubits, marked_states, iterations):
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    circuit.h(qr)
    
    # Grover iterations
    for _ in range(iterations):
        # Oracle - mark specified states
        for state in marked_states:
            # Convert state to binary and apply oracle
            binary = format(state, f'0{n_qubits}b')
            for i, bit in enumerate(binary):
                if bit == '0':
                    circuit.x(qr[i])
            
            # Multi-controlled Z
            if n_qubits == 1:
                circuit.z(qr[0])
            elif n_qubits == 2:
                circuit.cz(qr[0], qr[1])
            else:
                circuit.mcz(list(range(n_qubits-1)), n_qubits-1, qr)
            
            # Restore
            for i, bit in enumerate(binary):
                if bit == '0':
                    circuit.x(qr[i])
        
        # Diffusion operator
        circuit.h(qr)
        circuit.x(qr)
        if n_qubits == 1:
            circuit.z(qr[0])
        elif n_qubits == 2:
            circuit.cz(qr[0], qr[1])
        else:
            circuit.mcz(list(range(n_qubits-1)), n_qubits-1, qr)
        circuit.x(qr)
        circuit.h(qr)
    
    circuit.measure(qr, cr)
    return circuit

# Use the parameters provided
n_qubits = params['n_qubits']
marked_states = params['marked_states']
iterations = params['optimal_iterations']

circuit = create_grover_circuit(n_qubits, marked_states, iterations)
simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()
"""
    
    # Test the framework
    print("Testing Parameterized Validation Framework")
    print("=" * 50)
    
    results = validator.validate_algorithm(
        reference_code=reference_code,
        target_code=reference_code,  # Testing against itself
        algorithm_type='grover',
        n_trials=3
    )
    
    print(f"Success rate: {results['success_rate']:.2f}")
    print(f"Meets threshold: {results['meets_threshold']}")
    print(f"Summary: {results['summary']}")