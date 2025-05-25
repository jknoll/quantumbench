#!/usr/bin/env python3
"""
Parameterized Validation System

This module extends the existing validation script to support parameterized testing
for implementation-independent correctness verification. It can validate both
legacy datasets (with expected_output) and new parameterized datasets.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import traceback
from datetime import datetime

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

# Import the parameter testing framework
from parameter_testing_framework import (
    ParameterizedValidator, ValidationConfig, QuantumParameterGenerator,
    QuantumCircuitExecutor, StatisticalComparator
)

# Import existing validation components
from validate_dataset import DatasetValidator, ValidationResult


@dataclass
class ParameterizedValidationResult:
    """Extended validation result with parameterized testing information."""
    problem_id: str
    difficulty: str
    category: str
    validation_type: str  # 'legacy' or 'parameterized'
    code_executed: bool
    success_rate: Optional[float]
    meets_threshold: bool
    total_parameter_tests: Optional[int]
    successful_parameter_tests: Optional[int]
    execution_error: Optional[str]
    details: Dict[str, Any]


class ParameterizedDatasetValidator:
    """Enhanced validator that supports both legacy and parameterized testing."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.legacy_validator = DatasetValidator(dataset_path)
        
        # Configure parameterized validation
        self.param_config = ValidationConfig(
            shots=1000,
            tolerance=0.05,
            min_success_rate=0.8,
            statistical_threshold=0.01
        )
        self.param_validator = ParameterizedValidator(self.param_config)
        self.results: List[ParameterizedValidationResult] = []
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate a dataset using the appropriate method (legacy or parameterized)."""
        print("Loading dataset...")
        
        try:
            dataset = self.legacy_validator.load_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {"error": str(e)}
        
        # Determine if this is a single problem or collection
        if 'problems' in dataset:
            return self._validate_collection(dataset)
        else:
            return self._validate_single_problem(dataset)
    
    def _validate_collection(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a collection of problems."""
        problems = dataset.get('problems', [])
        total_problems = len(problems)
        
        print(f"Validating collection with {total_problems} problems...")
        
        for i, problem in enumerate(tqdm(problems, desc="Validating problems")):
            result = self._validate_single_problem_data(problem, f"{i+1}/{total_problems}")
            self.results.append(result)
        
        return self._generate_collection_summary(dataset)
    
    def _validate_single_problem(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single problem."""
        print("Validating single problem...")
        result = self._validate_single_problem_data(dataset, "1/1")
        self.results.append(result)
        
        return self._generate_single_summary(dataset)
    
    def _validate_single_problem_data(self, problem_data: Dict[str, Any], progress: str) -> ParameterizedValidationResult:
        """Validate a single problem's data."""
        problem_id = problem_data.get('problem_id', 'unknown')
        difficulty = problem_data.get('difficulty', 'unknown')
        category = problem_data.get('category', 'unknown')
        
        print(f"[{progress}] Validating {problem_id}...")
        
        # Check if this is a parameterized dataset
        has_param_specs = 'parameter_specs' in problem_data
        has_test_cases = 'test_cases' in problem_data
        has_algorithm_type = 'algorithm_type' in problem_data
        
        if has_param_specs or has_test_cases or has_algorithm_type:
            return self._validate_parameterized(problem_data)
        else:
            return self._validate_legacy(problem_data)
    
    def _validate_parameterized(self, problem_data: Dict[str, Any]) -> ParameterizedValidationResult:
        """Validate using parameterized testing."""
        problem_id = problem_data.get('problem_id', 'unknown')
        difficulty = problem_data.get('difficulty', 'unknown')
        category = problem_data.get('category', 'unknown')
        
        try:
            # Extract solution code
            solution = problem_data.get('solution', {})
            if isinstance(solution, str):
                # Handle simplified format
                code = solution
            else:
                code = solution.get('code', '')
            
            if not code:
                return ParameterizedValidationResult(
                    problem_id=problem_id,
                    difficulty=difficulty,
                    category=category,
                    validation_type='parameterized',
                    code_executed=False,
                    success_rate=None,
                    meets_threshold=False,
                    total_parameter_tests=None,
                    successful_parameter_tests=None,
                    execution_error="No solution code found",
                    details={}
                )
            
            # Get algorithm type
            algorithm_type = problem_data.get('algorithm_type', 'custom')
            
            # For parameterized testing, we test the code against itself
            # In practice, this would test against a target LLM's generated code
            results = self.param_validator.validate_algorithm(
                reference_code=code,
                target_code=code,  # Testing against itself for now
                algorithm_type=algorithm_type,
                n_trials=5
            )
            
            return ParameterizedValidationResult(
                problem_id=problem_id,
                difficulty=difficulty,
                category=category,
                validation_type='parameterized',
                code_executed=True,
                success_rate=results['success_rate'],
                meets_threshold=results['meets_threshold'],
                total_parameter_tests=results['total_tests'],
                successful_parameter_tests=results['successful_tests'],
                execution_error=None,
                details=results
            )
            
        except Exception as e:
            return ParameterizedValidationResult(
                problem_id=problem_id,
                difficulty=difficulty,
                category=category,
                validation_type='parameterized',
                code_executed=False,
                success_rate=None,
                meets_threshold=False,
                total_parameter_tests=None,
                successful_parameter_tests=None,
                execution_error=str(e),
                details={'traceback': traceback.format_exc()}
            )
    
    def _validate_legacy(self, problem_data: Dict[str, Any]) -> ParameterizedValidationResult:
        """Validate using legacy expected output comparison."""
        problem_id = problem_data.get('problem_id', 'unknown')
        difficulty = problem_data.get('difficulty', 'unknown')  
        category = problem_data.get('category', 'unknown')
        
        try:
            # Use existing legacy validation logic
            validator = DatasetValidator("dummy")
            
            # Extract solution based on format
            solution = problem_data.get('solution', {})
            if isinstance(solution, str):
                # Simplified format
                code = solution
                expected_output = ""
            else:
                # Full format
                code = solution.get('code', '')
                expected_output = solution.get('expected_output', '')
            
            if not code:
                return ParameterizedValidationResult(
                    problem_id=problem_id,
                    difficulty=difficulty,
                    category=category,
                    validation_type='legacy',
                    code_executed=False,
                    success_rate=None,
                    meets_threshold=False,
                    total_parameter_tests=None,
                    successful_parameter_tests=None,
                    execution_error="No solution code found",
                    details={}
                )
            
            # Execute the code
            executed, actual_output, error = validator.safe_execute_code(code)
            
            if executed:
                # Compare outputs
                match = validator.compare_outputs(expected_output, actual_output)
                success_rate = 1.0 if match else 0.0
                meets_threshold = success_rate >= 0.8
            else:
                success_rate = 0.0
                meets_threshold = False
            
            return ParameterizedValidationResult(
                problem_id=problem_id,
                difficulty=difficulty,
                category=category,
                validation_type='legacy',
                code_executed=executed,
                success_rate=success_rate,
                meets_threshold=meets_threshold,
                total_parameter_tests=1,
                successful_parameter_tests=1 if meets_threshold else 0,
                execution_error=error if not executed else None,
                details={
                    'expected_output': expected_output,
                    'actual_output': actual_output,
                    'outputs_match': meets_threshold
                }
            )
            
        except Exception as e:
            return ParameterizedValidationResult(
                problem_id=problem_id,
                difficulty=difficulty,
                category=category,
                validation_type='legacy',
                code_executed=False,
                success_rate=None,
                meets_threshold=False,
                total_parameter_tests=None,
                successful_parameter_tests=None,
                execution_error=str(e),
                details={'traceback': traceback.format_exc()}
            )
    
    def _generate_collection_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for a collection of problems."""
        total = len(self.results)
        executed = sum(1 for r in self.results if r.code_executed)
        successful = sum(1 for r in self.results if r.meets_threshold)
        
        # Separate by validation type
        parameterized = [r for r in self.results if r.validation_type == 'parameterized']
        legacy = [r for r in self.results if r.validation_type == 'legacy']
        
        # Create summary table
        table_data = []
        for result in self.results:
            table_data.append([
                result.problem_id,
                result.difficulty,
                result.category,
                result.validation_type,
                '✓' if result.code_executed else '✗',
                f"{result.success_rate:.2f}" if result.success_rate is not None else "N/A",
                '✓' if result.meets_threshold else '✗',
                f"{result.successful_parameter_tests}/{result.total_parameter_tests}" if result.total_parameter_tests else "N/A"
            ])
        
        # Print summary table
        headers = ['Problem ID', 'Difficulty', 'Category', 'Type', 'Executed', 'Success Rate', 'Meets Threshold', 'Param Tests']
        print("\n" + "="*100)
        print("PARAMETERIZED VALIDATION SUMMARY")
        print("="*100)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"Total Problems: {total}")
        print(f"Executed Successfully: {executed}/{total} ({executed/total*100:.1f}%)")
        print(f"Meets Success Threshold: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"Parameterized Tests: {len(parameterized)}")
        print(f"Legacy Tests: {len(legacy)}")
        
        return {
            'total_problems': total,
            'executed_count': executed,
            'successful_count': successful,
            'execution_rate': executed/total if total > 0 else 0,
            'success_rate': successful/total if total > 0 else 0,
            'parameterized_count': len(parameterized),
            'legacy_count': len(legacy),
            'results': self.results
        }
    
    def _generate_single_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for a single problem."""
        result = self.results[0]
        
        print("\n" + "="*80)
        print("PARAMETERIZED VALIDATION RESULT")
        print("="*80)
        print(f"Problem ID: {result.problem_id}")
        print(f"Difficulty: {result.difficulty}")
        print(f"Category: {result.category}")
        print(f"Validation Type: {result.validation_type}")
        print(f"Code Executed: {'✓' if result.code_executed else '✗'}")
        
        if result.success_rate is not None:
            print(f"Success Rate: {result.success_rate:.2f}")
        
        print(f"Meets Threshold: {'✓' if result.meets_threshold else '✗'}")
        
        if result.total_parameter_tests:
            print(f"Parameter Tests: {result.successful_parameter_tests}/{result.total_parameter_tests}")
        
        if result.execution_error:
            print(f"Error: {result.execution_error}")
        
        return {
            'problem': result,
            'success': result.meets_threshold
        }
    
    def save_results(self, output_dir: str = "dataset_validations") -> str:
        """Save validation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model name and timestamp from dataset path
        if self.legacy_validator.is_directory:
            base_name = os.path.basename(self.dataset_path)
        else:
            base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            if base_name.startswith('qiskit_dataset_'):
                base_name = base_name[15:]  # Remove 'qiskit_dataset_' prefix
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = os.path.join(output_dir, f"parameterized_validation_{base_name}_{timestamp}.json")
        
        def serialize_details(details):
            """Convert details to JSON-serializable format."""
            if isinstance(details, dict):
                serialized = {}
                for key, value in details.items():
                    if hasattr(value, '__dict__'):
                        # Convert dataclass or object to dict
                        serialized[key] = str(value)
                    elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                        # Convert list of objects to list of strings
                        serialized[key] = [str(item) for item in value]
                    else:
                        serialized[key] = value
                return serialized
            else:
                return str(details)
        
        json_data = {
            'validation_timestamp': timestamp,
            'dataset_path': self.dataset_path,
            'validation_config': {
                'shots': self.param_config.shots,
                'tolerance': self.param_config.tolerance,
                'min_success_rate': self.param_config.min_success_rate
            },
            'results': [
                {
                    'problem_id': r.problem_id,
                    'difficulty': r.difficulty,
                    'category': r.category,
                    'validation_type': r.validation_type,
                    'code_executed': r.code_executed,
                    'success_rate': r.success_rate,
                    'meets_threshold': r.meets_threshold,
                    'total_parameter_tests': r.total_parameter_tests,
                    'successful_parameter_tests': r.successful_parameter_tests,
                    'execution_error': r.execution_error,
                    'details': serialize_details(r.details)
                }
                for r in self.results
            ]
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Validation results saved to: {json_filename}")
        return json_filename


def main():
    """Main entry point for parameterized validation."""
    if len(sys.argv) != 2:
        print("Usage: python parameterized_validation.py <dataset_file_or_directory>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print("Parameterized Dataset Validation")
    print("="*50)
    print(f"Dataset: {dataset_path}")
    print(f"Validation method: Automatic (legacy or parameterized)")
    
    # Create validator and run validation
    validator = ParameterizedDatasetValidator(dataset_path)
    results = validator.validate_dataset()
    
    # Save results
    validator.save_results()
    
    print("\n✅ Parameterized validation complete!")


if __name__ == "__main__":
    main()