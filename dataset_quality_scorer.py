#!/usr/bin/env python3
"""
Dataset Quality Scoring System

This module provides functionality to assess the quality of generated datasets
based on various metrics including completeness, code validity, and educational value.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class QualityMetrics:
    """Container for dataset quality metrics."""
    completeness_score: float
    code_validity_score: float
    educational_value_score: float
    consistency_score: float
    overall_score: float
    details: Dict[str, Any]


class DatasetQualityScorer:
    """Evaluates dataset quality based on multiple criteria."""
    
    def __init__(self):
        self.required_fields = [
            'problem_id', 'prompt', 'difficulty', 'category',
            'learning_objectives', 'prerequisites', 'reasoning_trace', 
            'solutions', 'common_mistakes', 'validation_tests', 'extensions'
        ]
        
        self.reasoning_fields = [
            'problem_analysis', 'quantum_physics_reasoning', 
            'algorithm_choice', 'parameter_selection', 'implementation_strategy'
        ]
        
        self.solution_fields = [
            'preference_rank', 'code', 'code_reasoning', 
            'expected_output', 'output_interpretation'
        ]
    
    def score_completeness(self, dataset: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score dataset completeness based on required fields."""
        details = {}
        
        # Check main fields
        missing_fields = [field for field in self.required_fields if field not in dataset]
        main_score = (len(self.required_fields) - len(missing_fields)) / len(self.required_fields)
        details['missing_main_fields'] = missing_fields
        
        # Check reasoning trace completeness
        reasoning_score = 0
        if 'reasoning_trace' in dataset:
            reasoning = dataset['reasoning_trace']
            present_reasoning = [field for field in self.reasoning_fields if field in reasoning and reasoning[field]]
            reasoning_score = len(present_reasoning) / len(self.reasoning_fields)
            details['missing_reasoning_fields'] = [f for f in self.reasoning_fields if f not in reasoning or not reasoning[f]]
        
        # Check solutions completeness
        solutions_score = 0
        if 'solutions' in dataset:
            solutions = dataset['solutions']
            if solutions:
                total_solution_score = 0
                for i, solution in enumerate(solutions):
                    present_fields = [field for field in self.solution_fields if field in solution and solution[field]]
                    solution_score = len(present_fields) / len(self.solution_fields)
                    total_solution_score += solution_score
                    details[f'solution_{i+1}_missing_fields'] = [f for f in self.solution_fields if f not in solution or not solution[f]]
                solutions_score = total_solution_score / len(solutions)
        
        # Weighted average
        completeness_score = (main_score * 0.5 + reasoning_score * 0.3 + solutions_score * 0.2)
        details['main_score'] = main_score
        details['reasoning_score'] = reasoning_score
        details['solutions_score'] = solutions_score
        
        return completeness_score, details
    
    def score_code_validity(self, dataset: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score code validity based on syntax and structure."""
        details = {}
        total_score = 0
        code_count = 0
        
        # Check solutions code
        if 'solutions' in dataset:
            for i, solution in enumerate(dataset['solutions']):
                if 'code' in solution:
                    code = solution['code']
                    validity_score = self._analyze_code_validity(code)
                    total_score += validity_score
                    code_count += 1
                    details[f'solution_{i+1}_code_score'] = validity_score
        
        # Check common mistakes code
        if 'common_mistakes' in dataset:
            for i, mistake in enumerate(dataset['common_mistakes']):
                if 'incorrect_code' in mistake:
                    code = mistake['incorrect_code']
                    # For mistakes, we expect them to be potentially invalid, so score differently
                    validity_score = self._analyze_code_validity(code, expect_valid=False)
                    total_score += validity_score
                    code_count += 1
                    details[f'mistake_{i+1}_code_score'] = validity_score
        
        # Check validation tests code
        if 'validation_tests' in dataset:
            for i, test in enumerate(dataset['validation_tests']):
                if 'test_code' in test:
                    code = test['test_code']
                    validity_score = self._analyze_code_validity(code)
                    total_score += validity_score
                    code_count += 1
                    details[f'test_{i+1}_code_score'] = validity_score
        
        overall_code_score = total_score / code_count if code_count > 0 else 0
        details['total_code_blocks'] = code_count
        
        return overall_code_score, details
    
    def _analyze_code_validity(self, code: str, expect_valid: bool = True) -> float:
        """Analyze individual code block validity."""
        score = 0.0
        
        # Basic syntax checks
        try:
            # Try to parse as Python (basic syntax check)
            compile(code, '<string>', 'exec')
            score += 0.3
        except SyntaxError:
            if not expect_valid:
                score += 0.2  # Mistakes might have syntax errors
        
        # Check for Qiskit imports/usage
        qiskit_patterns = [
            r'from qiskit import|import qiskit',
            r'QuantumCircuit|QuantumRegister|ClassicalRegister',
            r'\.h\(|\.x\(|\.cx\(|\.measure\(',
            r'Aer\.|execute\('
        ]
        
        qiskit_score = sum(1 for pattern in qiskit_patterns if re.search(pattern, code)) / len(qiskit_patterns)
        score += qiskit_score * 0.4
        
        # Check for proper structure (functions, variables)
        if 'def ' in code:
            score += 0.1
        if re.search(r'circuit\s*=|qr\s*=|cr\s*=', code):
            score += 0.1
        if 'return' in code or 'result' in code or 'counts' in code:
            score += 0.1
            
        return min(score, 1.0)
    
    def score_educational_value(self, dataset: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score educational value based on explanations and reasoning."""
        details = {}
        
        # Check learning objectives quality
        objectives_score = 0
        if 'learning_objectives' in dataset:
            objectives = dataset['learning_objectives']
            if isinstance(objectives, list) and len(objectives) >= 3:
                objectives_score = min(len(objectives) / 5, 1.0)  # Optimal around 5 objectives
        details['objectives_count'] = len(dataset.get('learning_objectives', []))
        
        # Check reasoning depth
        reasoning_depth_score = 0
        if 'reasoning_trace' in dataset:
            reasoning = dataset['reasoning_trace']
            total_length = sum(len(str(reasoning.get(field, ''))) for field in self.reasoning_fields)
            reasoning_depth_score = min(total_length / 1000, 1.0)  # Expect around 1000 chars of reasoning
        details['reasoning_total_length'] = total_length if 'reasoning_trace' in dataset else 0
        
        # Check code explanations
        explanation_score = 0
        explanation_count = 0
        if 'solutions' in dataset:
            for solution in dataset['solutions']:
                if 'code_reasoning' in solution:
                    reasoning_length = len(str(solution['code_reasoning']))
                    explanation_score += min(reasoning_length / 200, 1.0)  # Expect around 200 chars per explanation
                    explanation_count += 1
        
        if explanation_count > 0:
            explanation_score /= explanation_count
        details['average_explanation_length'] = explanation_score * 200 if explanation_count > 0 else 0
        
        # Check common mistakes explanations
        mistakes_score = 0
        if 'common_mistakes' in dataset:
            mistakes = dataset['common_mistakes']
            if len(mistakes) > 0:
                total_explanation_length = sum(len(mistake.get('explanation', '') + mistake.get('debugging_reasoning', '')) for mistake in mistakes)
                mistakes_score = min(total_explanation_length / (len(mistakes) * 150), 1.0)
        details['mistakes_explanation_quality'] = mistakes_score
        
        # Weighted average
        educational_score = (objectives_score * 0.2 + reasoning_depth_score * 0.4 + 
                           explanation_score * 0.3 + mistakes_score * 0.1)
        
        return educational_score, details
    
    def score_consistency(self, dataset: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score consistency across the dataset."""
        details = {}
        
        # Check difficulty consistency
        stated_difficulty = dataset.get('difficulty', '')
        consistency_score = 0
        
        # Check if learning objectives match stated difficulty
        objectives = dataset.get('learning_objectives', [])
        beginner_keywords = ['basic', 'introduction', 'simple', 'fundamental']
        advanced_keywords = ['complex', 'advanced', 'optimization', 'error correction']
        
        if stated_difficulty == 'beginner':
            beginner_matches = sum(1 for obj in objectives if any(kw in obj.lower() for kw in beginner_keywords))
            consistency_score += min(beginner_matches / len(objectives), 1.0) * 0.3 if objectives else 0
        elif stated_difficulty == 'advanced':
            advanced_matches = sum(1 for obj in objectives if any(kw in obj.lower() for kw in advanced_keywords))
            consistency_score += min(advanced_matches / len(objectives), 1.0) * 0.3 if objectives else 0
        else:  # intermediate
            consistency_score += 0.3  # Intermediate is neutral
        
        # Check category consistency
        stated_category = dataset.get('category', '')
        category_keywords = {
            'quantum_algorithms': ['algorithm', 'grover', 'shor', 'search', 'factoring'],
            'circuit_construction': ['circuit', 'gate', 'qubit', 'construction'],
            'optimization': ['optimization', 'transpilation', 'compile'],
            'error_mitigation': ['error', 'noise', 'mitigation', 'correction'],
            'hardware_interaction': ['hardware', 'backend', 'device', 'ibm'],
            'visualization': ['plot', 'visualize', 'draw', 'display']
        }
        
        if stated_category in category_keywords:
            keywords = category_keywords[stated_category]
            description = dataset.get('prompt', '').lower()
            keyword_matches = sum(1 for kw in keywords if kw in description)
            consistency_score += min(keyword_matches / len(keywords), 1.0) * 0.4
        
        # Check solution preference ranking consistency
        ranking_score = 0
        if 'solutions' in dataset:
            solutions = dataset['solutions']
            ranks = [sol.get('preference_rank') for sol in solutions if 'preference_rank' in sol]
            if ranks:
                expected_ranks = list(range(1, len(ranks) + 1))
                sorted_ranks = sorted(ranks)
                ranking_score = 1.0 if sorted_ranks == expected_ranks else 0.5
        
        consistency_score += ranking_score * 0.3
        
        details['difficulty_consistency'] = consistency_score
        details['stated_difficulty'] = stated_difficulty
        details['stated_category'] = stated_category
        
        return consistency_score, details
    
    def score_dataset(self, dataset: Dict[str, Any]) -> QualityMetrics:
        """Generate comprehensive quality score for a dataset."""
        completeness_score, completeness_details = self.score_completeness(dataset)
        code_validity_score, code_details = self.score_code_validity(dataset)
        educational_score, educational_details = self.score_educational_value(dataset)
        consistency_score, consistency_details = self.score_consistency(dataset)
        
        # Calculate weighted overall score
        overall_score = (
            completeness_score * 0.3 +
            code_validity_score * 0.25 +
            educational_score * 0.25 +
            consistency_score * 0.2
        )
        
        details = {
            'completeness': completeness_details,
            'code_validity': code_details,
            'educational_value': educational_details,
            'consistency': consistency_details
        }
        
        return QualityMetrics(
            completeness_score=completeness_score,
            code_validity_score=code_validity_score,
            educational_value_score=educational_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            details=details
        )
    
    def generate_quality_report(self, dataset: Dict[str, Any], save_to_file: bool = True) -> str:
        """Generate a comprehensive quality report."""
        metrics = self.score_dataset(dataset)
        
        report = f"""
DATASET QUALITY REPORT
{'='*50}
Generated at: {datetime.now().isoformat()}
Problem ID: {dataset.get('problem_id', 'Unknown')}

OVERALL SCORE: {metrics.overall_score:.2f}/1.00
{'='*50}

COMPONENT SCORES:
- Completeness:     {metrics.completeness_score:.2f}/1.00
- Code Validity:    {metrics.code_validity_score:.2f}/1.00
- Educational Value: {metrics.educational_value_score:.2f}/1.00
- Consistency:      {metrics.consistency_score:.2f}/1.00

DETAILED ANALYSIS:
{'-'*50}

Completeness:
- Main fields score: {metrics.details['completeness']['main_score']:.2f}
- Reasoning score: {metrics.details['completeness']['reasoning_score']:.2f}
- Solutions score: {metrics.details['completeness']['solutions_score']:.2f}
- Missing fields: {metrics.details['completeness']['missing_main_fields']}

Code Validity:
- Total code blocks: {metrics.details['code_validity']['total_code_blocks']}
- Average validity score: {metrics.code_validity_score:.2f}

Educational Value:
- Learning objectives count: {metrics.details['educational_value']['objectives_count']}
- Reasoning total length: {metrics.details['educational_value']['reasoning_total_length']} chars
- Average explanation length: {metrics.details['educational_value']['average_explanation_length']:.0f} chars

Consistency:
- Difficulty: {metrics.details['consistency']['stated_difficulty']}
- Category: {metrics.details['consistency']['stated_category']}
- Consistency score: {metrics.consistency_score:.2f}

QUALITY GRADE: {self._get_quality_grade(metrics.overall_score)}
"""
        
        if save_to_file:
            os.makedirs("dataset_validations", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_validations/quality_report_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"Quality report saved to: {filename}")
        
        return report
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Good)"
        elif score >= 0.7:
            return "C (Satisfactory)"
        elif score >= 0.6:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"


def main():
    """Main entry point for standalone quality scoring."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_quality_scorer.py <dataset_file.json>")
        return 1
    
    dataset_file = sys.argv[1]
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        scorer = DatasetQualityScorer()
        report = scorer.generate_quality_report(dataset)
        print(report)
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())