#!/usr/bin/env python3
"""
Dataset Export Utilities

This module provides functionality to export datasets to various formats
including CSV, JSONL, and custom training formats.
"""

import json
import csv
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class DatasetExporter:
    """Export datasets to various formats for different use cases."""
    
    def __init__(self):
        self.export_dir = "exports"
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_to_csv(self, dataset: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export dataset to CSV format for tabular analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_dir, f"dataset_export_{timestamp}.csv")
        
        # Flatten dataset for CSV export
        rows = []
        
        # Main dataset info
        base_row = {
            'problem_id': dataset.get('problem_id', ''),
            'prompt': dataset.get('prompt', ''),
            'difficulty': dataset.get('difficulty', ''),
            'category': dataset.get('category', ''),
            'learning_objectives': '; '.join(dataset.get('learning_objectives', [])),
            'prerequisites': '; '.join(dataset.get('prerequisites', [])),
        }
        
        # Add reasoning trace
        reasoning = dataset.get('reasoning_trace', {})
        for key, value in reasoning.items():
            base_row[f'reasoning_{key}'] = str(value)
        
        # Add solutions
        solutions = dataset.get('solutions', [])
        for i, solution in enumerate(solutions):
            row = base_row.copy()
            row.update({
                'entry_type': 'solution',
                'preference_rank': solution.get('preference_rank', ''),
                'code': solution.get('code', ''),
                'expected_output': solution.get('expected_output', ''),
                'output_interpretation': solution.get('output_interpretation', ''),
                'why_less_preferred': solution.get('why_less_preferred', '')
            })
            
            # Add code reasoning
            code_reasoning = solution.get('code_reasoning', {})
            for key, value in code_reasoning.items():
                row[f'code_reasoning_{key}'] = str(value)
            
            rows.append(row)
        
        # Add common mistakes
        mistakes = dataset.get('common_mistakes', [])
        for i, mistake in enumerate(mistakes):
            row = base_row.copy()
            row.update({
                'entry_type': 'common_mistake',
                'preference_rank': '',
                'code': mistake.get('incorrect_code', ''),
                'expected_output': 'Error expected',
                'output_interpretation': mistake.get('explanation', ''),
                'error_type': mistake.get('error_type', ''),
                'debugging_reasoning': mistake.get('debugging_reasoning', '')
            })
            rows.append(row)
        
        # Add validation tests
        tests = dataset.get('validation_tests', [])
        for i, test in enumerate(tests):
            row = base_row.copy()
            row.update({
                'entry_type': 'validation_test',
                'preference_rank': '',
                'code': test.get('test_code', ''),
                'expected_output': test.get('expected_test_result', ''),
                'output_interpretation': test.get('test_description', '')
            })
            rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Dataset exported to CSV: {filename}")
        
        return filename
    
    def export_to_jsonl(self, dataset: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export dataset to JSONL format for streaming/ML training."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_dir, f"dataset_export_{timestamp}.jsonl")
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Export solutions as separate JSONL entries
            solutions = dataset.get('solutions', [])
            for solution in solutions:
                entry = {
                    'problem_id': dataset.get('problem_id', ''),
                    'difficulty': dataset.get('difficulty', ''),
                    'category': dataset.get('category', ''),
                    'prompt': dataset.get('prompt', ''),
                    'entry_type': 'solution',
                    'preference_rank': solution.get('preference_rank', 0),
                    'code': solution.get('code', ''),
                    'expected_output': solution.get('expected_output', ''),
                    'reasoning': dataset.get('reasoning_trace', {}),
                    'code_reasoning': solution.get('code_reasoning', {})
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Export common mistakes
            mistakes = dataset.get('common_mistakes', [])
            for mistake in mistakes:
                entry = {
                    'problem_id': dataset.get('problem_id', ''),
                    'difficulty': dataset.get('difficulty', ''),
                    'category': dataset.get('category', ''),
                    'prompt': dataset.get('prompt', ''),
                    'entry_type': 'common_mistake',
                    'preference_rank': -1,  # Negative for mistakes
                    'code': mistake.get('incorrect_code', ''),
                    'expected_output': 'error',
                    'error_explanation': mistake.get('explanation', ''),
                    'error_type': mistake.get('error_type', '')
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Dataset exported to JSONL: {filename}")
        return filename
    
    def export_to_training_format(self, dataset: Dict[str, Any], format_type: str = "instruction_following", filename: Optional[str] = None) -> str:
        """Export dataset in formats suitable for fine-tuning."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_dir, f"training_data_{format_type}_{timestamp}.jsonl")
        
        with open(filename, 'w', encoding='utf-8') as f:
            if format_type == "instruction_following":
                self._export_instruction_following(dataset, f)
            elif format_type == "code_completion":
                self._export_code_completion(dataset, f)
            elif format_type == "preference_ranking":
                self._export_preference_ranking(dataset, f)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
        
        print(f"Training data exported: {filename}")
        return filename
    
    def _export_instruction_following(self, dataset: Dict[str, Any], file):
        """Export in instruction-following format for fine-tuning."""
        problem_desc = dataset.get('prompt', '')
        difficulty = dataset.get('difficulty', '')
        category = dataset.get('category', '')
        
        # Create instruction from problem description
        instruction = f"Create a {difficulty} level Qiskit solution for the following {category} problem: {problem_desc}"
        
        # Export each solution
        solutions = dataset.get('solutions', [])
        for solution in solutions:
            entry = {
                "instruction": instruction,
                "input": "",
                "output": solution.get('code', ''),
                "metadata": {
                    "problem_id": dataset.get('problem_id', ''),
                    "difficulty": difficulty,
                    "category": category,
                    "preference_rank": solution.get('preference_rank', 0),
                    "expected_output": solution.get('expected_output', '')
                }
            }
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def _export_code_completion(self, dataset: Dict[str, Any], file):
        """Export in code completion format."""
        solutions = dataset.get('solutions', [])
        for solution in solutions:
            code = solution.get('code', '')
            if code:
                # Split code into prompt and completion
                lines = code.strip().split('\n')
                if len(lines) > 5:
                    # Take first few lines as prompt
                    prompt_lines = lines[:len(lines)//2]
                    completion_lines = lines[len(lines)//2:]
                    
                    entry = {
                        "prompt": '\n'.join(prompt_lines),
                        "completion": '\n'.join(completion_lines),
                        "metadata": {
                            "problem_id": dataset.get('problem_id', ''),
                            "difficulty": dataset.get('difficulty', ''),
                            "category": dataset.get('category', ''),
                            "preference_rank": solution.get('preference_rank', 0)
                        }
                    }
                    file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def _export_preference_ranking(self, dataset: Dict[str, Any], file):
        """Export in preference ranking format for DPO/preference training."""
        solutions = dataset.get('solutions', [])
        if len(solutions) >= 2:
            # Create preference pairs
            for i in range(len(solutions)):
                for j in range(i + 1, len(solutions)):
                    sol1 = solutions[i]
                    sol2 = solutions[j]
                    
                    # Determine which is preferred based on rank
                    rank1 = sol1.get('preference_rank', float('inf'))
                    rank2 = sol2.get('preference_rank', float('inf'))
                    
                    if rank1 < rank2:  # Lower rank number = higher preference
                        chosen = sol1.get('code', '')
                        rejected = sol2.get('code', '')
                    else:
                        chosen = sol2.get('code', '')
                        rejected = sol1.get('code', '')
                    
                    entry = {
                        "prompt": f"Solve this {dataset.get('difficulty', '')} level quantum computing problem: {dataset.get('prompt', '')}",
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "problem_id": dataset.get('problem_id', ''),
                            "difficulty": dataset.get('difficulty', ''),
                            "category": dataset.get('category', '')
                        }
                    }
                    file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def export_collection_to_formats(self, datasets: List[Dict[str, Any]], formats: List[str] = None) -> Dict[str, str]:
        """Export a collection of datasets to multiple formats."""
        if formats is None:
            formats = ["csv", "jsonl", "instruction_following"]
        
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == "csv":
                # Combine all datasets for CSV
                all_rows = []
                for dataset in datasets:
                    filename = os.path.join(self.export_dir, f"collection_export_{timestamp}.csv")
                    self.export_to_csv(dataset, filename)
                exported_files[format_type] = filename
            
            elif format_type == "jsonl":
                filename = os.path.join(self.export_dir, f"collection_export_{timestamp}.jsonl")
                with open(filename, 'w', encoding='utf-8') as f:
                    for dataset in datasets:
                        # Write each dataset entry
                        f.write(json.dumps(dataset, ensure_ascii=False) + '\n')
                exported_files[format_type] = filename
            
            elif format_type in ["instruction_following", "code_completion", "preference_ranking"]:
                filename = os.path.join(self.export_dir, f"collection_training_{format_type}_{timestamp}.jsonl")
                with open(filename, 'w', encoding='utf-8') as f:
                    for dataset in datasets:
                        if format_type == "instruction_following":
                            self._export_instruction_following(dataset, f)
                        elif format_type == "code_completion":
                            self._export_code_completion(dataset, f)
                        elif format_type == "preference_ranking":
                            self._export_preference_ranking(dataset, f)
                exported_files[format_type] = filename
        
        print(f"Collection exported to {len(exported_files)} formats:")
        for format_type, filename in exported_files.items():
            print(f"  {format_type}: {filename}")
        
        return exported_files


def main():
    """Main entry point for standalone export."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_exporter.py <dataset_file.json> [format]")
        print("Formats: csv, jsonl, instruction_following, code_completion, preference_ranking")
        return 1
    
    dataset_file = sys.argv[1]
    export_format = sys.argv[2] if len(sys.argv) > 2 else "csv"
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        exporter = DatasetExporter()
        
        if export_format == "csv":
            exporter.export_to_csv(dataset)
        elif export_format == "jsonl":
            exporter.export_to_jsonl(dataset)
        elif export_format in ["instruction_following", "code_completion", "preference_ranking"]:
            exporter.export_to_training_format(dataset, export_format)
        else:
            print(f"Unknown format: {export_format}")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())