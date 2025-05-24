#!/usr/bin/env python3
"""
Dataset Visualization Tools

This module provides visualization capabilities for analyzing datasets,
including distribution plots, quality metrics, and validation results.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np


class DatasetVisualizer:
    """Create visualizations for dataset analysis."""
    
    def __init__(self):
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_dataset_overview(self, datasets: List[Dict[str, Any]], save_plots: bool = True) -> str:
        """Create overview visualizations for a collection of datasets."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract metadata
        difficulties = [d.get('difficulty', 'unknown') for d in datasets]
        categories = [d.get('category', 'unknown') for d in datasets]
        
        # Solution counts
        solution_counts = [len(d.get('solutions', [])) for d in datasets]
        mistake_counts = [len(d.get('common_mistakes', [])) for d in datasets]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dataset Overview - {len(datasets)} Problems', fontsize=16, fontweight='bold')
        
        # 1. Difficulty distribution
        difficulty_counts = Counter(difficulties)
        axes[0, 0].pie(difficulty_counts.values(), labels=difficulty_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Difficulty Distribution')
        
        # 2. Category distribution
        category_counts = Counter(categories)
        axes[0, 1].bar(category_counts.keys(), category_counts.values())
        axes[0, 1].set_title('Category Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Solutions per problem
        axes[1, 0].hist(solution_counts, bins=max(solution_counts) if solution_counts else 1, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Solutions per Problem')
        axes[1, 0].set_xlabel('Number of Solutions')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Common mistakes per problem
        axes[1, 1].hist(mistake_counts, bins=max(mistake_counts) if mistake_counts else 1, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Common Mistakes per Problem')
        axes[1, 1].set_xlabel('Number of Mistakes')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(self.viz_dir, f"dataset_overview_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Overview visualization saved: {filename}")
            return filename
        else:
            plt.show()
            return ""
    
    def visualize_quality_metrics(self, quality_scores: List[Dict[str, Any]], save_plots: bool = True) -> str:
        """Visualize quality metrics across datasets."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract scores
        completeness_scores = [q.get('completeness_score', 0) for q in quality_scores]
        code_validity_scores = [q.get('code_validity_score', 0) for q in quality_scores]
        educational_scores = [q.get('educational_value_score', 0) for q in quality_scores]
        consistency_scores = [q.get('consistency_score', 0) for q in quality_scores]
        overall_scores = [q.get('overall_score', 0) for q in quality_scores]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Quality Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall score distribution
        axes[0, 0].hist(overall_scores, bins=10, alpha=0.7, edgecolor='black', color='green')
        axes[0, 0].set_title('Overall Quality Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(overall_scores), color='red', linestyle='--', label=f'Mean: {np.mean(overall_scores):.2f}')
        axes[0, 0].legend()
        
        # 2. Score component comparison
        score_data = pd.DataFrame({
            'Completeness': completeness_scores,
            'Code Validity': code_validity_scores,
            'Educational Value': educational_scores,
            'Consistency': consistency_scores,
            'Overall': overall_scores
        })
        
        axes[0, 1].boxplot([completeness_scores, code_validity_scores, educational_scores, consistency_scores, overall_scores],
                          labels=['Complete', 'Code', 'Educational', 'Consistent', 'Overall'])
        axes[0, 1].set_title('Score Components Comparison')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Correlation heatmap
        correlation_matrix = score_data.corr()
        im = axes[0, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[0, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[0, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[0, 2].set_yticklabels(correlation_matrix.columns)
        axes[0, 2].set_title('Score Correlation Matrix')
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                axes[0, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white')
        
        # 4. Score trends (if problem IDs are sequential)
        axes[1, 0].plot(range(len(overall_scores)), overall_scores, 'o-', alpha=0.7)
        axes[1, 0].set_title('Quality Score Trends')
        axes[1, 0].set_xlabel('Dataset Index')
        axes[1, 0].set_ylabel('Overall Score')
        
        # 5. Grade distribution
        grades = []
        for score in overall_scores:
            if score >= 0.9:
                grades.append('A')
            elif score >= 0.8:
                grades.append('B')
            elif score >= 0.7:
                grades.append('C')
            elif score >= 0.6:
                grades.append('D')
            else:
                grades.append('F')
        
        grade_counts = Counter(grades)
        axes[1, 1].bar(grade_counts.keys(), grade_counts.values(), color=['green', 'lightgreen', 'yellow', 'orange', 'red'][:len(grade_counts)])
        axes[1, 1].set_title('Quality Grade Distribution')
        axes[1, 1].set_xlabel('Grade')
        axes[1, 1].set_ylabel('Count')
        
        # 6. Score statistics table
        stats_text = f"""Quality Statistics:
Mean Overall: {np.mean(overall_scores):.3f}
Std Dev: {np.std(overall_scores):.3f}
Min: {np.min(overall_scores):.3f}
Max: {np.max(overall_scores):.3f}

Component Means:
Completeness: {np.mean(completeness_scores):.3f}
Code Validity: {np.mean(code_validity_scores):.3f}
Educational: {np.mean(educational_scores):.3f}
Consistency: {np.mean(consistency_scores):.3f}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, fontsize=10,
                        verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Quality Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(self.viz_dir, f"quality_metrics_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Quality metrics visualization saved: {filename}")
            return filename
        else:
            plt.show()
            return ""
    
    def visualize_validation_results(self, validation_data: Dict[str, Any], save_plots: bool = True) -> str:
        """Visualize validation results and execution statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = validation_data.get('results', [])
        
        # Extract data
        execution_success = [r.get('code_executed', False) for r in results]
        output_matches = [r.get('outputs_match', False) for r in results if r.get('code_executed', False)]
        entry_types = [r.get('entry_type', 'unknown') for r in results]
        difficulties = [r.get('difficulty', 'unknown') for r in results]
        categories = [r.get('category', 'unknown') for r in results]
        rmse_errors = [r.get('rmse_error') for r in results if r.get('rmse_error') is not None]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Validation Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. Execution success rate
        success_counts = Counter(['Success' if s else 'Failed' for s in execution_success])
        colors = ['green', 'red']
        axes[0, 0].pie(success_counts.values(), labels=success_counts.keys(), autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Code Execution Success Rate')
        
        # 2. Output matching rate
        if output_matches:
            match_counts = Counter(['Match' if m else 'No Match' for m in output_matches])
            axes[0, 1].pie(match_counts.values(), labels=match_counts.keys(), autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            axes[0, 1].set_title('Output Matching Rate\n(Among Successful Executions)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Successful\nExecutions', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Output Matching Rate')
        
        # 3. Success by entry type
        type_success = {}
        for entry_type in set(entry_types):
            type_results = [execution_success[i] for i, t in enumerate(entry_types) if t == entry_type]
            type_success[entry_type] = sum(type_results) / len(type_results) if type_results else 0
        
        axes[0, 2].bar(type_success.keys(), [v * 100 for v in type_success.values()])
        axes[0, 2].set_title('Success Rate by Entry Type')
        axes[0, 2].set_ylabel('Success Rate (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Success by difficulty
        difficulty_success = {}
        for difficulty in set(difficulties):
            diff_results = [execution_success[i] for i, d in enumerate(difficulties) if d == difficulty]
            difficulty_success[difficulty] = sum(diff_results) / len(diff_results) if diff_results else 0
        
        axes[1, 0].bar(difficulty_success.keys(), [v * 100 for v in difficulty_success.values()], color='skyblue')
        axes[1, 0].set_title('Success Rate by Difficulty')
        axes[1, 0].set_ylabel('Success Rate (%)')
        
        # 5. RMSE error distribution
        if rmse_errors:
            axes[1, 1].hist(rmse_errors, bins=10, alpha=0.7, edgecolor='black', color='orange')
            axes[1, 1].set_title('RMSE Error Distribution')
            axes[1, 1].set_xlabel('RMSE Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(np.mean(rmse_errors), color='red', linestyle='--', label=f'Mean: {np.mean(rmse_errors):.3f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No RMSE\nErrors Available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('RMSE Error Distribution')
        
        # 6. Summary statistics
        total_entries = len(results)
        successful_executions = sum(execution_success)
        matching_outputs = sum(output_matches) if output_matches else 0
        
        summary_text = f"""Validation Summary:
Total Entries: {total_entries}
Successful Executions: {successful_executions} ({successful_executions/total_entries*100:.1f}%)
Output Matches: {matching_outputs} ({matching_outputs/successful_executions*100:.1f}% of successful)

By Entry Type:
{chr(10).join([f'{k}: {v*100:.1f}%' for k, v in type_success.items()])}

By Difficulty:
{chr(10).join([f'{k}: {v*100:.1f}%' for k, v in difficulty_success.items()])}

RMSE Stats:
Mean: {np.mean(rmse_errors):.3f if rmse_errors else 'N/A'}
Count: {len(rmse_errors)}"""
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, fontsize=9,
                        verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Validation Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(self.viz_dir, f"validation_results_{timestamp}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Validation results visualization saved: {filename}")
            return filename
        else:
            plt.show()
            return ""
    
    def create_comprehensive_report(self, datasets: List[Dict[str, Any]], 
                                  quality_scores: List[Dict[str, Any]] = None,
                                  validation_data: Dict[str, Any] = None,
                                  save_plots: bool = True) -> List[str]:
        """Create a comprehensive visual report."""
        print("Creating comprehensive visualization report...")
        
        saved_files = []
        
        # Dataset overview
        overview_file = self.visualize_dataset_overview(datasets, save_plots)
        if overview_file:
            saved_files.append(overview_file)
        
        # Quality metrics (if available)
        if quality_scores:
            quality_file = self.visualize_quality_metrics(quality_scores, save_plots)
            if quality_file:
                saved_files.append(quality_file)
        
        # Validation results (if available)
        if validation_data:
            validation_file = self.visualize_validation_results(validation_data, save_plots)
            if validation_file:
                saved_files.append(validation_file)
        
        print(f"Comprehensive report created with {len(saved_files)} visualizations")
        return saved_files


def main():
    """Main entry point for standalone visualization."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_visualizer.py <dataset_file.json|validation_file.json> [type]")
        print("Types: overview, quality, validation, comprehensive")
        return 1
    
    data_file = sys.argv[1]
    viz_type = sys.argv[2] if len(sys.argv) > 2 else "overview"
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        visualizer = DatasetVisualizer()
        
        if viz_type == "overview":
            # Assume it's a single dataset or collection
            datasets = data.get('datasets', [data]) if 'datasets' in data else [data]
            visualizer.visualize_dataset_overview(datasets)
        
        elif viz_type == "validation":
            visualizer.visualize_validation_results(data)
        
        elif viz_type == "comprehensive":
            datasets = data.get('datasets', [data]) if 'datasets' in data else [data]
            visualizer.create_comprehensive_report(datasets)
        
        else:
            print(f"Unknown visualization type: {viz_type}")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())