#!/usr/bin/env python3
"""
Clean Python Files

This script fixes the duplicate header issue in the extracted Python files.
"""

import os
import re

def clean_python_file(file_path: str):
    """Clean a Python file by removing duplicate headers and fixing formatting."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the problem ID from filename
    filename = os.path.basename(file_path)
    problem_id = filename.replace('.py', '')
    
    # Split into lines and find the real start of code
    lines = content.split('\n')
    
    # Find where the actual Python code starts (after parameter definitions)
    code_start = 0
    found_params = False
    found_shots = False
    
    for i, line in enumerate(lines):
        if 'params = {' in line:
            found_params = True
        elif 'shots = ' in line and found_params:
            found_shots = True
        elif found_shots and (line.strip().startswith('import') or line.strip().startswith('from')):
            code_start = i
            break
    
    # Reconstruct the file
    header_lines = []
    param_lines = []
    code_lines = []
    
    # Extract header section (up to params)
    for i, line in enumerate(lines):
        if 'params = {' in line:
            break
        if not ('Generated Python code for:' in line and i > 10):  # Skip duplicate headers
            header_lines.append(line)
    
    # Extract parameter section
    in_params = False
    for i, line in enumerate(lines):
        if 'params = {' in line:
            in_params = True
            param_lines.append(line)
        elif in_params and line.strip() == '}':
            param_lines.append(line)
            in_params = False
        elif in_params:
            param_lines.append(line)
        elif 'shots = ' in line:
            param_lines.append(line)
            break
    
    # Extract code section
    if code_start > 0:
        code_lines = lines[code_start:]
        # Remove any remaining duplicate headers or broken lines
        cleaned_code_lines = []
        for line in code_lines:
            if not ('Generated Python code for:' in line or 
                   'Extracted from:' in line or
                   line.strip() == '"""'):
                cleaned_code_lines.append(line)
        code_lines = cleaned_code_lines
    
    # Reconstruct the clean file
    clean_content = '\n'.join(header_lines[:8])  # Keep only the main header
    clean_content += '\n\n' + '\n'.join(param_lines)
    clean_content += '\n\n' + '\n'.join(code_lines)
    
    # Write the cleaned file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)

def main():
    """Clean all Python files in the dataset directory."""
    
    dataset_dir = "datasets/claude_sonnet_4_20250514_20250524_130955"
    
    print("🧹 CLEANING PYTHON FILES")
    print("=" * 30)
    
    py_files = [f for f in os.listdir(dataset_dir) if f.endswith('.py')]
    py_files.sort(key=lambda x: int(x.split('.')[0]))
    
    for py_file in py_files:
        py_path = os.path.join(dataset_dir, py_file)
        print(f"🧹 Cleaning {py_file}...")
        clean_python_file(py_path)
        print(f"   ✅ {py_file} cleaned")
    
    print(f"\n✅ All {len(py_files)} Python files cleaned!")
    
    # Test one file
    print(f"\n🧪 Testing cleaned file...")
    test_file = os.path.join(dataset_dir, "1.py")
    
    try:
        # Try to compile the file
        with open(test_file, 'r') as f:
            code = f.read()
        
        compile(code, test_file, 'exec')
        print("✅ File compiles successfully!")
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
    except Exception as e:
        print(f"❌ Other error: {e}")

if __name__ == "__main__":
    main()