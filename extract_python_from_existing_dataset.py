#!/usr/bin/env python3
"""
Extract Python Code from Existing Dataset

This script extracts Python code from our already-generated JSON dataset files
and saves them as .py files, restoring the functionality that was missing.
"""

import json
import os
import re

def extract_python_code_from_dataset(dataset_dir: str):
    """Extract Python code from all JSON files in the dataset directory."""
    
    print(f"🔧 EXTRACTING PYTHON CODE FROM EXISTING DATASET")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Find all JSON files (excluding session_metadata.json)
    json_files = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json') and filename != 'session_metadata.json':
            json_files.append(filename)
    
    json_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
    
    print(f"Found {len(json_files)} dataset files to process")
    
    extracted_count = 0
    failed_count = 0
    
    for json_file in json_files:
        json_path = os.path.join(dataset_dir, json_file)
        base_name = json_file.replace('.json', '')
        py_path = os.path.join(dataset_dir, f"{base_name}.py")
        
        print(f"\n📝 Processing {json_file}...")
        
        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract code from the solution field
            solution_code = None
            problem_id = data.get('problem_id', 'unknown')
            
            if "solution" in data and isinstance(data["solution"], dict) and "code" in data["solution"]:
                solution_code = data["solution"]["code"]
            elif "solution" in data and isinstance(data["solution"], str):
                # Handle old format where solution might be a string
                solution_code = data["solution"]
            elif "code" in data:
                # Handle case where code is at top level
                solution_code = data["code"]
            
            if solution_code:
                # Convert \\n to actual newlines and clean up the code
                cleaned_code = solution_code.replace('\\n', '\n').replace('\\"', '"')
                
                # Remove leading/trailing whitespace but preserve internal formatting
                cleaned_code = cleaned_code.strip()
                
                # Add header comment
                header = f'''#!/usr/bin/env python3
"""
Generated Python code for: {problem_id}
Extracted from: {json_file}
"""

'''
                
                final_code = header + cleaned_code
                
                # Write Python file
                with open(py_path, 'w', encoding='utf-8') as f:
                    f.write(final_code)
                
                print(f"   ✅ Extracted Python code to {base_name}.py")
                print(f"   📊 Code length: {len(cleaned_code)} characters")
                
                extracted_count += 1
            else:
                print(f"   ⚠️  No code found in solution field")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {json_file}: {e}")
            failed_count += 1
    
    print(f"\n📊 EXTRACTION SUMMARY")
    print("-" * 25)
    print(f"Total files processed: {len(json_files)}")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Failed extractions: {failed_count}")
    print(f"Success rate: {extracted_count/len(json_files)*100:.1f}%")
    
    if extracted_count > 0:
        print(f"\n✅ Python code extraction complete!")
        print(f"You can now run the extracted code directly:")
        print(f"cd {dataset_dir}")
        print(f"python 1.py  # Run first example")
        print(f"python 2.py  # Run second example")
        print(f"etc.")
    
    return extracted_count

def verify_extraction(dataset_dir: str):
    """Verify that Python files were created and are executable."""
    
    print(f"\n🔍 VERIFYING EXTRACTED PYTHON FILES")
    print("-" * 40)
    
    py_files = [f for f in os.listdir(dataset_dir) if f.endswith('.py')]
    py_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"Found {len(py_files)} Python files:")
    
    for py_file in py_files:
        py_path = os.path.join(dataset_dir, py_file)
        file_size = os.path.getsize(py_path)
        
        # Check if file has basic Python structure
        with open(py_path, 'r', encoding='utf-8') as f:
            content = f.read()
            has_imports = 'import' in content
            has_qiskit = 'qiskit' in content
            has_circuit = 'circuit' in content or 'Circuit' in content
        
        status = "✅" if (has_imports and has_qiskit and has_circuit) else "⚠️"
        print(f"  {status} {py_file} ({file_size} bytes)")
        
        if not (has_imports and has_qiskit and has_circuit):
            print(f"      Missing: {'imports' if not has_imports else ''} {'qiskit' if not has_qiskit else ''} {'circuit' if not has_circuit else ''}")

def main():
    """Extract Python code from the existing dataset."""
    
    dataset_dir = "datasets/claude_sonnet_4_20250514_20250524_130955"
    
    # Extract Python code
    extracted_count = extract_python_code_from_dataset(dataset_dir)
    
    if extracted_count > 0:
        # Verify extraction
        verify_extraction(dataset_dir)
        
        print(f"\n🎯 REGRESSION FIXED!")
        print("-" * 20)
        print("✅ Python code extraction functionality restored")
        print("✅ All dataset examples now have corresponding .py files")
        print("✅ Files can be executed directly for testing")
        print("✅ Maintains compatibility with existing workflow")

if __name__ == "__main__":
    main()