# CLAUDE.md

This file contains information about the project structure and common commands for Claude Code.

## Project Overview
This is a project to produce a dataset of prompts describing computational problems which are mapped to Qiskit code to execute the problem, along with expected correct outputs and reasoning traces.

## Project Goals
1. Create a Python script to call the Anthropic API and use the prompt found in `prompt.md` to request creation of a single JSON response as described in `prompt.md`
2. Add a mechanism to query for the current Anthropic organizations' billing stats and current total balance at the start and end of the script, and print summary statistics (starting balance, stopping balance, tokens in, tokens out, total cost) at the end of execution

## Project Structure
This appears to be a Qiskit-related Python project with the following files:
- `claude-test.py` - Test file for Claude interactions
- `prompt.md` - Project documentation/prompts containing the main prompt for dataset generation
- `requirements.txt` - Python dependencies
- `test.py` - Test file

## Virtual Environment
This project runs inside of a Python virtualenv environment, which is defined in the `.qiskit` folder. Be sure to activate this environment before taking other actions in the project if it is not already activated.

```bash
  $ source .qiskit/bin/activate
```

## Dependencies
Be sure that the virtual environment is activated as indicated above. For any new dependencies, be sure to add them to the requirements.txt file and install them using `pip install -r requirements.txt`. Do not install them using pip install [dependency]. Always use the requirements.txt file.

## Testing
For all new functionality, create pytest tests within the `tests/` directory. For modifications to existing functionality, ensure that regression tests exist to verify correct behavior. 

## Scripts

### Dataset Generation
- `generate_dataset.py` - Main script to generate datasets using Anthropic API with directory structure and timestamped files. Now automatically extracts Python code from solutions and saves as executable `.py` files alongside JSON.
- `test_generate_dataset.py` - Test suite for the dataset generation functionality

### Dataset Validation and Analysis
- `validate_dataset.py` - Validates dataset entries by executing Qiskit code and comparing outputs, saves results to dataset_validations/ with correlated filenames
- `dataset_quality_scorer.py` - Comprehensive quality scoring system for datasets
- `dataset_exporter.py` - Export datasets to various formats (CSV, JSONL, training formats)
- `dataset_visualizer.py` - Create visualizations for dataset analysis and quality metrics
- `create_test_dataset.py` - Creates test datasets for validation testing

## Directory Structure
- `datasets/` - Generated datasets with timestamped filenames. Each generation session creates a subdirectory containing numbered JSON files (1.json, 2.json, etc.) and corresponding executable Python files (1.py, 2.py, etc.)
- `dataset_validations/` - Validation results, quality reports, and analysis files
- `exports/` - Exported datasets in various formats
- `visualizations/` - Generated visualization plots and charts

## Common Commands

### Generate Dataset
```bash
export ANTHROPIC_API_KEY=your_api_key_here
source .qiskit/bin/activate

# Basic generation (1 example, default model)
python generate_dataset.py

# Generate multiple examples
python generate_dataset.py --num-examples 3

# Use different model
python generate_dataset.py --model claude-3-haiku-20240307

# Generate targeted datasets
python generate_dataset.py --difficulty beginner --category circuit_construction
python generate_dataset.py --num-examples 2 --difficulty advanced --category quantum_algorithms

# See all options
python generate_dataset.py --help
```

**Note:** The `generate_dataset.py` script now automatically extracts executable Python code from each generated solution and saves it as a `.py` file alongside the JSON. This allows for easy command-line testing of the generated code:

```bash
# After generation, test the extracted code directly
source .qiskit/bin/activate
cd datasets/[model_timestamp_directory]/
python 1.py  # Run the first generated example
python 2.py  # Run the second generated example
```

### Validate Dataset
```bash
source .qiskit/bin/activate
python validate_dataset.py [dataset_file.json]

# The validation outputs will use the same model name and timestamp as the input dataset:
# Input:  datasets/qiskit_dataset_claude_3_5_sonnet_20241022_20250523_175609.json
# Output: dataset_validations/dataset_validation_claude_3_5_sonnet_20241022_20250523_175609.txt
#         dataset_validations/dataset_validation_claude_3_5_sonnet_20241022_20250523_175609.json
```

### Quality Scoring
```bash
source .qiskit/bin/activate
python dataset_quality_scorer.py dataset_file.json
```

### Export Dataset
```bash
source .qiskit/bin/activate
python dataset_exporter.py dataset_file.json [csv|jsonl|instruction_following|code_completion|preference_ranking]
```

### Create Visualizations
```bash
source .qiskit/bin/activate
python dataset_visualizer.py dataset_file.json [overview|validation|comprehensive]
```

### Generate Targeted Datasets
```bash
source .qiskit/bin/activate
python -c "
from generate_dataset import DatasetGenerator
generator = DatasetGenerator()
datasets = generator.generate_targeted_dataset('beginner', 'circuit_construction', 3)
generator.save_datasets_collection(datasets)
"
```

### Run Tests
```bash
source .qiskit/bin/activate
python test_generate_dataset.py
```

## Notes
- This is a Python project
- Dependencies are managed via requirements.txt