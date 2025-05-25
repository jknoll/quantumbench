# Project Milestones

## ✅ Milestone 1: Basic Dataset Generator Implementation
- [x] Create a Python script to call the Anthropic API and use the prompt found in `prompt.md` to request creation of a single JSON response as described in `prompt.md`
- [x] Add a mechanism to query for the current Anthropic organizations' billing stats and current total balance at the start of the above script and at the end of the script, and print summary statistics (starting balance, stopping balance, tokens in, tokens out, total cost) at the end of execution
- [x] Implement proper error handling and JSON parsing
- [x] Create comprehensive test suite to verify functionality

**Status: ✅ COMPLETED**

## ✅ Milestone 2: Dataset Validation and Testing
- [x] Create a separate script which reads the resulting `qiskit_dataset.json` file and executes sequentially the code from each entry using Qiskit
- [x] Compare the expected output and the actual output
- [x] In cases where they do not match, calculate an RMSE error if possible
- [x] Use `tqdm` to provide progress information through the dataset
- [x] Provide a table summarizing the output when execution of all code is complete
- [x] Table should include: problem ID, difficulty, category, preference rank of solution, and categorical column for whether it is a solution or an example of a common mistake

**Status: ✅ COMPLETED**

## ✅ Milestone 3: Output Structuring
- [x] Modify generate dataset to write into `datasets` directory with logic to create directory if it doesn't exist
- [x] Change filename format to `qiskit_dataset_model_name_and_version_<timestamp>`
- [x] Modify validate_dataset to print summary table to stdout and write to `dataset_validations` directory
- [x] Create dataset_validations directory logic
- [x] Name validation output files as `dataset_validation_<timestamp>.txt`
- [x] Write JSON equivalent with structured data from validation table

**Status: ✅ COMPLETED** 

## ✅ Milestone 4: Configuration
- [x] Add command-line parameter to configure number of examples to generate (default: 1)
- [x] Make number of examples a templated value in prompt.md instead of hardcoded
- [x] Add command-line parameter to specify Anthropic model (default: claude-3-5-sonnet-20241022)
- [x] Update generate_dataset.py to use configurable model parameter

**Status: ✅ COMPLETED**


## ✅ Milestone 5: Default Model, Validation Output Formatting
- [x] 5.1: **Default Model Update** - Change default model from `claude-3-5-sonnet-20241022` to `claude-sonnet-4-20250514`
- [x] 5.2: **Filename Correlation** - Fix dataset validation filename extraction to properly read model name and timestamp from dataset filename (not `unknown_model`)
- [x] 5.3: **Common Mistakes Format** - Update prompt.md to require complete, executable code for common mistakes instead of code snippets
- [x] 5.4: **Validation Enhancement** - Improve validation to handle complete common mistakes without NameErrors
- [x] 5.5: **Collection Support** - Add support for validating all examples in multi-problem dataset collections
- [x] 5.6: **Separate Category Analysis** - Break down validation results into separate solutions vs common mistakes sections with targeted metrics

**Status: ✅ COMPLETED** 

## ✅ Milestone 6: Statistical Analysis and Shot Optimization
- [x] 6.1: **Shot Count Analysis** - Create statistical analysis tool to determine optimal number of shots for quantum circuit validation
- [x] 6.2: **Statistical Power Calculation** - Implement binomial confidence intervals and power analysis for distinguishing correct vs incorrect algorithms
- [x] 6.3: **Validation Recommendations** - Provide data-driven recommendations for shot counts (validated 1000 shots as optimal)
- [x] 6.4: **Current Dataset Analysis** - Analyze existing validation results to demonstrate statistical reliability

**Status: ✅ COMPLETED**

## ✅ Milestone 7: API Usage Budget Management
- [x] Add command-line parameter for minimum API balance threshold (`--min-balance`)
- [x] Add command-line parameter for maximum cost limit (`--max-cost`)
- [x] Implement budget checking before each generation loop
- [x] Stop generation when balance approaches minimum threshold
- [x] Stop generation when cost would exceed maximum limit
- [x] Output informative message when stopping due to budget constraints
- [x] Ensure valid JSON output even with incomplete generation
- [x] Report requested vs completed example counts with completion rates
- [x] Enhanced summary statistics with budget information
- [x] Model-specific cost estimation for different Claude models

When a minimum balance check is in effect, if the balance is for some reason not available via the API, then the balance check should absolutely not be bypassed, nor should a mock value for the balance be used.

It is of paramount importance to the project that the minimum balance as specified by the user be preserved in all cases. 

**Status: ✅ COMPLETED**

## ✅ Milestone 8: Prompt Caching
- [x] Modify prompt.md to move static elements to the beginning and dynamic/templated elements to the end
- [x] Ensure semantic references remain correct after reordering (verified "above" and "following" references)
- [x] Update generate_dataset.py to split prompt into cacheable and dynamic parts
- [x] Enable prompt caching using cache_control with ephemeral type for static content
- [x] Add cache usage tracking and reporting in API response handling

## ✅ Milestone 9: Multi-file output
- [x] Modified generate_dataset.py to write one JSON file per example instead of single file
- [x] Implemented graceful Ctrl+C interrupt handling using signal handlers
- [x] Created directory structure: datasets/<model_name>_<timestamp>/
- [x] Write numbered JSON files (1.json, 2.json, etc.) for each generated example
- [x] Added session metadata tracking (session_metadata.json) with generation statistics
- [x] Files are saved immediately after generation to preserve progress on interrupt
- [x] Updated budget and progress reporting for multi-file structure

**Status: ✅ COMPLETED**

## ✅ Milestone 10: Simplify dataset format and reflect the simplification in `prompt.md` 
- [x] **Single Solution Format**: Changed from multiple ranked solutions to single "solution" field
- [x] **Unified Reasoning Trace**: Replaced multi-section reasoning_trace with single comprehensive string 
- [x] **Removed Common Mistakes**: Eliminated common_mistakes section to focus on correct solutions only
- [x] **Prompt Simplification**: Updated prompt.md to reflect simplified structure and requirements
- [x] **Validator Compatibility**: Updated validator to handle both old and new formats seamlessly
- [x] **Format Validation**: Tested generation and validation of simplified format successfully

**Status: ✅ COMPLETED** 

## ✅ Milestone 11: Make the correctness check independent of the chosen code implementation
The generated code currently includes logic at the end of the quantum circuit setup and execution to look at the output and verify through assertion that the output is correct. The dataset schema also includes, in the `solution` attribute, an `expected output` string which is checked against the actual output to verify correctness.

The problem is that when using this system for fine-tuning another LLM, we can't guarantee that the target LLM will generate code that has the same syntax and logic for verifying its correctness. Therefore, the expected output, which is tightly bound to the solution provided in the dataset, will not be acceptable for checking correctness of the code of the target LLM.

I want to allow for a generic mechanism for testing correctness of a target LLM's completions for the prompts which will be included in this dataset.

The approach that KernelBench takes is to include information about input parameters in the prompt and expect the target LLM to generate a completion which takes these parameters into account.

It then calls the target LLM's generated code and the reference implementation with a set of random parameters and verifies that the outputs match. If they do, then the target LLM's generated code is scored as correct. 

Is it possible (and advisable?) to modify this system so that the generated dataset `prompt` attribute includes a specification of input parameters to the quantum circuit, and a set of expected outputs attributes would include input-output pairs for which we could test the code in the dataset listed as the solution, as well as any code generated by the fine-tuned target LLM?

**Status: ✅ COMPLETED**

**Implementation Summary:**
- ✅ **Parameterized Dataset Schema**: Extended dataset format to include parameter specifications, test cases, and algorithm types
- ✅ **Statistical Validation Framework**: Implemented measurement distribution comparison using chi-squared tests and total variation distance
- ✅ **Parameter Generation System**: Automatic generation of test parameters for Grover's, Bell states, QFT, and custom algorithms
- ✅ **Implementation-Independent Testing**: Validation system compares any two implementations using the same parameter sets
- ✅ **Backward Compatibility**: Seamlessly handles both legacy (expected output) and parameterized datasets
- ✅ **LLM Evaluation Ready**: Framework enables robust evaluation of fine-tuned model outputs against reference implementations

**Files Created:**
- `parameter_testing_framework.py`: Core parameterized testing system
- `parameterized_validation.py`: Enhanced validation script with automatic format detection
- `test_parameterized_dataset.json`: Example parameterized dataset
- `demo_parameterized_testing.py`: Demonstration of implementation equivalence testing

**Key Benefits:**
1. **Robustness**: No longer dependent on exact output format matching
2. **Flexibility**: Works with any implementation style or coding approach
3. **Statistical Rigor**: Uses proper statistical methods for quantum measurement comparison
4. **Scalability**: Easy to extend to new algorithm types and parameter patterns
5. **Fine-tuning Ready**: Perfect for evaluating LLM-generated quantum code against reference implementations

## Milestone x: Batch Processing
.Add an option for the user to generate the data set using batch processing for reduced overall cost 

## Milestone x: Comparison of Claude 4 Sonnet and Claude 4 Opus
Performance of Claude 4 Sonnet and Claude 4 Opus, and the performance of each when extended reasoning (or extended thinking) is enabled versus disabled.

Generate one datasets for each of the following conditions:
- Claude 4 Sonnet with extended reasoning enabled
- Claude 4 Sonnet with extended reasoning disabled
- Claude 4 Opus with extended reasoning enabled
- Claude 4 Opus with extended reasoning disabled

Each dataset should contain 20 examples.

Validate each dataset and output summary statistics about the percentage interpretable and percentage correct stats that each dataset's solutions yield. Indicate which datasets are best to worst as well as best to worst in terms of cost-adjusted correctness. 

## Milestone x: Simplify validation by removing common_mistake logic.
Update the validation and test scripts  to comport with the changes described in Milestone 10 so that it doesn't look for common mistakes and verify that they are interpretable but do not yield the correct answer. Instead, it should focus only on executing the listed examples and ensuring that they're interpretable and yield the correct answers. 

## 📋 Milestone x: Production Dataset Generation
- [ ] 9.1: **Large Scale Generation** - Generate comprehensive 100+ problem dataset across all categories
- [ ] 9.2: **Quality Assurance** - Implement automated quality checks during generation
- [ ] 9.3: **Dataset Versioning** - Add version control and metadata tracking for datasets
- [ ] 9.4: **Export Formats** - Support multiple export formats (JSONL, HuggingFace datasets, etc.)
- [ ] 9.5: **Documentation** - Create comprehensive usage documentation

**Status: 📋 PENDING**

## 📋 Milestone x : Advanced Validation Features
- [ ] 8.1: **Table Footer Statistics** - Move overall statistics from separate block to table footer rows as requested
- [ ] 8.2: **Enhanced Output Matching** - Improve quantum algorithm output validation for multi-target scenarios (e.g., Grover's with multiple marked states)
- [ ] 8.3: **Validation Test Execution** - Implement execution of validation_tests from dataset entries
- [ ] 8.4: **Performance Metrics** - Add circuit depth, gate count, and execution time metrics to validation
- [ ] 8.5: **Batch Validation** - Support validating multiple dataset files in single command

**Status: 📋 PENDING**