## Quantum Computing Fine Tune Dataset

This is a project to produce a data set of (prompt, completion) pairs where the prompt describes a quantum computational problem and the completion is Qiskit code to solve the problem, along with expected correct input/output pairs and reasoning traces. This dataset can be used for fine tuning a target LLM.

By default, we use Claude-Sonnet-4 or Claude-Opus-4 for code generation.

The flow is as follows. The `prompt.md` file is used as a templated input to `generate_dataset.py`, which breaks the prompt into cacheable and variable regions and submits for completion to the Anthropic API using the model specified in the command line parameter, optionally using extended thinking and generating a variable number of examples.

The dataset is output in the `datasets` directory. 

Each example is written as both a `.json` file and a standalone `.py` file for ease of standalone execution.

The `validate_dataset.py` script is used to read the output dataset and call the code from the completion with parameters and to check the generated code's output against the expected outputs.

The completions are tested on two attributes: interpretability (i.e., no runtime errors) and correctness (i.e., computed outputs match expected outputs for a given set of inputs).

Possible extensions to measurement of the completion code include execution time and quantum speedup. 

See `milestones.md` for project milestones and implementation status.

N.B.: Budget management functionality Work at present because Anthropic's API does not expose officially supported balance checking endpoints. Do not rely on budget management to keep a dataset generation run within your limit.

Because generation takes place and writes example JSON incrementally, it's possible to monitor the Anthropic Console billing balance figure manually and terminate the run if you exceed an intended budget. 