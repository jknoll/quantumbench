#!/usr/bin/env python3
"""
Quantum Computing Fine-Tune Dataset Generator

This script generates a dataset for fine-tuning language models to produce 
valid Qiskit code by calling the Anthropic API with the prompt specifications.
"""

import os
import json
import time
import argparse
import signal
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import anthropic
from anthropic.types import Usage


class DatasetGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514", 
                 min_balance: float = 0.0, max_cost: Optional[float] = None, 
                 extended_thinking: bool = False, use_caching: bool = True):
        """Initialize the dataset generator with Anthropic API client."""
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.min_balance = min_balance
        self.max_cost = max_cost
        self.extended_thinking = extended_thinking
        self.use_caching = use_caching
        self.start_balance = None
        self.end_balance = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost = 0.0
        self.generation_stopped_by_budget = False
        self.generation_interrupted = False
        self.requested_examples = 0
        self.completed_examples = 0
        self.output_directory = None
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Configuration for different difficulty levels and categories
        self.difficulty_levels = ["beginner", "intermediate", "advanced"]
        self.categories = [
            "circuit_construction",
            "quantum_algorithms", 
            "optimization",
            "error_mitigation",
            "hardware_interaction",
            "visualization"
        ]
        
        # Supported Anthropic models
        self.supported_models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Received interrupt signal. Finishing current generation and saving progress...")
        self.generation_interrupted = True
    
    def get_billing_info(self) -> Dict[str, Any]:
        """Get current billing information from Anthropic API."""
        try:
            # Note: The Anthropic API doesn't have a direct billing endpoint
            # This is a placeholder for when such functionality becomes available
            # For now, we'll track tokens and estimate costs
            return {
                "balance": "Not available via API",
                "usage": "Tracked via token counting"
            }
        except Exception as e:
            print(f"Warning: Could not retrieve billing info: {e}")
            return {"error": str(e)}
    
    def load_prompt(self, num_examples: int = 1) -> Tuple[str, str]:
        """Load the prompt from prompt.md file and split into cacheable and dynamic parts."""
        print("📄 Loading prompt template from prompt.md...")
        try:
            with open("prompt.md", "r", encoding="utf-8") as f:
                prompt_template = f.read()
            print(f"✓ Prompt template loaded ({len(prompt_template)} characters)")
            
            # Split the prompt at the Extended Thinking Mode Configuration section
            # Everything before this is static and can be cached
            print("🔍 Splitting prompt into static and dynamic parts for caching optimization...")
            split_marker = "## Extended Thinking Mode Configuration"
            if split_marker in prompt_template:
                static_part, dynamic_part = prompt_template.split(split_marker, 1)
                # Add the marker back to dynamic part
                dynamic_part = split_marker + dynamic_part
                print(f"✓ Prompt split: {len(static_part)} chars static, {len(dynamic_part)} chars dynamic")
            else:
                # Fallback if marker not found - treat entire prompt as dynamic
                static_part = ""
                dynamic_part = prompt_template
                print("⚠️  Split marker not found - treating entire prompt as dynamic")
            
            # Template the dynamic part with variables
            print(f"🔧 Templating dynamic prompt with variables (examples: {num_examples}, extended_thinking: {self.extended_thinking})...")
            dynamic_part = dynamic_part.replace("{num_examples}", str(num_examples))
            dynamic_part = dynamic_part.replace("{extended_thinking}", str(self.extended_thinking).lower())
            print("✓ Prompt templating completed")
            
            return static_part.strip(), dynamic_part.strip()
        except FileNotFoundError:
            raise FileNotFoundError("prompt.md file not found. Please ensure it exists in the current directory.")
    
    def estimate_cost(self, usage: Usage) -> float:
        """Estimate cost based on token usage."""
        # Claude pricing - updated for different models
        if "claude-3-5-sonnet" in self.model:
            input_cost_per_token = 0.000003  # $3 per million input tokens
            output_cost_per_token = 0.000015  # $15 per million output tokens
        elif "claude-sonnet-4" in self.model or "claude-4" in self.model:
            # Claude 4 pricing (estimated - adjust based on actual pricing)
            input_cost_per_token = 0.000015  # $15 per million input tokens
            output_cost_per_token = 0.000075  # $75 per million output tokens
        elif "claude-3-haiku" in self.model:
            input_cost_per_token = 0.00000025  # $0.25 per million input tokens
            output_cost_per_token = 0.00000125  # $1.25 per million output tokens
        else:
            # Default to Claude 3.5 Sonnet pricing
            input_cost_per_token = 0.000003
            output_cost_per_token = 0.000015
        
        input_cost = usage.input_tokens * input_cost_per_token
        output_cost = usage.output_tokens * output_cost_per_token
        return input_cost + output_cost
    
    def check_budget_constraints(self) -> Tuple[bool, str]:
        """Check if generation should continue based on budget constraints."""
        # Since we can't get real balance from API, we simulate with cost tracking
        # In practice, you might implement actual balance checking here
        
        # Check maximum cost constraint
        if self.max_cost is not None and self.total_cost >= self.max_cost:
            return False, f"Maximum cost limit reached (${self.total_cost:.6f} >= ${self.max_cost:.2f})"
        
        # Estimate cost for next generation (rough estimate based on current average)
        if self.completed_examples > 0:
            avg_cost_per_example = self.total_cost / self.completed_examples
            estimated_next_cost = self.total_cost + avg_cost_per_example
            
            if self.max_cost is not None and estimated_next_cost > self.max_cost:
                return False, f"Next generation would exceed maximum cost (estimated ${estimated_next_cost:.6f} > ${self.max_cost:.2f})"
        
        # Since Anthropic API doesn't provide billing info, we can only check max_cost
        # When min_balance is specified but balance cannot be verified, we must NOT bypass the check
        # as per Milestone 7 requirements: "the balance check should absolutely not be bypassed"
        if self.min_balance > 0:
            return False, f"Cannot verify minimum balance (${self.min_balance:.2f}) - API balance unavailable. Generation stopped for safety."
        
        return True, "Budget constraints satisfied (no minimum balance specified)"
    
    def extract_and_fix_json(self, response_text: str) -> Dict[str, Any]:
        """Extract and attempt to fix JSON from API response."""
        import re
        
        print(f"🔍 Analyzing response content ({len(response_text)} chars)...")
        
        # Log first and last 500 characters of response for debugging
        print("📝 Response preview (first 500 chars):")
        print(f"'{response_text[:500]}{'...' if len(response_text) > 500 else ''}'")
        if len(response_text) > 1000:
            print("📝 Response preview (last 500 chars):")
            print(f"'...{response_text[-500:]}'")
        
        # First try to find JSON within code blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"✓ Found JSON in code block ({len(json_str)} chars)")
        else:
            # Look for JSON object in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                print(f"✓ Found JSON object ({len(json_str)} chars, positions {json_start}-{json_end})")
            else:
                print("❌ No JSON structure found in response")
                return {"raw_response": response_text, "error": "No JSON found in response"}
        
        # Log the extracted JSON for debugging
        print("📄 Extracted JSON preview (first 1000 chars):")
        print(f"'{json_str[:1000]}{'...' if len(json_str) > 1000 else ''}'")
        
        # Try to fix common JSON issues
        try:
            # First attempt: parse as-is
            print("🔄 Attempting to parse JSON as-is...")
            result = json.loads(json_str)
            print("✅ JSON parsed successfully!")
            return result
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Error position: line {e.lineno}, column {e.colno}")
            
            # Show context around the error
            lines = json_str.split('\n')
            if e.lineno <= len(lines):
                error_line = lines[e.lineno - 1] if e.lineno > 0 else ""
                print(f"   Error line {e.lineno}: '{error_line}'")
                if e.colno > 0 and e.colno <= len(error_line):
                    pointer = " " * (e.colno - 1) + "^"
                    print(f"   Error position: {pointer}")
            
            print("🔧 Attempting to fix common JSON issues...")
            
            # Common fixes for AI-generated JSON
            fixed_json = json_str
            
            # Fix unescaped quotes in strings (major issue)
            # This is a more aggressive fix for multiline code strings
            lines = fixed_json.split('\n')
            fixed_lines = []
            in_string = False
            string_delimiter = None
            
            for line_num, line in enumerate(lines, 1):
                if line_num == e.lineno:
                    print(f"🔧 Applying aggressive fix to problem line {line_num}")
                    # For the problematic line, try to close any open strings
                    if '"' in line and not line.strip().endswith('"') and not line.strip().endswith('",'):
                        line = line.rstrip() + '"'
                        print(f"   Fixed line: {line}")
                
                fixed_lines.append(line)
            
            fixed_json = '\n'.join(fixed_lines)
            
            # Additional fixes
            # Fix escaped newlines in strings that break JSON
            fixed_json = re.sub(r'\\n(?=\d)', ' ', fixed_json)  # Remove \n before numbers
            fixed_json = re.sub(r'\\n', ' ', fixed_json)  # Replace remaining \n with spaces
            
            # Fix any remaining control characters
            fixed_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', fixed_json)
            
            # Fix common trailing comma issues
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
            
            # More aggressive fix for unterminated strings - truncate problematic code blocks
            lines = fixed_json.split('\n')
            if e.lineno <= len(lines):
                problematic_line = lines[e.lineno - 1]
                if '"incorrect_code":' in problematic_line and not problematic_line.strip().endswith('"'):
                    # Truncate the code and close the string
                    truncation_point = min(e.colno + 100, len(problematic_line))
                    lines[e.lineno - 1] = problematic_line[:truncation_point].rstrip() + '"'
                    print(f"🔧 Truncated problematic code field at position {truncation_point}")
                    
                    # Remove any subsequent lines that would be part of the unterminated string
                    # until we find the next JSON field or closing brace
                    for i in range(e.lineno, len(lines)):
                        if ('error_type":' in lines[i] or 
                            'explanation":' in lines[i] or 
                            '}' in lines[i]):
                            break
                        lines[i] = ''  # Clear problematic continuation lines
                    
                    fixed_json = '\n'.join(lines)
                    print(f"🔧 Cleaned {len([l for l in lines if l == ''])} problematic lines")
            
            # Try parsing the fixed version
            try:
                print("🔄 Attempting to parse fixed JSON...")
                result = json.loads(fixed_json)
                print("✅ Fixed JSON parsed successfully!")
                return result
            except json.JSONDecodeError as e2:
                print(f"❌ Fixed JSON parsing also failed: {e2}")
                print(f"   Error position: line {e2.lineno}, column {e2.colno}")
                
                # Show more detailed error information
                lines = fixed_json.split('\n')
                if e2.lineno <= len(lines):
                    error_line = lines[e2.lineno - 1] if e2.lineno > 0 else ""
                    print(f"   Fixed error line {e2.lineno}: '{error_line}'")
                
                return {
                    "raw_response": response_text,
                    "extracted_json": json_str[:2000] + "..." if len(json_str) > 2000 else json_str,
                    "fixed_json": fixed_json[:2000] + "..." if len(fixed_json) > 2000 else fixed_json,
                    "original_error": str(e),
                    "fixed_error": str(e2),
                    "error": "Failed to parse JSON even after attempted fixes"
                }

    def generate_dataset_entry(self, static_prompt: str, dynamic_prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """Generate a single dataset entry using the Anthropic API with retry logic and prompt caching."""
        
        print("🚀 Starting dataset entry generation...")
        
        # Enhanced dynamic prompt to ensure valid JSON
        print("🔧 Enhancing prompt with JSON formatting instructions...")
        enhanced_dynamic_prompt = dynamic_prompt + """

CRITICAL JSON FORMATTING REQUIREMENTS:
1. Your response must be a single, valid JSON object starting with { and ending with }
2. Do not include any text before or after the JSON
3. Do not use markdown code blocks (no ```json)
4. All code in "code" fields must be on single lines with \\n for line breaks
5. All strings must be properly escaped - replace actual newlines with \\n
6. Escape quotes inside strings with \\"
7. Ensure no unterminated strings
8. Test that your JSON is valid before responding

Example of proper code formatting in JSON:
"code": "from qiskit import QuantumCircuit\\nqc = QuantumCircuit(3)\\nqc.h([0,1,2])\\nprint('circuit created')"

Please provide only the JSON object."""

        for attempt in range(max_retries + 1):
            try:
                print(f"📡 API call attempt {attempt + 1}/{max_retries + 1}")
                
                # Create messages with optional prompt caching
                if self.use_caching:
                    print("📝 Constructing API messages with prompt caching...")
                else:
                    print("📝 Constructing API messages (caching disabled)...")
                messages = []
                
                # Add static (cacheable) part if it exists
                if static_prompt:
                    if self.use_caching:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": static_prompt,
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        })
                        print("✓ Added static (cacheable) prompt part")
                    else:
                        # Without caching, combine static and dynamic parts
                        combined_prompt = static_prompt + "\n\n" + enhanced_dynamic_prompt
                        messages.append({
                            "role": "user",
                            "content": combined_prompt
                        })
                        print("✓ Added combined prompt (no caching)")
                        # Skip adding the dynamic part separately
                        enhanced_dynamic_prompt = None
                
                # Add dynamic part (if not already combined)
                if enhanced_dynamic_prompt:
                    messages.append({
                        "role": "user", 
                        "content": enhanced_dynamic_prompt
                    })
                    print("✓ Added dynamic prompt part")
                
                print(f"🔄 Sending request to {self.model} (max_tokens: 4000, temperature: 0.3)...")
                
                # Log message structure for debugging
                print(f"📝 Request structure: {len(messages)} message(s)")
                for i, msg in enumerate(messages):
                    if isinstance(msg.get('content'), list):
                        print(f"  Message {i+1}: role={msg['role']}, content={len(msg['content'])} parts")
                        for j, content_part in enumerate(msg['content']):
                            if content_part.get('type') == 'text':
                                text_len = len(content_part['text'])
                                has_cache = 'cache_control' in content_part
                                print(f"    Part {j+1}: text ({text_len} chars, cached={has_cache})")
                    else:
                        content_len = len(msg.get('content', ''))
                        print(f"  Message {i+1}: role={msg['role']}, content=text ({content_len} chars)")
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.3,  # Lower temperature for more consistent formatting
                    messages=messages
                )
                
                print("✅ API response received successfully!")
                
                # Log raw response structure for debugging
                print(f"🔍 Response structure analysis:")
                print(f"  Response type: {type(response)}")
                print(f"  Response content type: {type(response.content)}")
                print(f"  Content length: {len(response.content) if response.content else 0}")
                if response.content:
                    print(f"  First content type: {type(response.content[0]) if response.content else 'None'}")
                    if hasattr(response.content[0], 'text'):
                        print(f"  Text length: {len(response.content[0].text)}")
                
                # Track usage
                if response.usage:
                    print("📊 Processing usage metrics...")
                    
                    # Log all usage attributes for debugging
                    print(f"🔍 Usage object analysis:")
                    for attr in dir(response.usage):
                        if not attr.startswith('_'):
                            value = getattr(response.usage, attr, 'N/A')
                            print(f"  {attr}: {value}")
                    
                    self.total_tokens_in += response.usage.input_tokens
                    self.total_tokens_out += response.usage.output_tokens
                    cost = self.estimate_cost(response.usage)
                    self.total_cost += cost
                    
                    print(f"📈 Request completed:")
                    print(f"  Input tokens: {response.usage.input_tokens}")
                    print(f"  Output tokens: {response.usage.output_tokens}")
                    
                    # Show cache usage if available - check actual field names
                    cache_creation = getattr(response.usage, 'cache_creation_input_tokens', 0)
                    cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
                    if cache_creation > 0:
                        print(f"  Cache creation tokens: {cache_creation}")
                    if cache_read > 0:
                        print(f"  Cache read tokens: {cache_read}")
                    
                    print(f"  Estimated cost: ${cost:.6f}")
                
                # Extract the response content
                print("📄 Extracting response content...")
                response_text = response.content[0].text if response.content else ""
                print(f"✓ Response content extracted ({len(response_text)} characters)")
                
                # Try to extract and parse JSON
                print("🔍 Parsing and validating JSON response...")
                result = self.extract_and_fix_json(response_text)
                
                # Check if we got a valid dataset structure
                if "problem_id" in result and "error" not in result:
                    print("✓ Successfully generated valid dataset JSON")
                    return result
                elif attempt < max_retries:
                    print(f"✗ Invalid JSON structure, retrying... (attempt {attempt + 1})")
                    continue
                else:
                    print("✗ Failed to generate valid JSON after all retries")
                    return result
                    
            except Exception as e:
                print(f"🚨 Exception occurred during API call:")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Exception message: {str(e)}")
                
                # Log more details for specific exception types
                if hasattr(e, 'response'):
                    print(f"  HTTP response: {getattr(e.response, 'status_code', 'N/A')}")
                    print(f"  Response text: {getattr(e.response, 'text', 'N/A')[:500]}")
                
                if hasattr(e, 'body'):
                    print(f"  Error body: {e.body}")
                
                if attempt < max_retries:
                    print(f"✗ API call failed, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    print(f"  Waiting 2 seconds before retry...")
                    time.sleep(2)  # Add a small delay before retry
                    continue
                else:
                    return {
                        "error": f"API call failed after {max_retries + 1} attempts: {str(e)}",
                        "exception_type": type(e).__name__,
                        "exception_details": str(e)
                    }
        
        return {"error": "Unexpected error in retry loop"}
    
    def generate_multiple_datasets(self, num_examples: int = 1, difficulty: str = None, category: str = None) -> List[Dict[str, Any]]:
        """Generate multiple dataset entries with budget management."""
        datasets = []
        self.requested_examples = num_examples
        self.completed_examples = 0
        static_prompt, dynamic_prompt = self.load_prompt(num_examples)
        
        # If specific difficulty/category requested, use targeted generation
        if difficulty or category:
            if difficulty and difficulty not in self.difficulty_levels:
                raise ValueError(f"Difficulty must be one of: {self.difficulty_levels}")
            if category and category not in self.categories:
                raise ValueError(f"Category must be one of: {self.categories}")
            
            for i in range(num_examples):
                # Check for interrupt signal
                if self.generation_interrupted:
                    print(f"\n🛑 GENERATION INTERRUPTED - Stopping generation")
                    print(f"Generated {self.completed_examples}/{self.requested_examples} examples before interruption")
                    break
                
                # Check budget constraints before each generation
                can_continue, message = self.check_budget_constraints()
                if not can_continue:
                    print(f"\n🛑 BUDGET CONSTRAINT REACHED - Stopping generation")
                    print(f"Reason: {message}")
                    print(f"Generated {self.completed_examples}/{self.requested_examples} examples before stopping")
                    self.generation_stopped_by_budget = True
                    break
                
                print(f"\nGenerating dataset entry {i+1}/{num_examples} - {difficulty or 'any'}/{category or 'any'}")
                print(f"Budget status: ${self.total_cost:.4f} spent so far")
                
                # Customize dynamic prompt for specific difficulty and category
                targeted_dynamic_prompt = dynamic_prompt
                if difficulty or category:
                    targeted_dynamic_prompt += f"""

SPECIFIC REQUIREMENTS FOR THIS GENERATION:
- Difficulty Level: {difficulty or 'any appropriate level'}
- Category: {category or 'any appropriate category'}
- Problem ID should include difficulty and category indicators
- Ensure the problem complexity matches the specified level
- Focus specifically on the specified concepts

"""
                
                result = self.generate_dataset_entry(static_prompt, targeted_dynamic_prompt)
                if "error" not in result:
                    # Ensure the difficulty and category are set correctly if specified
                    if difficulty:
                        result["difficulty"] = difficulty
                    if category:
                        result["category"] = category
                    
                    # Save individual file immediately
                    example_number = self.completed_examples + 1
                    saved_file = self.save_individual_dataset(result, example_number)
                    
                    if saved_file:
                        datasets.append(result)
                        self.completed_examples += 1
                        print(f"✅ Successfully generated and saved example {self.completed_examples}")
                    else:
                        print(f"❌ Failed to save entry {i+1}")
                else:
                    print(f"❌ Failed to generate entry {i+1}: {result.get('error', 'Unknown error')}")
        else:
            # For non-targeted generation, generate one example at a time for better interrupt handling
            for i in range(num_examples):
                # Check for interrupt signal
                if self.generation_interrupted:
                    print(f"\n🛑 GENERATION INTERRUPTED - Stopping generation")
                    print(f"Generated {self.completed_examples}/{self.requested_examples} examples before interruption")
                    break
                
                # Check budget before generation
                can_continue, message = self.check_budget_constraints()
                if not can_continue:
                    print(f"\n🛑 BUDGET CONSTRAINT REACHED - Stopping generation")
                    print(f"Reason: {message}")
                    self.generation_stopped_by_budget = True
                    break
                
                print(f"\nGenerating dataset entry {i+1}/{num_examples}")
                print(f"Budget status: ${self.total_cost:.4f} spent so far")
                
                # Generate single example by modifying the dynamic prompt
                single_example_prompt = dynamic_prompt.replace(f"{num_examples} problem(s)", "1 problem")
                
                result = self.generate_dataset_entry(static_prompt, single_example_prompt)
                if "error" not in result:
                    # Save individual file immediately
                    example_number = self.completed_examples + 1
                    saved_file = self.save_individual_dataset(result, example_number)
                    
                    if saved_file:
                        datasets.append(result)
                        self.completed_examples += 1
                        print(f"✅ Successfully generated and saved example {self.completed_examples}")
                    else:
                        print(f"❌ Failed to save entry {i+1}")
                else:
                    print(f"❌ Failed to generate entry {i+1}: {result.get('error', 'Unknown error')}")
        
        return datasets
    
    def generate_targeted_dataset(self, difficulty: str = "intermediate", category: str = "quantum_algorithms", count: int = 1) -> List[Dict[str, Any]]:
        """Generate multiple dataset entries with specified difficulty and category. (Deprecated - use generate_multiple_datasets)"""
        return self.generate_multiple_datasets(count, difficulty, category)
    
    def save_session_metadata(self):
        """Save metadata about the generation session."""
        if self.output_directory is None:
            return None
        
        try:
            metadata = {
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "extended_thinking": self.extended_thinking,
                    "requested_examples": self.requested_examples,
                    "completed_examples": self.completed_examples,
                    "generation_stopped_by_budget": self.generation_stopped_by_budget,
                    "generation_interrupted": self.generation_interrupted
                },
                "budget_info": {
                    "min_balance_threshold": self.min_balance,
                    "max_cost_limit": self.max_cost,
                    "total_cost": self.total_cost,
                    "total_tokens_in": self.total_tokens_in,
                    "total_tokens_out": self.total_tokens_out
                },
                "output_info": {
                    "output_directory": self.output_directory,
                    "files_generated": [f"{i}.json" for i in range(1, self.completed_examples + 1)]
                }
            }
            
            metadata_file = os.path.join(self.output_directory, "session_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Session metadata saved to: {metadata_file}")
            return metadata_file
        except Exception as e:
            print(f"Error saving session metadata: {e}")
            return None
    
    def create_output_directories(self):
        """Create output directories if they don't exist."""
        print("📁 Creating output directories...")
        os.makedirs("datasets", exist_ok=True)
        os.makedirs("dataset_validations", exist_ok=True)
        print("✓ Created base directories: datasets/, dataset_validations/")
        
        # Create the specific output directory for this session
        if self.output_directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model.replace("-", "_")
            self.output_directory = os.path.join("datasets", f"{model_name}_{timestamp}")
            print(f"✓ Generated session directory: {self.output_directory}")
        
        os.makedirs(self.output_directory, exist_ok=True)
        print(f"✅ Output directory ready: {self.output_directory}")
        return self.output_directory
    
    def get_timestamped_filename(self, base_name: str, extension: str = ".json") -> str:
        """Generate a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.replace("-", "_")
        return f"{base_name}_{model_name}_{timestamp}{extension}"
    
    def save_individual_dataset(self, data: Dict[str, Any], example_number: int) -> str:
        """Save a single dataset entry to its own JSON file and extract code to a .py file."""
        try:
            print(f"💾 Saving dataset example {example_number}...")
            output_dir = self.create_output_directories()
            json_filename = os.path.join(output_dir, f"{example_number}.json")
            py_filename = os.path.join(output_dir, f"{example_number}.py")
            
            # Save JSON file
            print(f"📁 Writing JSON to file: {json_filename}")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Dataset example {example_number} JSON saved to {json_filename}")
            
            # Extract and save Python code
            try:
                # Extract code from the solution field
                solution_code = None
                if "solution" in data and isinstance(data["solution"], dict) and "code" in data["solution"]:
                    solution_code = data["solution"]["code"]
                elif "solution" in data and isinstance(data["solution"], str):
                    # Handle old format where solution might be a string
                    solution_code = data["solution"]
                elif "code" in data:
                    # Handle case where code is at top level
                    solution_code = data["code"]
                
                if solution_code:
                    # Convert \n to actual newlines and clean up the code
                    clean_code = solution_code.replace('\\n', '\n').replace('\\"', '"')
                    
                    print(f"📁 Writing Python code to file: {py_filename}")
                    with open(py_filename, "w", encoding="utf-8") as f:
                        f.write(clean_code)
                    print(f"✅ Python code extracted to {py_filename}")
                else:
                    print(f"⚠️  No code found in dataset example {example_number} to extract")
                    
            except Exception as py_error:
                print(f"⚠️  Error extracting Python code for example {example_number}: {py_error}")
            
            return json_filename
        except Exception as e:
            print(f"❌ Error saving dataset example {example_number}: {e}")
            return None
    
    def save_dataset(self, data: Dict[str, Any], custom_filename: str = None):
        """Save the generated dataset to a JSON file in the datasets directory."""
        try:
            self.create_output_directories()
            
            if custom_filename:
                filename = custom_filename
            else:
                filename = os.path.join("datasets", self.get_timestamped_filename("qiskit_dataset"))
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Dataset saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return None
    
    def save_datasets_collection(self, datasets: List[Dict[str, Any]], filename: str = None):
        """Save multiple datasets as a collection."""
        try:
            self.create_output_directories()
            
            if filename is None:
                filename = os.path.join("datasets", self.get_timestamped_filename("qiskit_dataset_collection"))
            
            collection = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_entries": len(datasets),
                    "model_used": self.model,
                    "generator_version": "1.0"
                },
                "datasets": datasets
            }
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(collection, f, indent=2, ensure_ascii=False)
            print(f"Dataset collection saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving dataset collection: {e}")
            return None
    
    def print_summary_stats(self):
        """Print summary statistics of the generation process."""
        print("\n" + "="*60)
        print("DATASET GENERATION SUMMARY")
        print("="*60)
        
        # Generation statistics
        print("GENERATION RESULTS:")
        print(f"  Requested examples: {self.requested_examples}")
        print(f"  Completed examples: {self.completed_examples}")
        if self.requested_examples > 0:
            completion_rate = (self.completed_examples / self.requested_examples) * 100
            print(f"  Completion rate: {completion_rate:.1f}%")
        
        if self.generation_stopped_by_budget:
            print("  🛑 Generation stopped due to budget constraints")
        elif self.generation_interrupted:
            print("  🛑 Generation stopped due to user interrupt")
        else:
            print("  ✅ Generation completed successfully")
        
        print()
        
        # Cost and usage statistics
        print("COST AND USAGE:")
        print(f"  Model used: {self.model}")
        print(f"  Total input tokens: {self.total_tokens_in:,}")
        print(f"  Total output tokens: {self.total_tokens_out:,}")
        print(f"  Total tokens: {self.total_tokens_in + self.total_tokens_out:,}")
        print(f"  Estimated total cost: ${self.total_cost:.6f}")
        
        if self.completed_examples > 0:
            avg_cost = self.total_cost / self.completed_examples
            print(f"  Average cost per example: ${avg_cost:.6f}")
        
        print()
        
        # Budget constraints
        print("BUDGET CONSTRAINTS:")
        if self.min_balance > 0:
            print(f"  Minimum balance threshold: ${self.min_balance:.2f} (Note: Balance verification not available)")
        else:
            print(f"  Minimum balance threshold: ${self.min_balance:.2f} (Disabled)")
        if self.max_cost:
            print(f"  Maximum cost limit: ${self.max_cost:.2f}")
            remaining_budget = self.max_cost - self.total_cost
            print(f"  Remaining budget: ${remaining_budget:.6f}")
        else:
            print("  Maximum cost limit: Not set")
        
        print()
        print("BILLING INFO:")
        print(f"  Starting status: {self.start_balance}")
        print(f"  Ending status: {self.end_balance}")
        print("="*60)
    
    def run(self, num_examples: int = 1, difficulty: str = None, category: str = None):
        """Main execution method with budget management."""
        print("🚀 Starting Quantum Computing Fine-Tune Dataset Generation")
        print("="*60)
        print(f"🤖 Model: {self.model}")
        print(f"📊 Examples to generate: {num_examples}")
        print(f"🧠 Extended thinking: {'Enabled' if self.extended_thinking else 'Disabled'}")
        if difficulty:
            print(f"🎯 Difficulty: {difficulty}")
        if category:
            print(f"📂 Category: {category}")
        
        # Display budget constraints
        print(f"\n💰 BUDGET CONSTRAINTS:")
        if self.min_balance > 0:
            print(f"  Minimum balance threshold: ${self.min_balance:.2f} (Warning: Balance verification not available)")
        else:
            print(f"  Minimum balance threshold: ${self.min_balance:.2f} (Disabled)")
        if self.max_cost:
            print(f"  Maximum cost limit: ${self.max_cost:.2f}")
        else:
            print("  Maximum cost limit: Not set")
        
        # Get initial billing info
        print("\n🔍 Checking initial billing status...")
        self.start_balance = self.get_billing_info()
        print(f"✓ Initial status: {self.start_balance}")
        
        # Initial budget check
        print("💸 Performing initial budget validation...")
        can_continue, message = self.check_budget_constraints()
        if not can_continue:
            print(f"\n🛑 Cannot start generation due to budget constraints")
            print(f"Reason: {message}")
            return
        print("✅ Budget constraints satisfied - proceeding with generation")
        
        # Generate dataset entries
        print(f"\n🏭 Generating {num_examples} dataset entry(ies)...")
        start_time = time.time()
        
        datasets = self.generate_multiple_datasets(num_examples, difficulty, category)
        
        generation_time = time.time() - start_time
        print(f"\n⏱️  Generation completed in {generation_time:.2f} seconds")
        
        # Files are already saved individually, just report the summary
        if self.completed_examples > 0:
            print(f"\n✅ {self.completed_examples} dataset files saved to: {self.output_directory}")
            if self.generation_stopped_by_budget:
                print("⚠️  Note: Generation was incomplete due to budget constraints")
            elif self.generation_interrupted:
                print("⚠️  Note: Generation was incomplete due to user interrupt")
        else:
            print("⚠️  No datasets generated")
        
        # Get final billing info
        print("\n🔍 Checking final billing status...")
        self.end_balance = self.get_billing_info()
        print(f"✓ Final status: {self.end_balance}")
        
        # Save session metadata
        if self.output_directory:
            print("📄 Saving session metadata...")
            self.save_session_metadata()
        
        # Print summary
        print("\n📈 Generating final summary report...")
        self.print_summary_stats()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Quantum Computing Fine-Tune Datasets using Anthropic API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py                           # Generate 1 dataset with default model
  python generate_dataset.py --num-examples 3         # Generate 3 datasets
  python generate_dataset.py --model claude-3-haiku-20240307  # Use different model
  python generate_dataset.py --difficulty beginner    # Generate beginner-level dataset
  python generate_dataset.py --category circuit_construction  # Generate circuit construction dataset
  python generate_dataset.py --num-examples 5 --difficulty advanced --category quantum_algorithms
  python generate_dataset.py --max-cost 0.10         # Limit total cost to $0.10
  python generate_dataset.py --min-balance 0         # Disable balance checking (recommended until API supports balance queries)
  python generate_dataset.py --extended-thinking     # Enable detailed reasoning traces
  python generate_dataset.py --num-examples 10 --max-cost 0.50  # Generate up to 10 examples, max $0.50
  python generate_dataset.py --extended-thinking --difficulty advanced  # Extended reasoning for advanced problems
        """
    )
    
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=1,
        help="Number of dataset examples to generate (default: 1)"
    )
    
    # Get supported models list from DatasetGenerator class
    temp_generator = DatasetGenerator.__new__(DatasetGenerator)
    temp_generator.supported_models = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022", 
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229"
    ]
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude-sonnet-4-20250514",
        help=f"Anthropic model to use (default: claude-sonnet-4-20250514). Supported models: {', '.join(temp_generator.supported_models)}"
    )
    
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["beginner", "intermediate", "advanced"],
        help="Target difficulty level for generated datasets"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["circuit_construction", "quantum_algorithms", "optimization", 
                "error_mitigation", "hardware_interaction", "visualization"],
        help="Target category for generated datasets"
    )
    
    parser.add_argument(
        "--min-balance", "-b",
        type=float,
        default=0.0,
        help="Minimum API balance threshold in USD (default: $0.00 - disabled). Note: Currently requires manual balance verification as Anthropic API doesn't provide balance info. Generation will stop if balance cannot be verified."
    )
    
    parser.add_argument(
        "--max-cost", "-x",
        type=float,
        default=None,
        help="Maximum total cost allowed for this generation session in USD"
    )
    
    parser.add_argument(
        "--extended-thinking", "-e",
        action="store_true",
        default=False,
        help="Enable extended multi-step reasoning trace in generated datasets"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is required")
        print("Please set it with: export ANTHROPIC_API_KEY=your_api_key_here")
        return 1
    
    # Validate arguments
    if args.num_examples < 1:
        print("Error: Number of examples must be at least 1")
        return 1
    
    if args.num_examples > 10:
        print("Warning: Generating more than 10 examples may be expensive and time-consuming")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0
    
    # Validate model parameter
    if args.model not in temp_generator.supported_models:
        print(f"Error: Unsupported model '{args.model}'")
        print(f"Supported models: {', '.join(temp_generator.supported_models)}")
        return 1
    
    # Validate budget parameters
    if args.min_balance < 0:
        print("Error: Minimum balance threshold must be non-negative")
        return 1
    
    if args.max_cost is not None and args.max_cost <= 0:
        print("Error: Maximum cost limit must be positive")
        return 1
    
    # Create and run the generator
    try:
        generator = DatasetGenerator(api_key, args.model, args.min_balance, args.max_cost, args.extended_thinking)
        generator.run(args.num_examples, args.difficulty, args.category)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())