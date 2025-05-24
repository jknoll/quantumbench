# Budget Management Features Demo

This document demonstrates the new budget management capabilities implemented in Milestone 7.

## 🚀 New Command Line Options

### `--min-balance` / `-b`
Set minimum API balance threshold (default: $0.00 - disabled)
**Important**: Currently requires manual balance verification as Anthropic API doesn't provide balance info. Generation will stop if balance cannot be verified.
```bash
python generate_dataset.py --min-balance 0      # Recommended: Disable balance checking
python generate_dataset.py --min-balance 1.00  # Will fail: Requires balance verification unavailable via API
```

### `--max-cost` / `-x`  
Set maximum total cost for the generation session
```bash
python generate_dataset.py --max-cost 0.50     # Limit total cost to $0.50
```

## 💰 Budget-Aware Generation

### Smart Cost Estimation
- **Model-specific pricing**: Different rates for Claude 3.5 Sonnet, Claude 4, and Haiku
- **Predictive cost checking**: Estimates next generation cost to prevent overage
- **Real-time budget tracking**: Shows cost after each generation

### Automatic Stopping
Generation stops automatically when:
- **Cost limit reached**: Total cost hits maximum
- **Next generation would exceed limit**: Prevents overage
- **Balance threshold reached**: Protects minimum balance

## 📊 Enhanced Reporting

### Generation Statistics
```
GENERATION RESULTS:
  Requested examples: 5
  Completed examples: 2
  Completion rate: 40.0%
  🛑 Generation stopped due to budget constraints
```

### Cost Analysis
```
COST AND USAGE:
  Model used: claude-sonnet-4-20250514
  Total input tokens: 15,000
  Total output tokens: 6,000
  Estimated total cost: $0.210000
  Average cost per example: $0.105000
```

### Budget Status
```
BUDGET CONSTRAINTS:
  Minimum balance threshold: $0.50
  Maximum cost limit: $0.10
  Remaining budget: $0.040000
```

## 🎯 Example Usage Scenarios

### Development/Testing (Low Cost)
```bash
python generate_dataset.py --num-examples 2 --max-cost 0.10
```

### Production Generation (Balanced)
```bash
python generate_dataset.py --num-examples 10 --max-cost 2.00 --min-balance 0
```

### Large Scale Generation (High Volume)
```bash
python generate_dataset.py --num-examples 50 --max-cost 10.00 --min-balance 0
```

### Cost-Only Protection (Recommended)
```bash
python generate_dataset.py --num-examples 100 --max-cost 5.00
# Uses max-cost limit only, since balance verification is not available via API
```

## ✅ Key Benefits

1. **Cost Control**: Never exceed intended budget
2. **Balance Protection**: Safely fails when balance verification unavailable (preserves minimum balance requirement)
3. **Partial Results**: Get valid JSON output even if stopped early
4. **Transparency**: Clear reporting of costs and constraints
5. **Flexibility**: Separate controls for maximum cost and minimum balance
6. **Safety**: Predictive checking prevents accidental overages

⚠️ **Current Limitation**: Minimum balance checking requires API balance verification which is not currently available from Anthropic. Use `--max-cost` for budget control instead.

## 🧪 Testing

Run the test suite to see budget management in action:
```bash
python test_budget_management.py
```

This demonstrates various scenarios including:
- No budget constraints
- Maximum cost limits
- Minimum balance thresholds  
- Combined constraints
- Early stopping behavior