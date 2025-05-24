#!/usr/bin/env python3
"""
Test script for the dataset generator functionality.
"""

import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from generate_dataset import DatasetGenerator


def test_prompt_loading():
    """Test that the prompt can be loaded correctly."""
    generator = DatasetGenerator()
    try:
        prompt = generator.load_prompt()
        assert len(prompt) > 0, "Prompt should not be empty"
        assert "JSON" in prompt, "Prompt should mention JSON format"
        print("✓ Prompt loading test passed")
        return True
    except Exception as e:
        print(f"✗ Prompt loading test failed: {e}")
        return False


def test_billing_info():
    """Test billing info retrieval."""
    generator = DatasetGenerator()
    billing_info = generator.get_billing_info()
    assert isinstance(billing_info, dict), "Billing info should be a dictionary"
    print("✓ Billing info test passed")
    return True


def test_cost_estimation():
    """Test cost estimation functionality."""
    generator = DatasetGenerator()
    
    # Mock usage object
    mock_usage = Mock()
    mock_usage.input_tokens = 1000
    mock_usage.output_tokens = 500
    
    cost = generator.estimate_cost(mock_usage)
    expected_cost = (1000 * 0.000003) + (500 * 0.000015)
    assert abs(cost - expected_cost) < 0.000001, f"Cost calculation incorrect: {cost} vs {expected_cost}"
    print("✓ Cost estimation test passed")
    return True


def test_dataset_saving():
    """Test dataset saving functionality."""
    generator = DatasetGenerator()
    test_data = {"test": "data", "number": 42}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        tmp_path = tmp.name
    
    try:
        generator.save_dataset(test_data, tmp_path)
        
        # Verify the file was created and contains correct data
        with open(tmp_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data == test_data, "Saved data doesn't match original"
        print("✓ Dataset saving test passed")
        return True
    finally:
        os.unlink(tmp_path)


def test_api_integration_mock():
    """Test API integration with mocked response."""
    # Mock the Anthropic client
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"test_problem": "mock_data", "category": "test"}'
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        
        mock_client.messages.create.return_value = mock_response
        
        generator = DatasetGenerator("fake_api_key")
        result = generator.generate_dataset_entry("test prompt")
        
        assert "test_problem" in result, "Should parse JSON response correctly"
        assert generator.total_tokens_in == 100, "Should track input tokens"
        assert generator.total_tokens_out == 50, "Should track output tokens"
        print("✓ API integration mock test passed")
        return True


def run_all_tests():
    """Run all tests and report results."""
    print("Running dataset generator tests...")
    print("=" * 50)
    
    tests = [
        test_prompt_loading,
        test_billing_info,
        test_cost_estimation,
        test_dataset_saving,
        test_api_integration_mock
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)