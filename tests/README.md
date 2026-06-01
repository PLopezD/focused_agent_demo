# PII Redaction Tests

This directory contains comprehensive tests for validating PII redaction middleware in the music store application.

## 🚀 Running Tests with Pytest

### Basic Commands

```bash
# Run all PII redaction tests
pytest tests/test_pii_redaction.py

# Run all tests with verbose output
pytest tests/test_pii_redaction.py -v

# Run all tests in the tests directory
pytest tests/

# Show which tests would run without executing them
pytest tests/test_pii_redaction.py --collect-only
```

### Running Specific Test Classes

```bash
# Test PII detection logic only
pytest tests/test_pii_redaction.py::TestPIIDetection -v

# Test PII middleware functionality only
pytest tests/test_pii_redaction.py::TestPIIMiddleware -v

# Test orchestrator PII redaction
pytest tests/test_pii_redaction.py::TestOrchestratorPIIRedaction -v

# Test PII pattern detection
pytest tests/test_pii_redaction.py::TestPIIPatterns -v
```

### Running Specific Tests

```bash
# Test PII detection on raw data
pytest tests/test_pii_redaction.py::TestPIIDetection::test_pii_detection_on_raw_data -v

# Test support agent middleware
pytest tests/test_pii_redaction.py::TestPIIMiddleware::test_support_agent_with_pii_middleware -v

# Test phone number pattern detection
pytest tests/test_pii_redaction.py::TestPIIPatterns::test_phone_pattern_detection -v
```

### Advanced Options

```bash
# Run tests and stop on first failure
pytest tests/test_pii_redaction.py -x

# Run tests with detailed output and timing
pytest tests/test_pii_redaction.py -v -s --tb=long

# Run only fast tests (skip slow integration tests)
pytest tests/test_pii_redaction.py -m "not slow"

# Run tests matching a pattern
pytest tests/test_pii_redaction.py -k "middleware" -v

# Run with coverage report
pytest tests/test_pii_redaction.py --cov=helpers.pii_config --cov-report=term-missing
```

### Parametrized Tests

The test suite includes parametrized tests that run the same test with different inputs:

```bash
# Run tests for multiple customers
pytest tests/test_pii_redaction.py::TestPIIMiddleware::test_pii_middleware_multiple_customers -v

# Run phone pattern tests with different numbers
pytest tests/test_pii_redaction.py::TestPIIPatterns::test_phone_pattern_detection -v

# Run zipcode pattern tests
pytest tests/test_pii_redaction.py::TestPIIPatterns::test_zipcode_pattern_detection -v
```

## 📋 Test Categories

### 1. **PII Detection Tests** (`TestPIIDetection`)
- ✅ Validates PII detection logic works correctly
- ✅ Tests detection on raw customer data
- ✅ Tests detection on properly redacted data
- ✅ Tests redaction marker detection

### 2. **PII Middleware Tests** (`TestPIIMiddleware`)
- ✅ Tests support agent with PII middleware
- ✅ Tests multiple customers with parametrized tests
- ✅ Validates middleware integration with agents

### 3. **Orchestrator Tests** (`TestOrchestratorPIIRedaction`)
- ✅ Tests full orchestrator flow with authentication
- ✅ Tests multiple customer emails
- ✅ Tests unauthenticated users don't get PII data

### 4. **Pattern Tests** (`TestPIIPatterns`)
- ✅ Tests phone number detection patterns
- ✅ Tests zip code detection patterns
- ✅ Tests edge cases and false positives

## 🧪 Test Structure

Each test uses fixtures for:
- `pii_validator`: PII redaction validation logic
- `sample_customer_data`: Sample data with PII for testing
- `orchestrator`: MusicStoreOrchestrator instance
- `real_customer_emails`: Real emails from database

## 📊 Example Output

```bash
$ pytest tests/test_pii_redaction.py::TestPIIDetection -v

============================= test session starts ==============================
collecting ... collected 4 items

tests/test_pii_redaction.py::TestPIIDetection::test_pii_detection_on_raw_data PASSED [25%]
tests/test_pii_redaction.py::TestPIIDetection::test_pii_detection_on_redacted_data PASSED [50%]
tests/test_pii_redaction.py::TestPIIDetection::test_middleware_redacted_data PASSED [75%]
tests/test_pii_redaction.py::TestPIIDetection::test_redaction_markers_detection PASSED [100%]

============================== 4 passed in 0.12s ========================
```

## 🛠 Troubleshooting

**If tests fail due to API keys:**
- Ensure `OPENAI_API_KEY` environment variable is set
- Some tests will skip gracefully if API is unavailable

**If tests are slow:**
- Use `-m "not slow"` to skip integration tests
- Run specific test classes instead of all tests

**For debugging:**
- Use `-s` flag to see print statements
- Use `--tb=long` for detailed traceback information
- Use `-x` to stop on first failure