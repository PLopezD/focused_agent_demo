"""
PII Configuration and Custom Detectors for Music Store Agents

This module provides custom PII detectors for phone numbers, zip codes, and addresses
to be used with LangChain's PIIMiddleware for protecting sensitive customer information.
"""

import re
from langchain.agents.middleware import PIIMiddleware

# Custom PII detection patterns
PII_PATTERNS = {
    "phone_number": [
        # US phone numbers in various formats
        r'\b(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b',
        # International phone numbers
        r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(\d{1,4}\)|\d{1,4})[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
    ],
    "zipcode": [
        # US ZIP codes (5 digits or 5+4 format)
        r'\b\d{5}(?:-\d{4})?\b',
        # Canadian postal codes
        r'\b[A-Za-z]\d[A-Za-z][-\s]?\d[A-Za-z]\d\b',
    ],
    "address": [
        # Street addresses with numbers and common street types
        r'\b\d+\s+(?:[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Circle|Cir|Court|Ct|Place|Pl)\.?)\b',
        # PO Box addresses
        r'\b(?:P\.?O\.?\s+)?Box\s+\d+\b',
        # Apartment/Unit indicators
        r'\b(?:Apt|Apartment|Unit|Suite|Ste)\.?\s+[A-Za-z0-9]+\b',
    ]
}

def create_pii_middleware():
    """
    Create a list of PIIMiddleware instances for phone numbers, zip codes, and addresses.
    Combines multiple patterns into single regex for each PII type to avoid duplicates.

    Returns:
        list: List of configured PIIMiddleware instances
    """
    middleware = []

    # Phone number detection - combine all patterns with OR operator
    phone_combined_pattern = '|'.join(f'({pattern})' for pattern in PII_PATTERNS["phone_number"])
    middleware.append(
        PIIMiddleware(
            pii_type="phone_number",
            strategy="mask",
            detector=phone_combined_pattern,
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    # ZIP code detection - combine all patterns with OR operator
    zipcode_combined_pattern = '|'.join(f'({pattern})' for pattern in PII_PATTERNS["zipcode"])
    middleware.append(
        PIIMiddleware(
            pii_type="zipcode",
            strategy="redact",
            detector=zipcode_combined_pattern,
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    # Address detection - combine all patterns with OR operator
    address_combined_pattern = '|'.join(f'({pattern})' for pattern in PII_PATTERNS["address"])
    middleware.append(
        PIIMiddleware(
            pii_type="address",
            strategy="redact",
            detector=address_combined_pattern,
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    return middleware

def get_comprehensive_pii_middleware():
    """
    Get comprehensive PII middleware including built-in types and custom detectors.

    Note: Excludes credit_card detection to avoid blocking legitimate financial information
    like order totals, prices, and transaction amounts in order history.

    Returns:
        list: List of all PIIMiddleware instances for comprehensive protection
    """
    middleware = []

    # Built-in PII types - EXCLUDING credit_card to avoid blocking cash values
    builtin_types = ["email", "ip", "mac_address", "url"]
    for pii_type in builtin_types:
        middleware.append(
            PIIMiddleware(
                pii_type=pii_type,
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            )
        )

    # Add custom PII detectors
    middleware.extend(create_pii_middleware())

    return middleware

def get_transaction_safe_pii_middleware():
    """
    Get minimal PII middleware specifically designed for transaction agents.
    This version only protects the most critical PII while allowing ALL financial data
    including prices, totals, and amounts to pass through unblocked.

    Returns:
        list: List of minimal PIIMiddleware instances for transaction processing
    """
    middleware = []

    # Only protect emails - the most sensitive PII that should never appear in transaction data
    middleware.append(
        PIIMiddleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    # Protect only explicit phone numbers with clear phone formatting
    # Use more restrictive patterns to avoid matching financial data
    phone_restrictive_patterns = [
        # Only match phone numbers with clear separators or parentheses
        r'\b(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[-.\s][2-9]\d{2}[-.\s]\d{4}\b',
        # International with + prefix
        r'\b\+\d{1,3}[-.\s]\d{1,4}[-.\s]\d{1,4}[-.\s]\d{1,9}\b',
    ]

    phone_pattern = '|'.join(f'({pattern})' for pattern in phone_restrictive_patterns)
    middleware.append(
        PIIMiddleware(
            pii_type="phone_number_strict",
            strategy="mask",
            detector=phone_pattern,
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    # NO zip code protection - these patterns are too broad and block financial amounts
    # NO address protection - addresses may contain numbers that look like prices
    # NO IP address protection - not relevant for transaction data
    # NO URL protection - transaction IDs or references may look like URLs

    return middleware

def get_minimal_transaction_pii_middleware():
    """
    Get absolutely minimal PII middleware for transaction agents.
    Only protects email addresses - nothing else.
    Use this if financial data is still being blocked.

    Returns:
        list: List with only email protection for transaction processing
    """
    middleware = []

    # Only protect emails - absolutely nothing else
    middleware.append(
        PIIMiddleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True
        )
    )

    return middleware

# Example usage patterns for testing
EXAMPLE_TEST_DATA = {
    "phone_numbers": [
        "Call me at 555-867-5309",
        "My number is (555) 123-4567",
        "Reach me at +1-555-234-5678",
        "Phone: 555.123.4567"
    ],
    "zipcodes": [
        "I live in 90210",
        "My ZIP is 12345-6789",
        "Postal code: K1A 0A6"
    ],
    "addresses": [
        "123 Main Street",
        "456 Oak Avenue, Apt 2B",
        "789 First Rd, Suite 100",
        "PO Box 567"
    ]
}