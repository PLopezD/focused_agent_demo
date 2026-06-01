"""
Order History Test - LLM as Judge Evaluation

This test reads the order history dataset from /tests/datasets/order_history_dataset.csv
and evaluates whether the last message in each conversation contains order history information
using an LLM as judge.
"""

import pandas as pd
import json
import pytest
import sys
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OrderHistoryLLMJudge:
    """LLM judge for evaluating order history information in responses."""

    def __init__(self):
        """Initialize the LLM judge with GPT-4o-mini for cost-effective evaluation."""
        try:
            self.judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM judge not available: {e}")
            self.judge_llm = None
            self.llm_available = False

    def evaluate_contains_order_info(self, message_content: str) -> Dict[str, Any]:
        """
        Use LLM as judge to determine if message contains order history information.

        Args:
            message_content: The message content to evaluate

        Returns:
            Dict containing evaluation results
        """

        judge_prompt = f"""You are evaluating whether a customer service response contains order history information.

Message to evaluate:
"{message_content}"

Determine if this message includes ANY of the following order/purchase information:
- Order numbers, invoice IDs, or transaction IDs
- Purchase amounts, prices, or totals (e.g., $12.99, $1.98)
- Purchase dates or order dates
- Product names, track names, album names, or items purchased
- Order summaries or purchase histories
- Any specific transaction details

Respond with ONLY a JSON object in this exact format:
{{"contains_order_info": true, "confidence": 0.95, "reasoning": "Message includes specific order numbers, purchase amounts, and product names"}}

Guidelines:
- contains_order_info: true if ANY order information is present, false if none
- confidence: 0.0 to 1.0 confidence level in your assessment
- reasoning: Brief explanation of your decision

Your response:"""

        if not self.llm_available or self.judge_llm is None:
            # Fallback: enhanced keyword check
            return self._fallback_evaluation(message_content)

        try:
            response = self.judge_llm.invoke([SystemMessage(content=judge_prompt)])
            result = json.loads(response.content.strip())

            return {
                "contains_order_info": result.get("contains_order_info", False),
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "raw_judge_response": response.content,
                "evaluation_method": "llm_judge"
            }

        except Exception as e:
            # Fallback: enhanced keyword check
            print(f"LLM judge failed, using fallback: {e}")
            return self._fallback_evaluation(message_content)

    def _fallback_evaluation(self, message_content: str) -> Dict[str, Any]:
        """Enhanced fallback evaluation using pattern matching."""

        # Strong indicators of order information
        strong_patterns = [
            r'order\s*#?\s*\d+',  # Order #123
            r'invoice\s*#?\s*\d+',  # Invoice #456
            r'\$\d+\.\d{2}',  # $12.99
            r'total:\s*\$',  # Total: $
            r'purchase.*date',  # Purchase date
            r'order.*date',  # Order date
        ]

        # Moderate indicators
        moderate_keywords = [
            "purchased", "bought", "track", "album", "song", "artist",
            "total", "amount", "items", "price"
        ]

        content_lower = message_content.lower()

        # Check for strong patterns
        import re
        strong_matches = sum(1 for pattern in strong_patterns if re.search(pattern, content_lower))

        # Check for moderate keywords
        moderate_matches = sum(1 for keyword in moderate_keywords if keyword in content_lower)

        # Scoring
        if strong_matches >= 2:
            confidence = 0.9
            contains_order_info = True
            reasoning = f"Strong order patterns found: {strong_matches} patterns, {moderate_matches} keywords"
        elif strong_matches >= 1 and moderate_matches >= 3:
            confidence = 0.8
            contains_order_info = True
            reasoning = f"Order patterns + keywords: {strong_matches} patterns, {moderate_matches} keywords"
        elif moderate_matches >= 5:
            confidence = 0.7
            contains_order_info = True
            reasoning = f"Multiple order keywords found: {moderate_matches} keywords"
        else:
            confidence = 0.3
            contains_order_info = False
            reasoning = f"Insufficient order indicators: {strong_matches} patterns, {moderate_matches} keywords"

        return {
            "contains_order_info": contains_order_info,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_judge_response": "",
            "evaluation_method": "pattern_matching"
        }


def load_order_history_dataset() -> pd.DataFrame:
    """Load the order history dataset from CSV file."""
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "order_history_dataset.csv"
    )

    if not os.path.exists(dataset_path):
        pytest.skip(f"Order history dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    return df


def extract_last_message(messages_json_str: str) -> str:
    """
    Extract the last AI message from the messages JSON string.

    Args:
        messages_json_str: JSON string containing conversation messages

    Returns:
        Content of the last AI message
    """
    try:
        messages = json.loads(messages_json_str)

        # Find the last AI message
        for message in reversed(messages):
            if message.get("type") == "ai":
                return message.get("content", "")

        return ""

    except Exception as e:
        print(f"Error extracting last message: {e}")
        return ""


@pytest.mark.integration
class TestOrderHistoryDataset:
    """Test order history functionality using dataset and LLM judge."""

    @pytest.fixture(scope="class")
    def llm_judge(self):
        """Create LLM judge instance."""
        return OrderHistoryLLMJudge()

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load the order history dataset."""
        return load_order_history_dataset()

    def test_order_history_dataset_evaluation(self, llm_judge, dataset):
        """
        Test that evaluates each conversation in the dataset to ensure
        the last message contains order history information.
        """

        print(f"\nEvaluating {len(dataset)} conversations from order history dataset...")

        results = []
        passed_count = 0

        for idx, row in dataset.iterrows():
            # Get customer info for context
            customer_email = row.get('customer_email', 'Unknown')

            # Use the second messages column which should contain the complete conversation with order history
            if 'messages.1' in dataset.columns:
                messages_column = row['messages.1']
            else:
                # Fallback to the first messages column
                messages_column = row['messages']

            last_message = extract_last_message(str(messages_column))

            if not last_message:
                print(f"\nRow {idx + 1} ({customer_email}): No AI message found - SKIP")
                continue

            # Evaluate with LLM judge
            evaluation = llm_judge.evaluate_contains_order_info(last_message)

            # Consider passed if contains order info with reasonable confidence
            passed = evaluation["contains_order_info"] and evaluation["confidence"] > 0.7

            if passed:
                passed_count += 1

            results.append({
                "row": idx + 1,
                "customer_email": customer_email,
                "passed": passed,
                "contains_order_info": evaluation["contains_order_info"],
                "confidence": evaluation["confidence"],
                "reasoning": evaluation["reasoning"],
                "last_message_preview": last_message[:200] + "..." if len(last_message) > 200 else last_message
            })

            # Print result for each conversation
            status = "PASS" if passed else "FAIL"
            print(f"\nRow {idx + 1} ({customer_email}): {status}")
            print(f"  Contains order info: {evaluation['contains_order_info']}")
            print(f"  Confidence: {evaluation['confidence']:.2f}")
            print(f"  Reasoning: {evaluation['reasoning']}")
            print(f"  Message preview: {last_message[:150]}...")

        # Calculate overall pass rate
        total_evaluated = len(results)
        pass_rate = passed_count / total_evaluated if total_evaluated > 0 else 0

        print(f"\n{'='*60}")
        print(f"ORDER HISTORY EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total conversations evaluated: {total_evaluated}")
        print(f"Passed (contains order info): {passed_count}")
        print(f"Failed (no order info): {total_evaluated - passed_count}")
        print(f"Pass rate: {pass_rate:.1%}")

        # Show failed cases for debugging
        failed_cases = [r for r in results if not r["passed"]]
        if failed_cases:
            print(f"\nFAILED CASES:")
            for case in failed_cases:
                print(f"- Row {case['row']} ({case['customer_email']}): {case['reasoning']}")

        # Assert that all conversations should contain order information
        # (since this is an order history dataset)
        assert pass_rate >= 0.9, (
            f"Order history dataset pass rate {pass_rate:.1%} too low. "
            f"Expected 90%+ of conversations to contain order information. "
            f"Failed: {total_evaluated - passed_count}/{total_evaluated}"
        )

    def test_dataset_structure(self, dataset):
        """Test that the dataset has the expected structure."""

        print(f"\nDataset structure validation:")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")

        # Check required columns exist
        assert 'customer_email' in dataset.columns, "Dataset missing customer_email column"
        assert 'messages' in dataset.columns, "Dataset missing messages column"

        # Check we have at least 3 conversations (as mentioned)
        assert len(dataset) >= 3, f"Expected at least 3 conversations, got {len(dataset)}"

        print(f" Dataset structure validation passed")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])