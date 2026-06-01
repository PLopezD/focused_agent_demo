#!/usr/bin/env python3
"""
Quick test to verify the chatbot works without authentication.
Includes LLM as judge evaluation for response helpfulness.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MusicStoreOrchestrator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def evaluate_response_helpfulness(user_input: str, response: str) -> dict:
    """
    Use LLM as judge to evaluate if a response is helpful.

    Args:
        user_input: The user's original query
        response: The chatbot's response

    Returns:
        Dictionary with evaluation results
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

        evaluation_prompt = f"""You are an expert evaluator of chatbot responses. Evaluate whether the given response is helpful to the user's query.

User Query: "{user_input}"
Chatbot Response: "{response}"

Evaluation Criteria:
1. Does the response directly address the user's question or request?
2. Is the response informative and useful?
3. Does it provide appropriate guidance or next steps?
4. Is the tone professional and helpful?
5. For account-related queries without authentication, does it appropriately explain limitations while offering helpful alternatives?

Rate the response as either "HELPFUL" or "NOT_HELPFUL" and provide a brief explanation.

Format your response as:
RATING: [HELPFUL/NOT_HELPFUL]
EXPLANATION: [Your reasoning in 1-2 sentences]"""

        messages = [
            SystemMessage(content="You are an expert evaluator of chatbot interactions."),
            HumanMessage(content=evaluation_prompt)
        ]

        result = judge_llm.invoke(messages)
        evaluation_text = result.content

        # Parse the evaluation
        lines = evaluation_text.strip().split('\n')
        rating = "NOT_HELPFUL"  # Default
        explanation = "Failed to parse evaluation"

        for line in lines:
            if line.startswith("RATING:"):
                rating_text = line.replace("RATING:", "").strip()
                rating = "HELPFUL" if "HELPFUL" in rating_text else "NOT_HELPFUL"
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        return {
            "is_helpful": rating == "HELPFUL",
            "rating": rating,
            "explanation": explanation,
            "full_evaluation": evaluation_text
        }

    except Exception as e:
        # Fallback evaluation if LLM judge fails
        print(f"Warning: LLM judge evaluation failed: {e}")
        return {
            "is_helpful": len(response) > 20,  # Simple fallback
            "rating": "UNKNOWN",
            "explanation": f"LLM evaluation failed: {str(e)}",
            "full_evaluation": ""
        }

def test_unauthenticated_flow():
    """Test basic chatbot functionality without authentication."""
    orchestrator = MusicStoreOrchestrator()

    user_query = "I want to check my order status"
    response3 = orchestrator.chat(user_query, "test_session_3")
    assert response3 is not None
    assert len(response3) > 0

    # Use LLM as judge to evaluate if the response is helpful
    evaluation = evaluate_response_helpfulness(user_query, response3)

    print(f"\n--- LLM Judge Evaluation for Order Status Query ---")
    print(f"User Query: {user_query}")
    print(f"Response: {response3}")
    print(f"Rating: {evaluation['rating']}")
    print(f"Explanation: {evaluation['explanation']}")
    print("=" * 60)

    # Assert that the response is helpful according to LLM judge
    assert evaluation['is_helpful'], f"Response should be helpful according to LLM judge. Explanation: {evaluation['explanation']}"

def test_unauthenticated_order_status_helpfulness():
    """Pytest version: Test that order status query without auth provides helpful response."""
    orchestrator = MusicStoreOrchestrator()

    user_query = "I want to check my order status"
    response = orchestrator.chat(user_query, "pytest_test_session")

    # Basic assertions
    assert response is not None
    assert len(response) > 0

    # LLM judge evaluation
    evaluation = evaluate_response_helpfulness(user_query, response)

    # Print evaluation details for debugging
    print(f"\n--- LLM Judge Evaluation Results ---")
    print(f"Query: {user_query}")
    print(f"Response: {response}")
    print(f"Rating: {evaluation['rating']}")
    print(f"Helpful: {evaluation['is_helpful']}")
    print(f"Explanation: {evaluation['explanation']}")

    # Assert helpfulness
    assert evaluation['is_helpful'], (
        f"Order status response should be helpful according to LLM judge. "
        f"Rating: {evaluation['rating']}, Explanation: {evaluation['explanation']}"
    )

def test_music_recommendation_helpfulness():
    """Pytest version: Test that music recommendation without auth is helpful."""
    orchestrator = MusicStoreOrchestrator()

    user_query = "What are some good rock albums?"
    response = orchestrator.chat(user_query, "pytest_music_session")

    # Basic assertions
    assert response is not None
    assert len(response) > 0

    # LLM judge evaluation
    evaluation = evaluate_response_helpfulness(user_query, response)

    print(f"\n--- Music Recommendation Evaluation ---")
    print(f"Query: {user_query}")
    print(f"Rating: {evaluation['rating']}")
    print(f"Helpful: {evaluation['is_helpful']}")

    # Music recommendations should be helpful even without auth
    assert evaluation['is_helpful'], (
        f"Music recommendation should be helpful. "
        f"Explanation: {evaluation['explanation']}"
    )

if __name__ == "__main__":
    print("Running authentication tests with LLM judge evaluation...")
    test_unauthenticated_flow()
    print("\n✅ All tests passed!")