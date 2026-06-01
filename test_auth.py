#!/usr/bin/env python3
"""
Quick test to verify the chatbot works without authentication.
"""

from orchestrator import MusicStoreOrchestrator

def test_unauthenticated_flow():
    """Test basic chatbot functionality without authentication."""
    orchestrator = MusicStoreOrchestrator()

    # Test 1: General greeting
    response1 = orchestrator.chat("Hello, I'm looking for some music recommendations", "test_session_1")
    print("=== Test 1: General greeting ===")
    print(f"User: Hello, I'm looking for some music recommendations")
    print(f"Bot: {response1}")
    print()

    # Test 2: Music question without auth
    response2 = orchestrator.chat("What are some good rock albums?", "test_session_2")
    print("=== Test 2: Music question without auth ===")
    print(f"User: What are some good rock albums?")
    print(f"Bot: {response2}")
    print()

    # Test 3: Account question without auth (should prompt for email)
    response3 = orchestrator.chat("I want to check my order status", "test_session_3")
    print("=== Test 3: Account question without auth ===")
    print(f"User: I want to check my order status")
    print(f"Bot: {response3}")
    print()

    # Test 4: Email authentication
    response4 = orchestrator.chat("My email is luisg@embraer.com.br", "test_session_4")
    print("=== Test 4: Email authentication ===")
    print(f"User: My email is luisg@embraer.com.br")
    print(f"Bot: {response4}")
    print()

if __name__ == "__main__":
    test_unauthenticated_flow()