#!/usr/bin/env python3
"""
Quick test to verify the chatbot works without authentication.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MusicStoreOrchestrator

def test_unauthenticated_flow():
    """Test basic chatbot functionality without authentication."""
    orchestrator = MusicStoreOrchestrator()

    # Test 1: General greeting
    response1 = orchestrator.chat("Hello, I'm looking for some music recommendations", "test_session_1")
    assert response1 is not None
    assert len(response1) > 0

    # Test 2: Music question without auth
    response2 = orchestrator.chat("What are some good rock albums?", "test_session_2")
    assert response2 is not None
    assert len(response2) > 0

    # Test 3: Account question without auth (should provide helpful response)
    response3 = orchestrator.chat("I want to check my order status", "test_session_3")
    assert response3 is not None
    assert len(response3) > 0
    # Should provide some reasonable response about order status

    # Test 4: Email authentication
    response4 = orchestrator.chat("My email is luisg@embraer.com.br", "test_session_4")
    assert response4 is not None
    assert len(response4) > 0