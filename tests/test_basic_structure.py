#!/usr/bin/env python3
"""
Basic structure test for the TavilyAgent class - no API calls required.
"""

def test_imports():
    """Test that all imports work correctly."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.tavily_agent import TavilyAgent, SearchResult, RAGContext

    # Check that classes are properly imported
    assert TavilyAgent is not None
    assert SearchResult is not None
    assert RAGContext is not None

def test_class_structure():
    """Test class structure and method availability."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.tavily_agent import TavilyAgent

    # Check if the class has expected methods
    expected_methods = [
        'enhanced_search',
        'add_to_rag_context',
        'search',
        'get_relevant_context',
        'clear_context'
    ]

    for method_name in expected_methods:
        assert hasattr(TavilyAgent, method_name), f"Missing method: {method_name}"

def test_dataclasses():
    """Test that dataclasses are properly defined."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.tavily_agent import SearchResult, RAGContext

    # Test SearchResult
    search_result = SearchResult(
        content="Test content",
        title="Test title",
        url="https://test.com"
    )
    assert search_result.content == "Test content"
    assert search_result.title == "Test title"
    assert search_result.url == "https://test.com"

    # Test RAGContext
    rag_context = RAGContext(
        search_results=[search_result],
        retrieved_docs=["doc1", "doc2"],
        relevance_scores=[0.9, 0.8]
    )
    assert len(rag_context.search_results) == 1
    assert len(rag_context.retrieved_docs) == 2
    assert len(rag_context.relevance_scores) == 2