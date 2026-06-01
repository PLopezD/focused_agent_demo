#!/usr/bin/env python3
"""
Test script for the enhanced TavilyAgent class.
"""

import asyncio
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"),
                    reason="API keys required for this test")
async def test_tavily_agent():
    """Test the TavilyAgent functionality."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.tavily_agent import TavilyAgent

    # Initialize the agent
    agent = TavilyAgent(
        model_name="gpt-3.5-turbo",  # Use a more reliable model for testing
        temperature=0.2,
        max_search_retries=2,
        max_search_results=5
    )

    # Test search functionality
    test_query = "What are the latest developments in artificial intelligence in 2024?"

    result = await agent.search(test_query)

    # Verify the result structure
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert result["messages"][-1].content is not None

    # Test context retrieval
    context = await agent.get_relevant_context(test_query, top_k=3)
    assert isinstance(context, list)

    # Clear context test
    agent.clear_context()
    # Verify context is cleared by checking empty context
    empty_context = await agent.get_relevant_context(test_query, top_k=3)
    assert len(empty_context) == 0

@pytest.mark.asyncio
async def test_rag_functionality():
    """Test the RAG functionality specifically."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.tavily_agent import TavilyAgent, SearchResult

    agent = TavilyAgent()

    # Create mock search results
    mock_results = [
        SearchResult(
            content="Artificial intelligence has seen significant advances in 2024, particularly in large language models.",
            title="AI Advances 2024",
            url="https://example.com/ai-2024"
        ),
        SearchResult(
            content="Machine learning techniques have improved dramatically with new transformer architectures.",
            title="ML Improvements",
            url="https://example.com/ml-2024"
        )
    ]

    # Test adding to RAG context
    rag_context = await agent.add_to_rag_context(mock_results, "AI developments 2024")

    # Verify RAG context structure
    assert rag_context is not None
    assert hasattr(rag_context, 'search_results')
    assert hasattr(rag_context, 'retrieved_docs')
    assert hasattr(rag_context, 'relevance_scores')

    assert len(rag_context.search_results) == 2
    assert len(rag_context.retrieved_docs) > 0
    assert len(rag_context.relevance_scores) > 0