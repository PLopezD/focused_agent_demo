"""
Main application entry point for the Music Store Customer Support Bot.
Includes LangSmith integration for monitoring and debugging.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from orchestrator import MusicStoreOrchestrator

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-store-support-bot"

def create_graph():
    """Create and return the LangGraph application for LangGraph Studio."""
    orchestrator = MusicStoreOrchestrator()
    return orchestrator.app

# For LangGraph Studio compatibility
graph = create_graph()


def generate_graph_image(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        Path("max_graph.png").write_bytes(png_bytes)
        print("✅ Graph image generated successfully")
    except Exception as e:
        print(f"⚠️  Graph image generation skipped due to: {e}")

# Safely try to generate graph image
generate_graph_image(graph)