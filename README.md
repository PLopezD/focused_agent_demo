# <� LangChain Music Store Customer Support Bot

A sophisticated customer support application built with LangGraph, featuring specialized agents for music recommendations, transaction management, and customer support with comprehensive PII protection.

## =� Quick Start with Docker

### Build and Run the Application

```bash
# Build the Docker image
docker build -t langchain-music-store .

# Run the application
docker run -p 8501:8501 langchain-music-store
```

The application will be available at `http://localhost:8501`

### Run Tests 
```bash
# All
pytest tests/ -v

# Specific
pytest tests/test_order_history.py -v

```

## <� System Architecture

### Main Components
- **Customer Authentication Layer** - Secure email-based authentication
- **Music Recommendation Agent** - Personalized music discovery and recommendations
- **Transaction Management Agent** - Order history and billing support
- **Customer Support Agent** - General account assistance and escalation
- **Tavily RAG Agent** - Web search for unauthenticated users

### Key Features
- = Secure customer authentication
- = PII Redaction Middleware for user privacy
- <� Personalized music recommendations
- =� Order history and billing support
- =� Full LangSmith monitoring

## <� Agent Graph Visualizations

### Music Recommendation Agent
![Music Agent Graph](music_agent_graph.png)

**Notes:**
_[Space for your notes about the music agent workflow and decision points]_

---

### Tavily RAG Agent
![Tavily Agent Graph](tavily_graph.png)

**Notes:**
_[Space for your notes about the tavily agent search and retrieval process]_

---

## =� Sample Test Customers

Use these customer emails for testing:
- `luisg@embraer.com.br`
- `leonekohler@surfeu.de`
- `ftremblay@gmail.com`
- `bjorn.hansen@yahoo.no`
- `frantisekw@jetbrains.com`

## =� Usage Examples

1. **Authentication**: Enter a customer email to authenticate
2. **Music Recommendations**: "Recommend music like jazz" or "Find albums similar to The Beatles"
3. **Order History**: "Show me my order history" or "What did I purchase last month?"
4. **General Support**: "Help with my account" or "Update my profile information"

## =' Development

### Environment Setup
The application requires the following environment variables:
- `OPENAI_API_KEY` - For LLM functionality
- `TAVILY_API_KEY` - For web search capabilities
- `LANGCHAIN_TRACING_V2=true` - For LangSmith monitoring
- `LANGCHAIN_PROJECT=music-store-support-bot` - LangSmith project name

### PII Protection
The system includes comprehensive PII middleware that:
- Protects customer emails, phone numbers, and addresses
- Uses minimal protection for transaction agents to preserve financial data
- Allows order totals and purchase amounts to display properly

## =� Testing and Quality Assurance

### Order History Validation
- Uses LLM-as-judge evaluation to ensure order information is properly displayed
- Validates that PII middleware doesn't block legitimate financial data
- Tests against real conversation datasets

### PII Redaction Testing
- Comprehensive tests for phone number, address, and email protection
- Validates appropriate redaction across different agent types
- Ensures financial data remains visible for transaction history