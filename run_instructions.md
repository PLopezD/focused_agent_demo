# 🎵 Music Store Support Bot - Setup & Run Instructions

## Quick Start

### 1. Set up API Keys
Edit the `.env` file and add your API keys:

```bash
# Required: Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required for LangSmith monitoring (get free key at smith.langchain.com)
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-api-key-here

# Already configured
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="music-store-support-bot"
DATABASE_PATH=./chinook.db
```

### 2. Run the Application

```bash
# Make sure you're in the project directory
cd /Users/pablolopezdomowicz/Desktop/inters/langchain_take_home_v1_codex

# Run the Streamlit app
streamlit run main.py
```

### 3. Test the Bot

The app will open in your browser at `http://localhost:8501`

**Sample Customer Emails for Testing:**
- `luisg@embraer.com.br` (Brazil customer)
- `leonekohler@surfeu.de` (Germany customer)
- `ftremblay@gmail.com` (Canada customer)
- `bjorn.hansen@yahoo.no` (Norway customer)
- `frantisekw@jetbrains.com` (Czech Republic customer)

**Sample Queries to Try:**
1. **Authentication:** "My email is luisg@embraer.com.br"
2. **Music Recommendations:** "Can you recommend some jazz music for me?"
3. **Order History:** "Show me my recent purchases"
4. **Account Support:** "I need help with my billing address"

## Demo Flow for Presentation

### 1. **Business Context** (5 mins)
- Music store needs 24/7 customer support
- Personalized recommendations drive sales
- Reduce human support workload

### 2. **Architecture Demo** (10 mins)
- Show LangGraph state machine in code
- Explain agent specialization (Music, Transaction, Support)
- Demonstrate customer data isolation

### 3. **Live Demo** (15 mins)
- Authenticate customer
- Music recommendations based on purchase history
- Order lookup with security validation
- Show LangSmith monitoring (if available)

### 4. **LangSmith Features** (5 mins)
- Real-time trace monitoring
- Agent decision visualization
- Performance analytics
- Debugging capabilities

## LangSmith Studio Setup

1. **Get LangSmith API Key:**
   - Sign up at https://smith.langchain.com
   - Create a new project: "music-store-support-bot"
   - Copy API key to `.env` file

2. **View Monitoring:**
   - Visit https://smith.langchain.com
   - Select your project
   - Watch real-time traces as you interact with the bot

## Architecture Highlights

**🏗️ LangGraph Orchestrator:**
- State machine manages conversation flow
- Intelligent agent routing based on intent
- Built-in error handling and escalation

**🔐 Security Features:**
- Customer authentication required
- SQL-level data isolation
- Invoice ownership verification

**🎵 Specialized Agents:**
- **Music Agent:** Personalized recommendations, search
- **Transaction Agent:** Order history, billing support
- **Support Agent:** Account management, escalation

**📊 Production Ready:**
- LangSmith monitoring integration
- Conversation state persistence
- Graceful error handling

## Troubleshooting

**Missing API Keys:**
- App will show errors if OpenAI or LangSmith keys are missing
- Get OpenAI key: https://platform.openai.com/api-keys
- Get LangSmith key: https://smith.langchain.com

**Database Issues:**
- Database `chinook.db` should be in the project root
- Contains 59 customers, 3,503 tracks, 412 invoices

**Import Errors:**
- Make sure all packages are installed: `pip install -r requirements.txt`
- Ensure you're using Python 3.8+