"""
Main Orchestrator using LangGraph for sophisticated agent routing and conversation management.
Implements customer authentication and secure agent coordination.
"""

import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from database import DatabaseManager
from agents.music_agent import MusicRecommendationAgent
from agents.transaction_agent import TransactionAgent
from agents.support_agent import CustomerSupportAgent
from agents.tavily_agent import TavilyAgent

class ConversationState(BaseModel):
    """State management for the conversation flow."""
    messages: List[BaseMessage] = []
    customer_id: Optional[int] = None
    customer_email: Optional[str] = None
    authenticated: bool = False
    current_agent: Optional[str] = None
    context: Dict[str, Any] = {}
    escalation_needed: bool = False

class MusicStoreOrchestrator:
    def __init__(self, use_memory: bool = False):
        self.use_memory = use_memory
        self.db = DatabaseManager()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Faster model for simple queries
        self.response_cache = {}  # Simple cache for common responses

        # Initialize specialized agents
        self.music_agent = MusicRecommendationAgent(self.db, self.llm)
        self.transaction_agent = TransactionAgent(self.db, self.llm)
        self.support_agent = CustomerSupportAgent(self.db, self.llm)
        self.tavily_agent = TavilyAgent(self.db, self.llm)

        # Build the conversation graph
        self.app = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine for conversation flow."""

        def check_authentication(state: ConversationState) -> ConversationState:
            """Check for email in message and authenticate if found."""
            # If already authenticated, skip authentication process
            if state.authenticated:
                return state

            latest_message = state.messages[-1] if state.messages else None
            if not latest_message or not isinstance(latest_message, HumanMessage):
                return state

            # Try to extract email from message using regex
            user_input = latest_message.content
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, user_input)

            if email_match:
                potential_email = email_match.group()
                customer = self.db.authenticate_customer(potential_email)
                if customer:
                    state.customer_id = customer['CustomerId']
                    state.customer_email = customer['Email']
                    state.authenticated = True
                    state.context['customer_info'] = customer

                    welcome_msg = f"Great! I've authenticated you as {customer['FirstName']} ({customer['Email']}). "
                    welcome_msg += "Now I can provide personalized assistance with your account, orders, and music recommendations. "
                    welcome_msg += "How can I help you today?"

                    state.messages.append(AIMessage(content=welcome_msg))
                    return state
                else:
                    auth_failure_msg = f"I couldn't find an account with email '{potential_email}'. "
                    auth_failure_msg += "You can still ask general questions about music, but I won't be able to access your account information."
                    state.messages.append(AIMessage(content=auth_failure_msg))
                    return state

            return state

        def send_welcome_message(state: ConversationState) -> ConversationState:
            """Send welcome message only when appropriate."""
            # Only send welcome if there are no messages yet (empty session initialization)
            if len(state.messages) == 0:
                welcome_msg = """Welcome to our Music Store Customer Support! 🎵

I can help you with:
• **General music questions** and recommendations
• **Account assistance** (provide your email for personalized help)
• **Order inquiries** (authentication required)
• **Technical support**

How can I assist you today?"""
                state.messages.append(AIMessage(content=welcome_msg))

            return state

        def should_continue_after_welcome(state: ConversationState) -> str:
            """Determine if we should continue processing or end after welcome."""
            # If we only have one message (the welcome message we just added), end here
            if len(state.messages) == 1 and isinstance(state.messages[0], AIMessage):
                return 'end'
            # If we have user messages, continue with processing
            return 'continue'

        def route_with_fast_path(state: ConversationState) -> str:
            """Check if we can use fast path, otherwise route to appropriate agent."""
            latest_message = state.messages[-1] if state.messages else None
            if not latest_message or not isinstance(latest_message, HumanMessage):
                if state.authenticated:
                    return 'authenticated_agent'
                else:
                    return 'tavily_rag_agent'

            message_lower = latest_message.content.lower()

            # Simple greetings and common responses
            simple_patterns = [
                'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
                'ok', 'okay', 'yes', 'no', 'help', 'what can you do',
                'features', 'capabilities', 'what do you do', 'show features'
            ]

            if any(pattern in message_lower for pattern in simple_patterns):
                print("routing to fast_response")
                return 'fast_response'

            # Short messages under 20 chars likely simple
            if len(latest_message.content.strip()) < 20:
                print("routing to fast_response")
                return 'fast_response'

            # Normal routing based on authentication
            if state.authenticated:
                return 'authenticated_agent'
            else:
                return 'tavily_agent'

        def handle_fast_response(state: ConversationState) -> ConversationState:
            """Handle simple responses with fast model and caching."""
            latest_message = state.messages[-1] if state.messages else None
            if not latest_message:
                return state

            message_lower = latest_message.content.lower().strip()

            # Check cache first
            if message_lower in self.response_cache:
                response = self.response_cache[message_lower]
                state.messages.append(AIMessage(content=response))
                return state

            # Pre-defined responses for common patterns (also cache these)
            if 'hello' in message_lower or 'hi' in message_lower:
                response = "Hello! How can I help you with your music needs today?"
            elif 'thank' in message_lower:
                response = "You're welcome! Is there anything else I can help you with?"
            elif 'bye' in message_lower or 'goodbye' in message_lower:
                response = "Goodbye! Feel free to come back if you need any music assistance!"
            elif any(word in message_lower for word in ['help', 'what can you do', 'features', 'capabilities', 'what do you do', 'show features']):
                response = """## 🎵 Music Store Support Bot Features

**🔐 Authentication & Account Management**
- Secure customer authentication via email
- Access to personalized account information
- Order history and billing support

**🎵 Music Recommendations**
- Personalized music suggestions based on preferences
- Genre-based recommendations (jazz, rock, classical, etc.)
- Artist and album discovery

**💳 Transaction Support**
- Order history lookup
- Invoice and billing inquiries
- Payment and purchase assistance

**🤝 Customer Support**
- General music store inquiries
- Technical support and troubleshooting
- Human escalation when needed

**🛡️ Advanced Architecture**
- Built with LangGraph for sophisticated conversation management
- LangSmith integration for monitoring and debugging
- Memory-enabled conversations for better context

How can I assist you today?"""
            else:
                # Use fast model for other simple queries
                try:
                    system_msg = "You are a helpful music store assistant. Give brief, friendly responses."
                    messages = [SystemMessage(content=system_msg), latest_message]
                    fast_response = self.fast_llm.invoke(messages)
                    response = fast_response.content
                except:
                    response = "I'd be happy to help! Could you please tell me more about what you need?"

            # Cache the response for future use (limit cache size)
            if len(self.response_cache) < 100:
                self.response_cache[message_lower] = response

            state.messages.append(AIMessage(content=response))
            return state


        def tavily_rag_agent(state: ConversationState) -> ConversationState:
            """Execute the tavily agent."""
            return self._execute_tavily_agent(state)

        def authenticated_agent(state: ConversationState) -> ConversationState:
            """Execute the authenticated agent."""
            return self._execute_authenticated_agent(state)

        # Build the graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("welcome", send_welcome_message)
        workflow.add_node("auth_check", check_authentication)
        workflow.add_node("fast_response", handle_fast_response)
        workflow.add_node("authenticated_agent", authenticated_agent)
        workflow.add_node("tavily_rag_agent", tavily_rag_agent)

        # Add edges - with fast path optimization
        workflow.set_entry_point("welcome")
        workflow.add_conditional_edges(
            "welcome",
            should_continue_after_welcome,
            {
                "end": END,
                "continue": "auth_check"
            }
        )
        workflow.add_conditional_edges(
            "auth_check",
            route_with_fast_path,
            {
                "fast_response": "fast_response",
                "authenticated_agent": "authenticated_agent",
                "tavily_rag_agent": "tavily_rag_agent"
            }
        )
        workflow.add_edge("fast_response", END)
        workflow.add_edge("authenticated_agent", END)
        workflow.add_edge("tavily_rag_agent", END)

        # Compile the workflow with memory for local execution
        if self.use_memory:
        # Note: When deployed to LangGraph API, persistence is handled automatically
            memory = MemorySaver()
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()


    def _execute_tavily_agent(self, state: ConversationState) -> ConversationState:
        """Execute the tavily agent."""
        try:
            # Convert ConversationState to GraphState format expected by TavilyAgent
            graph_state = {
                "messages": state.messages,
                "original_question": "",
                "attempted_search_queries": []
            }

            # Invoke the tavily agent
            result = self.tavily_agent.invoke(graph_state)

            # Update the conversation state with the results
            if isinstance(result, dict) and "messages" in result:
                # Only add new AI messages from the result
                new_messages = result["messages"]
                for msg in new_messages:
                    if hasattr(msg, 'type') and msg.type in ['ai', 'assistant']:
                        state.messages.append(msg)

            return state

        except Exception as e:
            from langchain_core.messages import AIMessage
            error_msg = f"I apologize, but I encountered an error while searching for information. Please try again. Error: {str(e)}"
            state.messages.append(AIMessage(content=error_msg))
            return state

    def _execute_music_agent(self, state: ConversationState) -> ConversationState:
        """Execute the music recommendation agent."""
        try:
            # Create agent executor with tools (without system_message parameter)
            agent_executor = create_react_agent(
                self.llm,
                self.music_agent.get_tools()
            )

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                state.messages.append(AIMessage(content="I didn't receive a valid message to process."))
                return state

            # Prepare input with customer context and system message
            messages = [
                self.music_agent.get_system_message(),
                SystemMessage(content=f"Customer ID: {state.customer_id if state.customer_id else 'Not authenticated'}"),
                user_messages[-1]  # Latest user message
            ]

            agent_input = {"messages": messages}

            # Execute agent
            result = agent_executor.invoke(agent_input)

            # Extract response
            if result.get("messages"):
                response = result["messages"][-1].content
                state.messages.append(AIMessage(content=response))

        except Exception as e:
            state.messages.append(AIMessage(content=f"I apologize, but I encountered an error while processing your music request. Please try again or contact support. Error: {str(e)}"))

        return state

    def _execute_transaction_agent(self, state: ConversationState) -> ConversationState:
        """Execute the transaction management agent."""
        try:
            # Create agent executor with tools
            agent_executor = create_react_agent(
                self.llm,
                self.transaction_agent.get_tools()
            )

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                state.messages.append(AIMessage(content="I didn't receive a valid message to process."))
                return state

            # Prepare input with customer context and system message
            messages = [
                self.transaction_agent.get_system_message(),
                SystemMessage(content=f"Customer ID: {state.customer_id}"),
                user_messages[-1]  # Latest user message
            ]

            agent_input = {"messages": messages}

            # Execute agent
            result = agent_executor.invoke(agent_input)

            # Extract response
            if result.get("messages"):
                response = result["messages"][-1].content
                state.messages.append(AIMessage(content=response))

        except Exception as e:
            state.messages.append(AIMessage(content=f"I apologize, but I encountered an error while processing your transaction request. Please try again or contact support. Error: {str(e)}"))

        return state

    def _execute_support_agent(self, state: ConversationState) -> ConversationState:
        """Execute the customer support agent using ToolNode."""
        try:
            # Create ToolNode with support agent tools
            tools = self.support_agent.get_tools()
            tool_node = ToolNode(tools=tools)

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                state.messages.append(AIMessage(content="I didn't receive a valid message to process."))
                return state

            # Prepare messages with system context and customer info
            messages = [
                self.support_agent.get_system_message(),
                SystemMessage(content=f"Customer ID: {state.customer_id if state.customer_id else 'Not authenticated'}"),
                user_messages[-1]  # Latest user message
            ]

            # Use LLM to process the request and decide if tools are needed
            llm_response = self.llm.bind_tools(tools).invoke(messages)

            # If the LLM wants to use tools, execute them
            if llm_response.tool_calls:
                # Add the LLM response to messages
                messages.append(llm_response)

                # Execute tools using ToolNode
                tool_state = {"messages": messages}
                tool_result = tool_node.invoke(tool_state)

                # Get the tool results and generate final response
                messages_with_tool_results = tool_result["messages"]
                final_response = self.llm.invoke(messages_with_tool_results)

                state.messages.append(AIMessage(content=final_response.content))
            else:
                # No tools needed, use the LLM response directly
                state.messages.append(AIMessage(content=llm_response.content))

            # Check for escalation triggers
            response_content = state.messages[-1].content
            if "escalat" in response_content.lower():
                state.escalation_needed = True

        except Exception as e:
            state.messages.append(AIMessage(content=f"I apologize, but I encountered an error while processing your support request. Please try again or let me escalate this to a human representative. Error: {str(e)}"))
            state.escalation_needed = True

        return state

    def _execute_authenticated_agent(self, state: ConversationState) -> ConversationState:
        """Execute the appropriate authenticated agent based on the user's request."""
        try:
            # Get the latest user message to determine routing
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                state.messages.append(AIMessage(content="I didn't receive a valid message to process."))
                return state

            latest_message = user_messages[-1].content.lower()

            # Route to transaction agent for order/billing/purchase related queries
            transaction_keywords = [
                'order', 'invoice', 'purchase', 'billing', 'payment', 'receipt',
                'transaction', 'bought', 'buy', 'paid', 'refund', 'charge',
                'history', 'previous orders', 'my orders'
            ]

            # Route to customer support agent for account/general support
            support_keywords = [
                'account', 'profile', 'information', 'update', 'change',
                'help', 'support', 'problem', 'issue', 'contact', 'phone',
                'address', 'email', 'password', 'login', 'escalate'
            ]

            # Check for transaction-related queries first
            if any(keyword in latest_message for keyword in transaction_keywords):
                return self._execute_transaction_agent(state)

            # Check for support-related queries
            elif any(keyword in latest_message for keyword in support_keywords):
                return self._execute_support_agent(state)

            # For ambiguous queries, use the support agent as default since it's more general
            else:
                return self._execute_support_agent(state)

        except Exception as e:
            state.messages.append(AIMessage(content=f"I apologize, but I encountered an error while processing your request. Please try again or contact support. Error: {str(e)}"))
            return state

    def _handle_escalation(self, state: ConversationState) -> ConversationState:
        """Handle escalation to human support."""
        state.messages.append(AIMessage(content="""
Your request has been escalated to our human support team. A customer service representative will be in touch with you shortly.

In the meantime, is there anything else I can help you with today?
"""))
        state.escalation_needed = False
        return state

    def initialize_session(self, session_id: str = "default") -> str:
        """Initialize a new session with welcome message."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            current_state = self.app.get_state(config)

            # Only send welcome if this is a truly new session
            if not current_state.values:
                # Just return the welcome message directly without invoking the full workflow
                welcome_msg = """Welcome to our Music Store Customer Support! 🎵

I can help you with:
• **General music questions** and recommendations
• **Account assistance** (provide your email for personalized help)
• **Order inquiries** (authentication required)
• **Technical support**

How can I assist you today?"""
                return welcome_msg

            # Session already exists, return empty string
            return ""

        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again or contact support directly. Error: {str(e)}"

    def chat(self, message: str, session_id: str = "default") -> str:
        """Main chat interface."""
        try:
            # Get current state
            config = {"configurable": {"thread_id": session_id}}
            current_state = self.app.get_state(config)

            # Create new state if none exists, but don't send welcome here
            if not current_state.values:
                state = ConversationState()
            else:
                state = ConversationState(**current_state.values)

            # Add user message
            state.messages.append(HumanMessage(content=message))
            # Process through the graph
            result = self.app.invoke(state, config)
            # Extract the final state (result should be dict with our state fields)
            if isinstance(result, dict):
                final_state = ConversationState(**result)
                ai_messages = [msg for msg in final_state.messages if isinstance(msg, AIMessage)]
                return ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't process your request."
            else:
                return "I'm sorry, I couldn't process your request."

        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again or contact support directly. Error: {str(e)}"

    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.app.get_state(config)

            if not state.values:
                return []

            history = []
            for msg in state.values.get('messages', []):
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})

            return history

        except Exception as e:
            return [{"role": "error", "content": f"Error retrieving history: {str(e)}"}]

    def get_authentication_status(self, session_id: str = "default") -> Dict[str, Any]:
        """Get current authentication status for a session."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.app.get_state(config)

            if not state.values:
                return {"authenticated": False, "customer_email": None, "customer_id": None}

            conversation_state = ConversationState(**state.values)
            return {
                "authenticated": conversation_state.authenticated,
                "customer_email": conversation_state.customer_email,
                "customer_id": conversation_state.customer_id
            }

        except Exception as e:
            return {"authenticated": False, "customer_email": None, "customer_id": None, "error": str(e)}