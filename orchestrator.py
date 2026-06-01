"""
Main Orchestrator using LangGraph for sophisticated agent routing and conversation management.
Implements customer authentication and secure agent coordination.
"""

import re
from typing import Any, List, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from database import DatabaseManager
from helpers.system_messages import SYSTEM_MESSAGES
from agents.music_agent import MusicRecommendationAgent
from agents.transaction_agent import TransactionAgent
from agents.support_agent import CustomerSupportAgent
from agents.tavily_agent import TavilyAgent
from agents.auth_agent import AuthenticationAgent

class ConversationState(BaseModel):
    """State management for the conversation flow."""
    messages: List[BaseMessage] = []
    customer_id: Optional[int] = None
    customer_email: Optional[str] = None
    authenticated: bool = False
    current_agent: Optional[str] = None
    context: dict[str, Any] = {}
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
        self.tavily_agent = TavilyAgent()

        # Initialize authentication agent with error handling
        try:
            self.auth_agent = AuthenticationAgent(self.llm)
        except Exception as e:
            print(f"Warning: AuthenticationAgent failed to initialize: {e}")
            print("Using fallback mode for routing")
            self.auth_agent = AuthenticationAgent(llm=None)  # Fallback mode

        # Build the conversation graph
        self.app = self._build_graph()

    def _conversation_state_to_dict(self, conv_state: ConversationState) -> dict:
        """Convert ConversationState to dictionary format for LangGraph."""
        return {
            "messages": conv_state.messages,
            "customer_id": conv_state.customer_id,
            "customer_email": conv_state.customer_email,
            "authenticated": conv_state.authenticated,
            "current_agent": conv_state.current_agent,
            "context": conv_state.context,
            "escalation_needed": conv_state.escalation_needed
        }

    def _add_error_message(self, state: ConversationState, agent_type: str, error: Exception) -> ConversationState:
        """Add standardized error message to conversation state."""
        error_messages = SYSTEM_MESSAGES["ERROR_MESSAGES"]

        error_msg = error_messages.get(agent_type, error_messages["general"])
        full_error_msg = f"{error_msg} Error: {str(error)}"

        state.messages.append(AIMessage(content=full_error_msg))

        # Set escalation flag for support errors
        if agent_type in ["support", "general"]:
            state.escalation_needed = True

        return state

    def _add_no_message_error(self, state: ConversationState) -> ConversationState:
        """Add standardized 'no message received' error."""
        state.messages.append(AIMessage(content=SYSTEM_MESSAGES["NO_MESSAGE_RECEIVED"]))
        return state

    def _format_error_message(self, agent_type: str, error: Exception) -> str:
        """Format standardized error message for string returns."""
        base_msg = SYSTEM_MESSAGES["GENERAL_ERROR"]
        return f"{base_msg} Error: {str(error)}"

    def _create_customer_context_message(self, state: ConversationState) -> SystemMessage:
        """Create standardized customer context system message."""
        customer_id = state.customer_id if state.customer_id else 'Not authenticated'
        return SystemMessage(content=f"Customer ID: {customer_id}")

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
                    state.context["customer_info"] = customer

                    welcome_msg = SYSTEM_MESSAGES["AUTHENTICATION_SUCCESS"].format(
                        first_name=customer['FirstName'],
                        email=customer['Email']
                    )

                    state.messages.append(AIMessage(content=welcome_msg))
                    return state
                else:
                    auth_failure_msg = SYSTEM_MESSAGES["AUTHENTICATION_FAILED"].format(email=potential_email)
                    state.messages.append(AIMessage(content=auth_failure_msg))
                    return state

            return state

        def send_welcome_message(state: ConversationState) -> ConversationState:
            """Send welcome message only when appropriate."""
            # Only send welcome if there are no messages yet (empty session initialization)
            if len(state.messages) == 0:
                welcome_msg = SYSTEM_MESSAGES["WELCOME"]
                state.messages.append(AIMessage(content=welcome_msg))

            return state

        def should_continue_after_welcome(state: ConversationState) -> str:
            """Determine if we should continue processing or end after welcome."""
            # If we only have one message (the welcome message we just added), end here
            messages = state.messages
            if len(messages) == 1 and isinstance(messages[0], AIMessage):
                return 'end'
            # If we have user messages, continue with processing
            return 'continue'

        def route_with_fast_path(state: ConversationState) -> str:
            """Check if we can use fast path, otherwise route to appropriate agent."""
            messages = state.messages
            latest_message = messages[-1] if messages else None
            if not latest_message or not isinstance(latest_message, HumanMessage):
                if state.authenticated:
                    return 'authenticated_agent'
                else:
                    return 'tavily_rag_agent'

            message_lower = latest_message.content.lower()

            # Check for account-related queries that should NEVER go to fast response when authenticated
            account_related_patterns = [
                'my account', 'account info', 'my profile', 'my orders', 'order history',
                'my purchases', 'my invoices', 'my spending', 'my transactions',
                'my support rep', 'support representative', 'escalate', 'billing'
            ]

            # If authenticated and asking about account-related things, route to authenticated agent
            if state.authenticated and any(pattern in message_lower for pattern in account_related_patterns):
                print("routing to authenticated_agent (account-related)")
                return 'authenticated_agent'

            # Simple greetings and common responses
            simple_patterns = [
                'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
                'ok', 'okay', 'yes', 'no', 'help', 'what can you do',
                'features', 'capabilities', 'what do you do', 'show features'
            ]

            if any(pattern in message_lower for pattern in simple_patterns):
                print("routing to fast_response")
                return 'fast_response'

            # Short messages under 20 chars likely simple (but not if account-related)
            if len(latest_message.content.strip()) < 20 and not any(pattern in message_lower for pattern in account_related_patterns):
                print("routing to fast_response")
                return 'fast_response'

            # Normal routing based on authentication
            if state.authenticated:
                return 'authenticated_agent'
            else:
                return 'tavily_rag_agent'

        def handle_fast_response(state: ConversationState) -> ConversationState:
            """Handle simple responses with fast model and caching, considering authentication state."""
            messages = state.messages
            latest_message = messages[-1] if messages else None
            if not latest_message:
                return state

            message_lower = latest_message.content.lower().strip()

            # Create cache key that includes authentication status
            auth_prefix = "auth_" if state.authenticated else "unauth_"
            cache_key = f"{auth_prefix}{message_lower}"

            # Check cache first (with authentication context)
            if cache_key in self.response_cache:
                response = self.response_cache[cache_key]
                state.messages.append(AIMessage(content=response))
                return state

            try:
                if state.authenticated:
                    customer_info = state.context.get('customer_info', {})
                    customer_name = customer_info.get('FirstName', '')
                    customer_email = state.customer_email or ""
                    system_msg = f"""You are a helpful music store assistant. The customer is authenticated as {customer_name} ({customer_email}).
                    You can help with account-related questions, personalized recommendations, and order inquiries.
                    Give brief, friendly, personalized responses."""
                else:
                    system_msg = """You are a helpful music store assistant. The customer is not authenticated yet.
                    You can help with general music questions, but suggest they provide their email for personalized assistance.
                    Give brief, friendly responses."""

                # Include recent conversation history for better context
                context_messages = [SystemMessage(content=system_msg)]

                # Add recent conversation history (last 3 messages for context)
                recent_messages = messages[-3:] if len(messages) > 1 else [latest_message]
                context_messages.extend(recent_messages)

                fast_response = self.fast_llm.invoke(context_messages)
                response = fast_response.content
            except Exception as e:
                if state.authenticated:
                    response = "I'd be happy to help with your account! Could you please tell me more about what you need?"
                else:
                    response = "I'd be happy to help! Could you please tell me more about what you need? (Provide your email for personalized account assistance)"

            # Cache the response with authentication context (limit cache size)
            if len(self.response_cache) < 200:  # Increased cache size to account for auth variants
                self.response_cache[cache_key] = response

            state.messages.append(AIMessage(content=response))
            return state


        def tavily_rag_agent(state: ConversationState) -> ConversationState:
            """Execute the tavily agent."""
            result_state = self._execute_tavily_agent(state)
            return result_state

        def authenticated_agent(state: ConversationState) -> ConversationState:
            """Execute the authenticated agent."""
            result_state = self._execute_authenticated_agent(state)
            return result_state

        # Build the graph - use dict state for better serialization
        
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
            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                return self._add_no_message_error(state)

            latest_message = user_messages[-1].content

            # Use the enhanced TavilyAgent search method
            result = self.tavily_agent.search(latest_message)

            # The search method returns a string response directly
            if isinstance(result, str) and result.strip():
                state.messages.append(AIMessage(content=result))
            else:
                state.messages.append(AIMessage(content=SYSTEM_MESSAGES["SEARCH_NO_CLEAR_RESPONSE"]))

            return state

        except Exception as e:
            return self._add_error_message(state, "tavily", e)

    def _execute_music_agent(self, state: ConversationState) -> ConversationState:
        """Execute the music recommendation agent."""
        try:
            # Set authenticated customer context if available
            if state.authenticated and state.customer_id:
                self.music_agent.set_authenticated_customer(state.customer_id)

            # Create agent executor with tools and PII middleware
            agent_executor = create_agent(
                model=self.llm,
                tools=self.music_agent.get_tools(),
                middleware=self.music_agent.get_pii_middleware()
            )

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                return self._add_no_message_error(state)

            # Prepare input with system message and customer context
            messages = [
                self.music_agent.get_system_message(),
                self._create_customer_context_message(state),
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
            return self._add_error_message(state, "music", e)

        return state

    def _execute_transaction_agent(self, state: ConversationState) -> ConversationState:
        """Execute the transaction management agent."""
        try:
            # Set authenticated customer context if available
            if state.authenticated and state.customer_id:
                self.transaction_agent.set_authenticated_customer(state.customer_id)

            # Create agent executor with tools and PII middleware
            agent_executor = create_agent(
                model=self.llm,
                tools=self.transaction_agent.get_tools(),
                middleware=self.transaction_agent.get_pii_middleware()
            )

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                return self._add_no_message_error(state)

            # Prepare input with system message and customer context
            messages = [
                self.transaction_agent.get_system_message(),
                self._create_customer_context_message(state),
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
            return self._add_error_message(state, "transaction", e)

        return state

    def _execute_support_agent(self, state: ConversationState) -> ConversationState:
        """Execute the customer support agent."""
        try:
            # Set authenticated customer context if available
            if state.authenticated and state.customer_id:
                self.support_agent.set_authenticated_customer(state.customer_id)

            # Create agent executor with tools and PII middleware
            agent_executor = create_agent(
                model=self.llm,
                tools=self.support_agent.get_tools(),
                middleware=self.support_agent.get_pii_middleware()
            )

            # Get the latest user message
            user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                return self._add_no_message_error(state)

            # Prepare input with system message and customer context
            messages = [
                self.support_agent.get_system_message(),
                self._create_customer_context_message(state),
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
            return self._add_error_message(state, "support", e)

        return state

    def _execute_authenticated_agent(self, state: ConversationState) -> ConversationState:
        """Execute the appropriate authenticated agent based on intelligent routing."""
        try:
            # Set authenticated customer context for the auth agent
            if state.authenticated and state.customer_id:
                self.auth_agent.set_authenticated_customer(state.customer_id)

            # Use the authentication agent to determine routing
            routing_result = self.auth_agent.route_request(
                messages=state.messages,
                customer_id=state.customer_id,
                customer_email=state.customer_email
            )

            agent_choice = routing_result.get("agent", "support")
            confidence = routing_result.get("confidence", 0.0)
            reasoning = routing_result.get("reasoning", "No reasoning provided")

            # Log the routing decision for debugging
            print(f"Auth agent routing: {agent_choice} (confidence: {confidence:.2f}) - {reasoning}")

            # Route to the appropriate specialized agent
            if agent_choice == "music":
                return self._execute_music_agent(state)
            elif agent_choice == "transaction":
                return self._execute_transaction_agent(state)
            else:  # Default to support for "support" or any unknown choice
                return self._execute_support_agent(state)

        except Exception as e:
            print(f"Error in authenticated agent routing: {e}")
            # Fallback to support agent
            return self._execute_support_agent(state)

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
                welcome_msg = SYSTEM_MESSAGES["WELCOME"]
                return welcome_msg

            # Session already exists, return empty string
            return ""

        except Exception as e:
            return self._format_error_message("general", e)

    def chat(self, message: str, session_id: str = "default") -> str:
        """Main chat interface."""
        try:
            config = {"configurable": {"thread_id": session_id}}

            # Handle different memory modes
            if self.use_memory:
                # Get current state from checkpoint
                current_state = self.app.get_state(config)
                if current_state.values:
                    state_values = current_state.values
                    state = ConversationState(
                        messages=state_values.get('messages', []),
                        customer_id=state_values.get('customer_id'),
                        customer_email=state_values.get('customer_email'),
                        authenticated=state_values.get('authenticated', False),
                        current_agent=state_values.get('current_agent'),
                        context=state_values.get('context', {}),
                        escalation_needed=state_values.get('escalation_needed', False)
                    )
                else:
                    state = ConversationState()
            else:
                # No memory mode - create fresh state
                state = ConversationState()

            # Add user message
            state.messages.append(HumanMessage(content=message))

            # Convert state to dict for LangGraph
            state_dict = {
                "messages": state.messages,
                "customer_id": state.customer_id,
                "customer_email": state.customer_email,
                "authenticated": state.authenticated,
                "current_agent": state.current_agent,
                "context": state.context,
                "escalation_needed": state.escalation_needed
            }

            # Process through the graph
            if self.use_memory:
                result = self.app.invoke(state_dict, config)
            else:
                result = self.app.invoke(state_dict)

            # Extract the final state
            if isinstance(result, dict) and "messages" in result:
                ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                return ai_messages[-1].content if ai_messages else SYSTEM_MESSAGES["REQUEST_PROCESSING_ERROR"]
            else:
                return SYSTEM_MESSAGES["REQUEST_PROCESSING_ERROR"]

        except Exception as e:
            return self._format_error_message("general", e)

    def get_conversation_history(self, session_id: str = "default") -> List[dict[str, str]]:
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

    def get_authentication_status(self, session_id: str = "default") -> dict[str, Any]:
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