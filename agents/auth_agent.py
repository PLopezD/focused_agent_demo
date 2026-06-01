"""
Authentication Agent - Intelligent routing agent for authenticated users.
Uses LangGraph to decide which specialized agent should handle the request.
"""

from typing import Any, List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from helpers.pii_config import get_comprehensive_pii_middleware


class AuthRoutingState(BaseModel):
    """State for the authentication agent routing process."""
    messages: List[Any] = []
    customer_id: Optional[int] = None
    customer_email: Optional[str] = None
    authenticated: bool = False
    routing_decision: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


class AuthenticationAgent:
    """Authentication Agent that routes authenticated users to appropriate specialized agents."""

    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm
        self.authenticated_customer_id = None

        # Configure PII middleware for this agent
        self.pii_middleware = get_comprehensive_pii_middleware()
        self.fallback_mode = llm is None

        try:
            self.workflow = self._build_workflow()
        except Exception as e:
            print(f"Warning: AuthenticationAgent workflow failed to build: {e}")
            self.fallback_mode = True
            self.workflow = None

    def set_authenticated_customer(self, customer_id: int):
        """Set the authenticated customer ID for context-aware routing."""
        self.authenticated_customer_id = customer_id

    def _build_workflow(self):
        """Build the LangGraph workflow for intelligent agent routing."""

        def analyze_request(state: dict[str, Any]) -> dict[str, Any]:
            """Analyze the user's request to determine the best agent to handle it."""
            messages = state.get("messages", [])
            if not messages:
                state["routing_decision"] = "support"
                state["confidence"] = 0.5
                state["reasoning"] = "No message to analyze, defaulting to support"
                return state

            # Get the latest user message
            user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            if not user_messages:
                state["routing_decision"] = "support"
                state["confidence"] = 0.5
                state["reasoning"] = "No user message found, defaulting to support"
                return state

            latest_message = user_messages[-1].content

            # Use LLM to intelligently classify the request
            classification_prompt = f"""You are an intelligent routing agent for a digital music store. Analyze the user's request and determine which specialized agent should handle it.

User's request: "{latest_message}"

Available agents:
1. **music** - Music recommendations, discovery, search, preferences, genres, artists, albums
2. **transaction** - Order history, purchases, billing, invoices, refunds, payment issues, spending analysis
3. **support** - Account management, profile updates, technical support, escalation to human support

Guidelines:
- If the request mentions music, songs, albums, artists, genres, recommendations, or discovery → use "music"
- If the request mentions orders, purchases, billing, invoices, payments, refunds, or spending → use "transaction"
- If the request mentions account info, profile, support, help, problems, or escalation → use "support"
- For ambiguous requests, choose the agent most likely to help and set confidence accordingly
- Default to "support" for unclear or general requests

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any markdown code blocks, explanations, or other text. Only return the JSON object:

{{"agent": "music", "confidence": 0.9, "reasoning": "User is asking for music recommendations"}}

Your response:"""

            try:
                response = self.llm.invoke([SystemMessage(content=classification_prompt)])
                response_content = response.content.strip()

                # Parse the JSON response
                import json

                # Handle cases where the response might be wrapped in code blocks
                if response_content.startswith("```json"):
                    # Extract JSON from code block
                    json_start = response_content.find('{')
                    json_end = response_content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = response_content[json_start:json_end]
                    else:
                        raise ValueError("No JSON object found in code block")
                elif response_content.startswith('{') and response_content.endswith('}'):
                    json_content = response_content
                else:
                    # Try to find JSON object within the response
                    json_start = response_content.find('{')
                    json_end = response_content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = response_content[json_start:json_end]
                    else:
                        raise ValueError(f"No valid JSON object found in response: {response_content}")

                classification = json.loads(json_content)

                state["routing_decision"] = classification.get("agent", "support")
                state["confidence"] = float(classification.get("confidence", 0.5))
                state["reasoning"] = classification.get("reasoning", "LLM classification")

            except Exception as e:
                # Fallback to keyword-based classification
                print(f"LLM classification failed: {e}, using fallback")

                message_lower = latest_message.lower()

                # Music-related keywords
                music_keywords = [
                    'music', 'song', 'album', 'artist', 'genre', 'recommend', 'recommendation',
                    'jazz', 'rock', 'pop', 'classical', 'blues', 'hip hop', 'country',
                    'discover', 'new music', 'similar', 'like', 'style', 'band', 'track'
                ]

                # Transaction-related keywords
                transaction_keywords = [
                    'order', 'invoice', 'purchase', 'bought', 'buy', 'billing', 'payment',
                    'receipt', 'transaction', 'paid', 'refund', 'charge', 'history',
                    'spending', 'orders', 'purchases'
                ]

                # Support-related keywords
                support_keywords = [
                    'account', 'profile', 'information', 'update', 'change', 'help',
                    'support', 'problem', 'issue', 'contact', 'phone', 'address',
                    'email', 'password', 'login', 'escalate'
                ]

                music_score = sum(1 for keyword in music_keywords if keyword in message_lower)
                transaction_score = sum(1 for keyword in transaction_keywords if keyword in message_lower)
                support_score = sum(1 for keyword in support_keywords if keyword in message_lower)

                max_score = max(music_score, transaction_score, support_score)

                if max_score == 0:
                    # No keywords found, default to support
                    state["routing_decision"] = "support"
                    state["confidence"] = 0.3
                    state["reasoning"] = "No specific keywords found, defaulting to support"
                elif music_score == max_score:
                    state["routing_decision"] = "music"
                    state["confidence"] = min(0.9, 0.5 + (music_score * 0.1))
                    state["reasoning"] = f"Music keywords detected (score: {music_score})"
                elif transaction_score == max_score:
                    state["routing_decision"] = "transaction"
                    state["confidence"] = min(0.9, 0.5 + (transaction_score * 0.1))
                    state["reasoning"] = f"Transaction keywords detected (score: {transaction_score})"
                else:
                    state["routing_decision"] = "support"
                    state["confidence"] = min(0.9, 0.5 + (support_score * 0.1))
                    state["reasoning"] = f"Support keywords detected (score: {support_score})"

            return state

        def should_route_directly(state: dict[str, Any]) -> str:
            """Determine if we should route directly or need confirmation."""
            confidence = state.get("confidence", 0.0)

            # If confidence is high, route directly
            if confidence >= 0.7:
                return "route_to_agent"

            # If confidence is medium, still route but note uncertainty
            elif confidence >= 0.4:
                return "route_to_agent"

            # If confidence is low, default to support
            else:
                return "route_to_support"

        def route_to_agent(state: dict[str, Any]) -> dict[str, Any]:
            """Final routing decision completed."""
            # Add routing information to state for the orchestrator to use
            return state

        def route_to_support(state: dict[str, Any]) -> dict[str, Any]:
            """Route to support agent when uncertain."""
            state["routing_decision"] = "support"
            state["confidence"] = 0.8
            state["reasoning"] = "Low confidence in classification, routing to support for safety"
            return state

        # Build the workflow
        workflow = StateGraph(dict[str, Any])

        # Add nodes
        workflow.add_node("analyze_request", analyze_request)
        workflow.add_node("route_to_agent", route_to_agent)
        workflow.add_node("route_to_support", route_to_support)

        # Add edges
        workflow.set_entry_point("analyze_request")
        workflow.add_conditional_edges(
            "analyze_request",
            should_route_directly,
            {
                "route_to_agent": "route_to_agent",
                "route_to_support": "route_to_support"
            }
        )
        workflow.add_edge("route_to_agent", END)
        workflow.add_edge("route_to_support", END)

        return workflow.compile()

    def route_request(self, messages: List[Any], customer_id: int = None, customer_email: str = None) -> dict[str, Any]:
        """Route an authenticated user's request to the appropriate agent."""

        # If in fallback mode, use simple keyword-based routing
        if self.fallback_mode or self.workflow is None:
            return self._fallback_route(messages)

        try:
            # Set up initial state
            initial_state = {
                "messages": messages,
                "customer_id": customer_id or self.authenticated_customer_id,
                "customer_email": customer_email,
                "authenticated": True,
                "routing_decision": None,
                "confidence": 0.0,
                "reasoning": ""
            }

            # Execute the workflow
            result = self.workflow.invoke(initial_state)

            return {
                "agent": result.get("routing_decision", "support"),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", "LangGraph workflow routing")
            }
        except Exception as e:
            print(f"Workflow routing failed: {e}, falling back to keyword routing")
            return self._fallback_route(messages)

    def _fallback_route(self, messages: List[Any]) -> dict[str, Any]:
        """Fallback keyword-based routing when LLM is not available."""
        if not messages:
            return {
                "agent": "support",
                "confidence": 0.5,
                "reasoning": "No message to analyze, defaulting to support (fallback mode)"
            }

        # Get the latest user message
        user_messages = [msg for msg in messages if hasattr(msg, 'content')]
        if not user_messages:
            return {
                "agent": "support",
                "confidence": 0.5,
                "reasoning": "No user message found, defaulting to support (fallback mode)"
            }

        latest_message = user_messages[-1].content.lower()

        # Music-related keywords
        music_keywords = [
            'music', 'song', 'album', 'artist', 'genre', 'recommend', 'recommendation',
            'jazz', 'rock', 'pop', 'classical', 'blues', 'hip hop', 'country',
            'discover', 'new music', 'similar', 'like', 'style', 'band', 'track'
        ]

        # Transaction-related keywords
        transaction_keywords = [
            'order', 'invoice', 'purchase', 'bought', 'buy', 'billing', 'payment',
            'receipt', 'transaction', 'paid', 'refund', 'charge', 'history',
            'spending', 'orders', 'purchases'
        ]

        # Support-related keywords
        support_keywords = [
            'account', 'profile', 'information', 'update', 'change', 'help',
            'support', 'problem', 'issue', 'contact', 'phone', 'address',
            'email', 'password', 'login', 'escalate'
        ]

        music_score = sum(1 for keyword in music_keywords if keyword in latest_message)
        transaction_score = sum(1 for keyword in transaction_keywords if keyword in latest_message)
        support_score = sum(1 for keyword in support_keywords if keyword in latest_message)

        max_score = max(music_score, transaction_score, support_score)

        if max_score == 0:
            return {
                "agent": "support",
                "confidence": 0.3,
                "reasoning": "No specific keywords found, defaulting to support (fallback mode)"
            }
        elif music_score == max_score:
            return {
                "agent": "music",
                "confidence": min(0.8, 0.5 + (music_score * 0.1)),
                "reasoning": f"Music keywords detected (score: {music_score}) (fallback mode)"
            }
        elif transaction_score == max_score:
            return {
                "agent": "transaction",
                "confidence": min(0.8, 0.5 + (transaction_score * 0.1)),
                "reasoning": f"Transaction keywords detected (score: {transaction_score}) (fallback mode)"
            }
        else:
            return {
                "agent": "support",
                "confidence": min(0.8, 0.5 + (support_score * 0.1)),
                "reasoning": f"Support keywords detected (score: {support_score}) (fallback mode)"
            }

    def get_system_message(self) -> SystemMessage:
        """Get the system message for this agent."""
        return SystemMessage(content="""
You are an intelligent routing agent for a digital music store's customer support system.
Your job is to analyze authenticated customers' requests and determine which specialized agent should handle them.

You have access to three specialized agents:
1. **Music Agent** - Handles music recommendations, discovery, preferences, genres, artists
2. **Transaction Agent** - Handles order history, purchases, billing, invoices, refunds
3. **Support Agent** - Handles account management, technical support, general assistance

Analyze requests carefully and route them to the most appropriate agent based on the user's intent.
""")

    def get_pii_middleware(self):
        """Return the list of PII middleware for this agent."""
        return self.pii_middleware