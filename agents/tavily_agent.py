import json
import asyncio
from datetime import datetime
from typing import List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_HELPFULNESS_PROMPT,
)
from helpers.pii_config import get_comprehensive_pii_middleware

def get_model():
    """Get or create the ChatOpenAI model instance."""
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0.1)
    except Exception:
        # Fallback for when API keys are not available
        return None

current_date = datetime.now().strftime("%A, %B %d, %Y")

MAX_SEARCH_RETRIES = 2


class GraphState(MessagesState):
    original_question: str
    attempted_search_queries: list[str]


def get_evaluators():
    """Get or create evaluator instances with proper error handling."""
    model = get_model()
    if model is None:
        return None, None
    try:
        relevance_evaluator = create_async_llm_as_judge(
            judge=model,
            prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {current_date}.",
            feedback_key="retrieval_relevance",
        )

        helpfulness_evaluator = create_async_llm_as_judge(
            judge=model,
            prompt=RAG_HELPFULNESS_PROMPT
            + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {current_date}.',
            feedback_key="helpfulness",
        )

        return relevance_evaluator, helpfulness_evaluator
    except Exception:
        return None, None


SYSTEM_PROMPT = """
Use the provided web search tool to find the latest information if you are not sure of what the user is asking for.
"""


# Simplify the Tavily search tool's input schema for a small local model
@tool
async def search_tool(query: str):
    """Search the web for information relevant to the query."""
    try:
        return await TavilySearch(max_results=10).ainvoke({"query": query})
    except Exception as e:
        return {"results": [], "error": str(e)}


def get_model_with_tools():
    """Get model with tools bound, with fallback handling."""
    model = get_model()
    if model:
        return model.bind_tools([search_tool])
    return None


async def relevance_filter(state: GraphState):
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == search_tool.name:
        search_results = json.loads(last_message.content).get("results", [])
        filtered_results = []

        # Get evaluators
        relevance_evaluator, _ = get_evaluators()

        if relevance_evaluator is None:
            # If evaluators not available, return all results without filtering
            return {"messages": [last_message]}

        # Create a semaphore to limit concurrent tasks to 2
        semaphore = asyncio.Semaphore(2)

        async def evaluate_with_semaphore(result):
            async with semaphore:
                try:
                    eval_result = await relevance_evaluator(
                        inputs=state["attempted_search_queries"][-1], context=result
                    )
                    return result, eval_result
                except Exception:
                    # If evaluation fails, include the result anyway
                    return result, {"score": True}

        # Create tasks for all results
        tasks = [evaluate_with_semaphore(result) for result in search_results]

        # Process tasks as they complete
        for completed_task in asyncio.as_completed(tasks):
            result, eval_result = await completed_task
            if eval_result["score"]:
                filtered_results.append(result)
        last_message.content = json.dumps({"results": filtered_results})
        return {"messages": [last_message]}
    else:
        raise Exception(f"Relevance filter node must be called after web search")


async def should_continue(state: GraphState):
    if len(state["attempted_search_queries"]) > MAX_SEARCH_RETRIES:
        return END
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "web_search"
    return "reflect"


async def call_model    (state: GraphState):
    model_with_tools = get_model_with_tools()
    if model_with_tools is None:
        # Fallback response when model is not available
        return {"messages": [AIMessage(content="I'm sorry, but I'm unable to process your request due to missing API credentials.")]}

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        response = await model_with_tools.ainvoke(messages)
        if response.tool_calls and response.tool_calls[0]["name"] == search_tool.name:
            search_query = response.tool_calls[0]["args"]["query"]
            return {
                "messages": [response],
                "attempted_search_queries": state["attempted_search_queries"]
                + [search_query],
            }
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"I encountered an error: {str(e)}")]}


async def web_search(state: GraphState):
    last_message = state["messages"][-1]
    search_results = await search_tool.ainvoke(last_message.tool_calls[0])
    return {"messages": [search_results]}


async def reflect(state: GraphState):
    last_message = state["messages"][-1]

    # Get evaluators
    _, helpfulness_evaluator = get_evaluators()

    if helpfulness_evaluator is None:
        # If evaluator not available, assume the response is helpful
        return {}

    try:
        helpfulness_eval_result = await helpfulness_evaluator(
            inputs=state["original_question"], outputs=last_message.content
        )
        print(f"Helpfulness eval result: {helpfulness_eval_result}")
        if not helpfulness_eval_result["score"]:
            return {
                "messages": [
                    HumanMessage(content=f"""
I originally asked you the following question:

<original_question>
{state["original_question"]}
</original_question>

Your answer was not helpful for the following reason:

<reason>
{helpfulness_eval_result.get('comment', 'The response did not adequately address the question')}
</reason>

Please check the conversation history carefully and try again. You may choose to fetch more information if you think the answer
to the original question is not somewhere in the conversation, but carefully consider if the answer is already in the conversation.

You have already attempted to answer the original question using the following search queries,
so if you choose to search again, you must rephrase your search query to be different from the ones below to avoid fetching redundant information:

<attempted_search_queries>
{state['attempted_search_queries']}
</attempted_search_queries>

As a reminder, check the previous conversation history and fetched context carefully before searching again!
""")
                ]
            }
    except Exception:
        # If evaluation fails, assume the response is helpful
        pass

    return {}


async def retry_or_end(state: GraphState):
    if state["messages"][-1].type == "human":
        return "agent"
    return END


workflow = StateGraph(GraphState, input=MessagesState, output=MessagesState)

workflow.add_node(
    "store_original_question",
    lambda state: {
        "original_question": state["messages"][-1].content,
        "attempted_search_queries": [],
    },
)
workflow.add_node("agent", call_model)
workflow.add_node("web_search", web_search)
workflow.add_node("relevance_filter", relevance_filter)
workflow.add_node("reflect", reflect)

workflow.add_edge(START, "store_original_question")
workflow.add_edge("store_original_question", "agent")
workflow.add_conditional_edges("agent", should_continue, ["web_search", "reflect", END])
workflow.add_edge("web_search", "relevance_filter")
workflow.add_edge("relevance_filter", "agent")
workflow.add_conditional_edges(
    "reflect",
    retry_or_end,
    ["agent", END],
)

agent = workflow.compile()

class TavilyAgent:
    """
    Enhanced Tavily Agent with RAG capabilities, relevance filtering, and self-reflection.
    Uses a sophisticated workflow that includes web search, relevance evaluation, and helpfulness assessment.
    """

    def __init__(self, llm: ChatOpenAI = None):
        """Initialize the TavilyAgent with optional LLM override."""
        self.llm = llm or get_model()
        self.workflow = agent

        # Configure PII middleware for this agent
        self.pii_middleware = get_comprehensive_pii_middleware()

    async def search_async(self, message: str) -> str:
        """
        Async search method that uses the full RAG workflow with relevance filtering and reflection.

        Args:
            message: User's search query/question

        Returns:
            Processed response after going through relevance filtering and reflection
        """
        try:
            # Format the message for the workflow
            formatted_input = {
                "messages": [HumanMessage(content=message)]
            }

            # Run the workflow
            result = await self.workflow.ainvoke(formatted_input)

            # Extract the final response
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    return final_message.content
                elif isinstance(final_message, dict) and 'content' in final_message:
                    return final_message['content']

            return "I apologize, but I couldn't find relevant information for your query."

        except Exception as e:
            return f"I encountered an error while searching: {str(e)}"

    def search(self, message: str) -> str:
        """
        Synchronous search method that wraps the async workflow.

        Args:
            message: User's search query/question

        Returns:
            Processed response after going through the full RAG workflow
        """
        try:
            # Run the async workflow in a new event loop if none exists
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop is None:
                # No event loop running, create a new one
                return asyncio.run(self.search_async(message))
            else:
                # Event loop is already running, need to handle differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.search_async(message))
                    return future.result(timeout=30)  # 30 second timeout

        except Exception as e:
            return f"I encountered an error while processing your search: {str(e)}"

    def get_system_message(self):
        """Get the system message for this agent."""
        return """You are an intelligent web search agent with RAG capabilities. You help users by:

1. **Web Search**: Finding the most current information on the web
2. **Relevance Filtering**: Evaluating search results for relevance to the user's question
3. **Self-Reflection**: Assessing if your answers are helpful and improving them
4. **Retry Logic**: Re-searching with different queries if initial results aren't helpful

Your workflow includes:
- Initial web search using optimized queries
- Relevance evaluation of search results using LLM judges
- Helpfulness assessment of generated answers
- Automatic retry with refined queries if needed (up to 3 attempts)

Always strive to provide accurate, current, and helpful information based on reliable web sources."""

    def get_pii_middleware(self):
        """Return the list of PII middleware for this agent."""
        return self.pii_middleware