import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_HELPFULNESS_PROMPT,
)


@dataclass
class SearchResult:
    content: str
    title: str
    url: str
    score: float = 0.0


@dataclass
class RAGContext:
    search_results: List[SearchResult]
    retrieved_docs: List[str]
    relevance_scores: List[float]


class TavilyAgent:
    def __init__(
        self,
        model_name: str = "ollama:qwen2.5:7b",
        temperature: float = 0.2,
        max_search_retries: int = 3,
        max_search_results: int = 10,
        embeddings: Optional[Embeddings] = None
    ):
        try:
            self.model = init_chat_model(model_name, temperature=temperature)
        except Exception:
            # Fallback to OpenAI if the specified model is not available
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.max_search_retries = max_search_retries
        self.max_search_results = max_search_results
        self.current_date = datetime.now().strftime("%A, %B %d, %Y")

        # Initialize embeddings for RAG
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vector_store = InMemoryVectorStore(self.embeddings)

        # Initialize evaluators
        self.relevance_evaluator = create_async_llm_as_judge(
            judge=self.model,
            prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {self.current_date}.",
            feedback_key="retrieval_relevance",
        )

        self.helpfulness_evaluator = create_async_llm_as_judge(
            judge=self.model,
            prompt=RAG_HELPFULNESS_PROMPT + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {self.current_date}.',
            feedback_key="helpfulness",
        )

        # Initialize search tool
        self.search_tool = self._create_search_tool()
        self.model_with_tools = self.model.bind_tools([self.search_tool])

        # Build workflow
        self.agent = self._build_workflow()

    def _create_search_tool(self):
        @tool
        async def search_tool(query: str):
            """Search the web for information relevant to the query."""
            print(f"Searching the web for information relevant to the query: {query}")
            search_api = TavilySearch(max_results=self.max_search_results)
            results = await search_api.ainvoke({"query": query})
            return results

        return search_tool


    async def enhanced_search(self, query: str) -> List[SearchResult]:
        """Enhanced search with vector similarity and relevance filtering."""
        search_api = TavilySearch(max_results=self.max_search_results)
        raw_results = await search_api.ainvoke({"query": query})

        search_results = []
        if raw_results.get("results"):
            for result in raw_results["results"]:
                search_result = SearchResult(
                    content=result.get("content", ""),
                    title=result.get("title", ""),
                    url=result.get("url", "")
                )
                search_results.append(search_result)

        return search_results

    async def add_to_rag_context(self, search_results: List[SearchResult], query: str) -> RAGContext:
        """Add search results to vector store and retrieve relevant documents."""
        # Add documents to vector store
        docs = []
        metadata = []
        for result in search_results:
            if result.content:
                docs.append(result.content)
                metadata.append({
                    "title": result.title,
                    "url": result.url,
                    "query": query
                })

        if docs:
            await self.vector_store.aadd_texts(docs, metadatas=metadata)

        # Retrieve most relevant documents
        retrieved_docs = await self.vector_store.asimilarity_search(
            query, k=min(5, len(docs)) if docs else 0
        )

        relevance_scores = []
        retrieved_content = []

        for doc in retrieved_docs:
            # Evaluate relevance of each document
            eval_result = await self.relevance_evaluator(
                inputs=query, context=doc.page_content
            )
            relevance_scores.append(eval_result.get("score", 0))
            retrieved_content.append(doc.page_content)
        
        print(f"Retrieved content: {retrieved_content}")
        return RAGContext(
            search_results=search_results,
            retrieved_docs=retrieved_content,
            relevance_scores=relevance_scores
        )

    def _build_workflow(self):
        """Build the LangGraph workflow for the agent."""

        class GraphState(MessagesState):
            original_question: str
            attempted_search_queries: list[str]
            rag_context: Optional[RAGContext] = None

        SYSTEM_PROMPT = f"""
You are a helpful assistant that uses web search to find accurate and up-to-date information.
Use the provided web search tool when you need to find current information or verify facts.
When you have search results, use them to provide comprehensive and well-informed answers.

Current date: {self.current_date}
"""

        async def relevance_filter(state: GraphState):
            last_message = state["messages"][-1]
            if last_message.type == "tool" and last_message.name == self.search_tool.name:
                search_results = json.loads(last_message.content).get("results", [])
                filtered_results = []

                # Create a semaphore to limit concurrent tasks to 2
                semaphore = asyncio.Semaphore(2)

                async def evaluate_with_semaphore(result):
                    async with semaphore:
                        eval_result = await self.relevance_evaluator(
                            inputs=state["attempted_search_queries"][-1], context=result.get("content", "")
                        )
                        return result, eval_result

                # Create tasks for all results
                tasks = [evaluate_with_semaphore(result) for result in search_results]

                # Process tasks as they complete
                for completed_task in asyncio.as_completed(tasks):
                    result, eval_result = await completed_task
                    if eval_result["score"]:
                        filtered_results.append(result)

                # Enhance with RAG
                if filtered_results:
                    search_objs = [
                        SearchResult(
                            content=r.get("content", ""),
                            title=r.get("title", ""),
                            url=r.get("url", "")
                        ) for r in filtered_results
                    ]
                    rag_context = await self.add_to_rag_context(search_objs, state["attempted_search_queries"][-1])

                    # Update message with enhanced context
                    enhanced_content = {
                        "results": filtered_results,
                        "rag_context": {
                            "retrieved_docs": rag_context.retrieved_docs,
                            "relevance_scores": rag_context.relevance_scores
                        }
                    }
                    last_message.content = json.dumps(enhanced_content)

                    return {
                        "messages": [last_message],
                        "rag_context": rag_context
                    }
                else:
                    last_message.content = json.dumps({"results": []})
                    return {"messages": [last_message]}
            else:
                raise Exception("Relevance filter node must be called after web search")

        async def should_continue(state: GraphState):
            if len(state["attempted_search_queries"]) > self.max_search_retries:
                return END
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "web_search"
            return "reflect"

        async def call_model(state: GraphState):
            # Enhanced system prompt with RAG context
            system_content = SYSTEM_PROMPT
            if state.get("rag_context") and state["rag_context"].retrieved_docs:
                system_content += "\n\nRelevant context from previous searches:\n"
                for i, doc in enumerate(state["rag_context"].retrieved_docs):
                    system_content += f"\n{i+1}. {doc[:500]}...\n"

            messages = [{"role": "system", "content": system_content}] + state["messages"]
            response = await self.model_with_tools.ainvoke(messages)

            if response.tool_calls and response.tool_calls[0]["name"] == self.search_tool.name:
                search_query = response.tool_calls[0]["args"]["query"]
                return {
                    "messages": [response],
                    "attempted_search_queries": state["attempted_search_queries"] + [search_query],
                }
            return {"messages": [response]}

        async def web_search(state: GraphState):
            last_message = state["messages"][-1]
            tool_call = last_message.tool_calls[0]
            search_results = await self.search_tool.ainvoke(tool_call)

            # Extract content from search results
            if hasattr(search_results, 'content'):
                content = search_results.content
            elif isinstance(search_results, dict):
                content = json.dumps(search_results)
            elif isinstance(search_results, str):
                content = search_results
            else:
                content = str(search_results)

            # Create proper ToolMessage to respond to the tool call
            tool_message = ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=self.search_tool.name
            )
            return {"messages": [tool_message]}

        async def reflect(state: GraphState):
            last_message = state["messages"][-1]
            helpfulness_eval_result = await self.helpfulness_evaluator(
                inputs=state["original_question"], outputs=last_message.content
            )
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
{helpfulness_eval_result.get('comment', 'Answer lacks sufficient detail or accuracy')}
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
            return {}

        async def retry_or_end(state: GraphState):
            if state["messages"][-1].type == "human":
                return "agent"
            return END

        # Build the workflow
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
        workflow.add_conditional_edges("reflect", retry_or_end, ["agent", END])

        return workflow.compile()

    async def search(self, query: str) -> Dict[str, Any]:
        """Main search method with RAG enhancement."""
        message = HumanMessage(content=query)
        result = await self.agent.ainvoke({"messages": [message]})
        return result

    async def get_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Get relevant context from vector store."""
        docs = await self.vector_store.asimilarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

    def clear_context(self):
        """Clear the RAG context/vector store."""
        self.vector_store = InMemoryVectorStore(self.embeddings)