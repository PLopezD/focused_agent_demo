"""
Main application entry point for the Music Store Customer Support Bot.
Includes LangSmith integration for monitoring and debugging.
"""

import os
import streamlit as st
from random import randrange
from dotenv import load_dotenv
from orchestrator import MusicStoreOrchestrator
from helpers.system_messages import SYSTEM_MESSAGES
import uuid

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-store-support-bot"

def main():
    st.set_page_config(
        page_title="Music Store Support Agent",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 Music Store Customer Support Agent")

    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        try:
            st.session_state.orchestrator = MusicStoreOrchestrator(use_memory=True)
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.error("Please check your environment variables and API keys.")
            st.stop()

    # Initialize session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(randrange(10000, 100000))

    # Check authentication status and display logged-in user
    auth_status = st.session_state.orchestrator.get_authentication_status(st.session_state.session_id)
    if auth_status["authenticated"] and auth_status["customer_email"]:
        st.success(f"🔐 Logged in as: **{auth_status['customer_email']}**")
    else:
        st.info("👤 Not authenticated - provide your email to access personalized features")

    # Sidebar with system info
    with st.sidebar:
        st.header("🏗️ System Architecture")
        st.markdown("""
        **Main Orchestrator** (LangGraph)
        - Customer Authentication Layer
        - Music Recommendation Agent
        - Transaction Management Agent
        - Customer Support Agent
        - Tavily RAG Agent (When not authenticated)

        **Key Features:**
        - 🔐 Secure customer authentication
        - 🔒 PII Redaction Middleware for user privacy
        - 🎵 Personalized music recommendations
        - 📋 Order history and billing support
        - 📊 Full LangSmith monitoring
        """)

        st.header("🧪 Demo Instructions")
        st.markdown("""
        1. **Authenticate** with a customer email
        2. **Try different queries:**
           - "Recommend music like jazz"
           - "Show my order history"
           - "Help with my account"
        3. **View** conversation flow in LangSmith Studio
        """)

        st.header("📊 Sample Customers")
        st.markdown("""
        - `luisg@embraer.com.br`
        - `leonekohler@surfeu.de`
        - `ftremblay@gmail.com`
        - `bjorn.hansen@yahoo.no`
        - `frantisekw@jetbrains.com`
        """)

    # Always display welcome message as the first message
    with st.chat_message("assistant"):
        st.write(SYSTEM_MESSAGES["WELCOME"])

    # Display conversation history
    history = st.session_state.orchestrator.get_conversation_history(st.session_state.session_id)

    if history:
        for exchange in history:
            if exchange["role"] == "user":
                with st.chat_message("user"):
                    st.write(exchange["content"])
            elif exchange["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(exchange["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Check authentication status before the message
        auth_before = st.session_state.orchestrator.get_authentication_status(st.session_state.session_id)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.orchestrator.chat(prompt, st.session_state.session_id)
                st.write(response)

        # Check if authentication status changed and trigger rerun
        auth_after = st.session_state.orchestrator.get_authentication_status(st.session_state.session_id)
        if auth_before["authenticated"] != auth_after["authenticated"]:
            if auth_after["authenticated"]:
                # User just got authenticated - show success message and rerun
                st.success(f"✅ Successfully authenticated as {auth_after['customer_email']}!")
                st.balloons()  # Celebration animation
            # Authentication status changed, rerun to update UI immediately
            st.rerun()

    # Clear conversation button with dynamic text based on authentication
    if auth_status["authenticated"]:
        button_text = "🗑️ Clear Conversation (Logout)"
    else:
        button_text = "🗑️ Clear Conversation"

    if st.button(button_text):
        # Reset session
        st.session_state.session_id = str(randrange(10000, 100000))
        st.rerun()



if __name__ == "__main__":
    main()

