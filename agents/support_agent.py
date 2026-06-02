"""
Customer Support Agent - General support, account management, and escalation handling.
"""

from typing import Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from database import DatabaseManager
from helpers.system_messages import SYSTEM_MESSAGES
from helpers.pii_config import get_comprehensive_pii_middleware

class CustomerSupportAgent:
    def __init__(self, db_manager: DatabaseManager, llm: ChatOpenAI):
        self.db = db_manager
        self.llm = llm
        self.authenticated_customer_id = None  # Will be set when processing authenticated requests

        # Configure PII middleware for this agent
        self.pii_middleware = get_comprehensive_pii_middleware()

        # Create tool functions with closure over self.db
        @tool
        def get_account_info() -> str:
            """Get your account information and profile details."""
            if not self.authenticated_customer_id:
                return SYSTEM_MESSAGES["AUTH_REQUIRED_ACCOUNT_INFO"]

            # Get customer info using authenticated customer ID
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT CustomerId, FirstName, LastName, Email, Company,
                           Address, City, State, Country, PostalCode, Phone
                    FROM Customer
                    WHERE CustomerId = ?
                """, (self.authenticated_customer_id,))

                customer = cursor.fetchone()
                if not customer:
                    return SYSTEM_MESSAGES["CUSTOMER_NOT_FOUND"]

                result = "**Your Account Information:**\n\n"
                result += f"Name: {customer['FirstName']} {customer['LastName']}\n"
                result += f"Email: {customer['Email']}\n"

                if customer['Company']:
                    result += f"Company: {customer['Company']}\n"

                result += "\n**Contact Information:**\n"
                # Note: Address information removed for privacy protection
                result += f"Country: {customer['Country']}\n"

                if customer['Phone']:
                    result += f"Phone: {customer['Phone']}\n"

                return result

        @tool
        def get_support_rep_contact() -> str:
            """Get your assigned support representative contact information."""
            if not self.authenticated_customer_id:
                return SYSTEM_MESSAGES["AUTH_REQUIRED_SUPPORT_REP"]

            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT e.FirstName, e.LastName, e.Email, e.Phone, e.Title
                    FROM Customer c
                    JOIN Employee e ON c.SupportRepId = e.EmployeeId
                    WHERE c.CustomerId = ?
                """, (self.authenticated_customer_id,))

                rep = cursor.fetchone()
                if not rep:
                    return SYSTEM_MESSAGES["NO_SUPPORT_REP"]

                result = "**Your Assigned Support Representative:**\n\n"
                result += f"Name: {rep['FirstName']} {rep['LastName']}\n"
                result += f"Title: {rep['Title']}\n"
                result += f"Email: {rep['Email']}\n"
                if rep['Phone']:
                    result += f"Phone: {rep['Phone']}\n"

                result += "\nFor complex issues that require human assistance, you can contact your dedicated support representative directly."
                return result

        @tool
        def escalate_to_human(issue_description: str) -> str:
            """Escalate customer issue to human support representative."""
            if not self.authenticated_customer_id:
                return SYSTEM_MESSAGES["AUTH_REQUIRED_ESCALATION"]

            # Get support rep info
            rep_info = self.db.get_support_rep_info(self.authenticated_customer_id)

            result = "**Escalating to Human Support**\n\n"
            result += f"Issue: {issue_description}\n\n"

            if rep_info:
                result += f"Your case has been escalated to {rep_info['FirstName']} {rep_info['LastName']}.\n"
                result += f"They will contact you at your registered email address.\n"
                result += f"For urgent matters, you can reach them directly at: {rep_info['Email']}\n"
            else:
                result += "Your case has been escalated to our general support team.\n"
                result += "A support representative will contact you within 24 hours.\n"

            result += f"\nCase Reference: CASE-{self.authenticated_customer_id}-{hash(issue_description) % 10000:04d}\n"
            result += "Please keep this reference number for your records."

            return result

        @tool
        def store_information(query: str = "") -> str:
            """Provide general store information, policies, and FAQ responses."""
            info_responses = SYSTEM_MESSAGES["STORE_INFO"]
            
            query_lower = query.lower()
            for key, response in info_responses.items():
                if key in query_lower:
                    return f"**{key.title()} Information:**\n{response}"

            # General store info
            return SYSTEM_MESSAGES["STORE_INFO_GENERAL"]

        # Store tools as instance attributes
        self.tools = [
            get_account_info,
            get_support_rep_contact,
            escalate_to_human,
            store_information
        ]

    def get_system_message(self) -> SystemMessage:
        auth_context = ""
        if self.authenticated_customer_id:
            auth_context = f"""
IMPORTANT: The customer is authenticated (Customer ID: {self.authenticated_customer_id}).
When they ask about "my account", "my profile", "my support rep", etc., assume they are asking about their OWN account.
Do NOT ask for customer ID or clarification - the tools will automatically use their authenticated account.
"""

        return SystemMessage(content=f"""
You are a customer support specialist for a digital music store. You handle general inquiries, account management, and escalation to human representatives when needed.

{auth_context}

Your capabilities:
- Provide account information and profile details (for authenticated users)
- Connect customers with their assigned support representatives
- Escalate complex issues to human support
- Answer questions about store policies, hours, and general information
- Help with account management tasks

Guidelines:
- Be friendly, professional, and empathetic
- Protect customer privacy and data security
- Provide accurate store information and policies
- Know when to escalate issues beyond your capabilities
- Follow up to ensure customer satisfaction
- Be proactive in offering additional assistance
- For authenticated users: assume account-related questions refer to THEIR account

When to escalate to human support:
- Technical issues beyond basic troubleshooting
- Billing disputes or complex refund requests
- Account security concerns
- Complaints requiring managerial attention
- Requests for policy exceptions
- Any issue you cannot fully resolve

Always ensure customers feel heard and supported, even when escalating to human representatives.
""")

    def set_authenticated_customer(self, customer_id: int):
        """Set the authenticated customer ID for context-aware tool responses."""
        self.authenticated_customer_id = customer_id

    def get_tools(self):
        """Return the list of tools available to this agent."""
        return self.tools

    def get_pii_middleware(self):
        """Return the list of PII middleware for this agent."""
        return self.pii_middleware