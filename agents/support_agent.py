"""
Customer Support Agent - General support, account management, and escalation handling.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from database import DatabaseManager

class CustomerSupportAgent:
    def __init__(self, db_manager: DatabaseManager, llm: ChatOpenAI):
        self.db = db_manager
        self.llm = llm

        # Create tool functions with closure over self.db
        @tool
        def get_account_info(customer_id: int) -> str:
            """Get customer account information and profile details."""
            # Get customer info (assuming it's available from auth context)
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT CustomerId, FirstName, LastName, Email, Company,
                           Address, City, State, Country, PostalCode, Phone
                    FROM Customer
                    WHERE CustomerId = ?
                """, (customer_id,))

                customer = cursor.fetchone()
                if not customer:
                    return "Customer account not found."

                result = "**Your Account Information:**\n\n"
                result += f"Name: {customer['FirstName']} {customer['LastName']}\n"
                result += f"Email: {customer['Email']}\n"

                if customer['Company']:
                    result += f"Company: {customer['Company']}\n"

                result += "\n**Contact Information:**\n"
                if customer['Address']:
                    result += f"Address: {customer['Address']}\n"
                    result += f"City: {customer['City']}\n"
                    if customer['State']:
                        result += f"State: {customer['State']}\n"
                    result += f"Country: {customer['Country']}\n"
                    if customer['PostalCode']:
                        result += f"Postal Code: {customer['PostalCode']}\n"

                if customer['Phone']:
                    result += f"Phone: {customer['Phone']}\n"

                return result

        @tool
        def get_support_rep_contact(customer_id: int) -> str:
            """Get assigned support representative contact information."""
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT e.FirstName, e.LastName, e.Email, e.Phone, e.Title
                    FROM Customer c
                    JOIN Employee e ON c.SupportRepId = e.EmployeeId
                    WHERE c.CustomerId = ?
                """, (customer_id,))

                rep = cursor.fetchone()
                if not rep:
                    return "No dedicated support representative assigned to your account."

                result = "**Your Assigned Support Representative:**\n\n"
                result += f"Name: {rep['FirstName']} {rep['LastName']}\n"
                result += f"Title: {rep['Title']}\n"
                result += f"Email: {rep['Email']}\n"
                if rep['Phone']:
                    result += f"Phone: {rep['Phone']}\n"

                result += "\nFor complex issues that require human assistance, you can contact your dedicated support representative directly."
                return result

        @tool
        def escalate_to_human(customer_id: int, issue_description: str) -> str:
            """Escalate customer issue to human support representative."""
            # Get support rep info
            rep_info = self.db.get_support_rep_info(customer_id)

            result = "**Escalating to Human Support**\n\n"
            result += f"Issue: {issue_description}\n\n"

            if rep_info:
                result += f"Your case has been escalated to {rep_info['FirstName']} {rep_info['LastName']}.\n"
                result += f"They will contact you at your registered email address.\n"
                result += f"For urgent matters, you can reach them directly at: {rep_info['Email']}\n"
            else:
                result += "Your case has been escalated to our general support team.\n"
                result += "A support representative will contact you within 24 hours.\n"

            result += f"\nCase Reference: CASE-{customer_id}-{hash(issue_description) % 10000:04d}\n"
            result += "Please keep this reference number for your records."

            return result

        @tool
        def store_information(query: str = "") -> str:
            """Provide general store information, policies, and FAQ responses."""
            info_responses = {
                "hours": "Our digital music store is available 24/7 for purchases and downloads. Customer support is available Monday-Friday 9AM-6PM EST.",
                "refund": "Refund Policy: Digital music purchases can be refunded within 14 days if there was a technical issue preventing download. Contact your support rep for assistance.",
                "formats": "We offer music in MP3 format (320 kbps) compatible with all devices and platforms.",
                "download": "After purchase, you can download your music immediately. Downloads are available for 30 days after purchase.",
                "account": "You can update your account information, billing address, and password through your account settings.",
                "payment": "We accept major credit cards (Visa, MasterCard, American Express) and PayPal.",
                "quality": "All tracks are high-quality MP3s encoded at 320 kbps for excellent sound quality."
            }

            query_lower = query.lower()
            for key, response in info_responses.items():
                if key in query_lower:
                    return f"**{key.title()} Information:**\n{response}"

            # General store info
            return """**Digital Music Store Information:**

• **Store Hours:** 24/7 for purchases, Support: Mon-Fri 9AM-6PM EST
• **Music Format:** High-quality MP3 (320 kbps)
• **Payment:** Major credit cards and PayPal accepted
• **Downloads:** Available immediately after purchase for 30 days
• **Refunds:** 14-day policy for technical issues
• **Account Management:** Update details in your account settings

For specific questions about our policies or technical issues, please let me know how I can help!"""

        # Store tools as instance attributes
        self.tools = [
            get_account_info,
            get_support_rep_contact,
            escalate_to_human,
            store_information
        ]

    def get_system_message(self) -> SystemMessage:
        return SystemMessage(content="""
You are a customer support specialist for a digital music store. You handle general inquiries, account management, and escalation to human representatives when needed.

Your capabilities:
- Provide account information and profile details
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

When to escalate to human support:
- Technical issues beyond basic troubleshooting
- Billing disputes or complex refund requests
- Account security concerns
- Complaints requiring managerial attention
- Requests for policy exceptions
- Any issue you cannot fully resolve

Always ensure customers feel heard and supported, even when escalating to human representatives.
""")

    def get_tools(self):
        """Return the list of tools available to this agent."""
        return self.tools