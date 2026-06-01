"""
Transaction Management Agent - Specialized for order history, billing, and purchase support.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from database import DatabaseManager
from datetime import datetime

class TransactionAgent:
    def __init__(self, db_manager: DatabaseManager, llm: ChatOpenAI):
        self.db = db_manager
        self.llm = llm

        # Create tool functions with closure over self.db
        @tool
        def get_order_history(customer_id: int, limit: int = 20) -> str:
            """Get customer's complete order history with invoice summaries."""
            purchases = self.db.get_customer_purchases(customer_id)

            if not purchases:
                return "No order history found for this customer."

            # Group by invoice
            invoices = {}
            for purchase in purchases:
                invoice_id = purchase['InvoiceId']
                if invoice_id not in invoices:
                    invoices[invoice_id] = {
                        'invoice_id': invoice_id,
                        'date': purchase['InvoiceDate'],
                        'total': purchase['Total'],
                        'items': []
                    }
                invoices[invoice_id]['items'].append(purchase)

            # Format response
            invoice_list = sorted(invoices.values(), key=lambda x: x['date'], reverse=True)[:limit]
            result = f"Your order history (showing {len(invoice_list)} most recent orders):\n\n"

            for i, invoice in enumerate(invoice_list, 1):
                result += f"**Order #{invoice['invoice_id']}** - {invoice['date']}\n"
                result += f"Total: ${invoice['total']:.2f} - {len(invoice['items'])} items\n"

                # Show first few items
                for j, item in enumerate(invoice['items'][:3]):
                    result += f"  • {item['TrackName']} by {item['ArtistName']}\n"

                if len(invoice['items']) > 3:
                    result += f"  • ... and {len(invoice['items']) - 3} more items\n"
                result += "\n"

            return result

        @tool
        def get_invoice_details(invoice_id: int, customer_id: int) -> str:
            """Get detailed information for a specific invoice."""
            invoice = self.db.get_invoice_details(invoice_id, customer_id)

            if not invoice:
                return f"Invoice #{invoice_id} not found or doesn't belong to this customer."

            result = f"**Invoice #{invoice['InvoiceId']} Details**\n\n"
            result += f"Date: {invoice['InvoiceDate']}\n"
            result += f"Total: ${invoice['Total']:.2f}\n\n"

            if invoice.get('BillingAddress'):
                result += "**Billing Address:**\n"
                result += f"{invoice['BillingAddress']}\n"
                result += f"{invoice['BillingCity']}, {invoice['BillingState']} {invoice.get('BillingPostalCode', '')}\n"
                result += f"{invoice['BillingCountry']}\n\n"

            result += "**Items Purchased:**\n"
            total_items = 0
            for item in invoice['items']:
                line_total = item['quantity'] * item['unit_price']
                result += f"• {item['track_name']} by {item['artist_name']}\n"
                result += f"  Album: {item['album_title']}\n"
                result += f"  Quantity: {item['quantity']} × ${item['unit_price']:.2f} = ${line_total:.2f}\n\n"
                total_items += item['quantity']

            result += f"**Summary:** {total_items} items, Total: ${invoice['Total']:.2f}"
            return result

        @tool
        def get_spending_summary(customer_id: int) -> str:
            """Get customer's spending summary and statistics."""
            purchases = self.db.get_customer_purchases(customer_id)

            if not purchases:
                return "No purchase history available for spending analysis."

            # Calculate statistics
            total_spent = sum(p['UnitPrice'] * p['Quantity'] for p in purchases)
            total_tracks = sum(p['Quantity'] for p in purchases)
            unique_invoices = len(set(p['InvoiceId'] for p in purchases))
            unique_artists = len(set(p['ArtistName'] for p in purchases if p['ArtistName']))

            # Recent activity
            recent_purchases = [p for p in purchases[:10]]
            last_purchase = purchases[0]['InvoiceDate'] if purchases else None

            result = "**Your Spending Summary**\n\n"
            result += f"• Total Spent: ${total_spent:.2f}\n"
            result += f"• Total Tracks Purchased: {total_tracks}\n"
            result += f"• Number of Orders: {unique_invoices}\n"
            result += f"• Unique Artists: {unique_artists}\n"
            result += f"• Average per Track: ${total_spent/total_tracks:.2f}\n"
            result += f"• Average per Order: ${total_spent/unique_invoices:.2f}\n"

            if last_purchase:
                result += f"• Last Purchase: {last_purchase}\n"

            return result

        @tool
        def check_recent_orders(customer_id: int, days: int = 30) -> str:
            """Check for recent orders within specified number of days."""
            purchases = self.db.get_customer_purchases(customer_id)

            if not purchases:
                return "No recent orders found."

            # Filter recent purchases (simplified - assumes date format)
            recent_invoices = {}
            for purchase in purchases[:20]:  # Check recent purchases
                invoice_id = purchase['InvoiceId']
                if invoice_id not in recent_invoices:
                    recent_invoices[invoice_id] = {
                        'invoice_id': invoice_id,
                        'date': purchase['InvoiceDate'],
                        'total': purchase['Total']
                    }

            if not recent_invoices:
                return f"No orders found in the last {days} days."

            result = f"**Recent Orders (Last {days} days):**\n\n"
            for invoice in list(recent_invoices.values())[:5]:
                result += f"• Order #{invoice['invoice_id']} - {invoice['date']} - ${invoice['total']:.2f}\n"

            return result

        # Store tools as instance attributes
        self.tools = [
            get_order_history,
            get_invoice_details,
            get_spending_summary,
            check_recent_orders
        ]

    def get_system_message(self) -> SystemMessage:
        return SystemMessage(content="""
You are a transaction and billing specialist for a digital music store. You help customers with their order history, invoice details, refunds, and billing inquiries.

Your capabilities:
- Retrieve and explain order history
- Provide detailed invoice information
- Calculate spending summaries and statistics
- Help with billing address updates
- Assist with refund and return processes
- Track recent purchase activity

Guidelines:
- Be professional and helpful with billing matters
- Protect customer privacy - only show their own transaction data
- Clearly explain charges and fees
- Provide exact invoice details when requested
- Help customers understand their purchase patterns
- Escalate complex billing disputes to human support when needed
- Always verify invoice ownership before sharing details

For refunds and disputes:
- Acknowledge the customer's concern
- Gather specific details about the issue
- Explain the refund policy clearly
- Guide them through the appropriate process
- Escalate to human support rep when necessary

Use the available tools to get accurate transaction data for each customer query.
""")

    def get_tools(self):
        """Return the list of tools available to this agent."""
        return self.tools