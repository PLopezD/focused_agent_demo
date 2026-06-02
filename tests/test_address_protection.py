"""
Address Protection Test

This test verifies that address information is never returned by any agent,
specifically testing the leonekohler@surfeu.de case that was leaking address data.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.support_agent import CustomerSupportAgent
from agents.transaction_agent import TransactionAgent
from database import DatabaseManager


class TestAddressProtection:
    """Test that no address information leaks through any agent."""

    @pytest.fixture(scope="class")
    def db_manager(self):
        """Create database manager instance."""
        return DatabaseManager()

    @pytest.fixture(scope="class")
    def mock_llm(self):
        """Create mock LLM for testing."""
        return MagicMock()

    def test_support_agent_no_address_leak(self, db_manager, mock_llm):
        """Test that support agent doesn't return address information."""

        support_agent = CustomerSupportAgent(db_manager, mock_llm)
        support_agent.set_authenticated_customer(2)  # Leonie Köhler

        # Get the account info tool
        tools = support_agent.get_tools()
        account_info_tool = None
        for tool in tools:
            if hasattr(tool, 'name') and 'account' in tool.name.lower():
                account_info_tool = tool
                break

        assert account_info_tool is not None, "Account info tool not found"

        # Execute the tool
        result = account_info_tool.invoke({})

        print(f"\nSupport agent account info result:")
        print(result)

        # Check that no address information is present
        forbidden_terms = [
            'address:', 'street', 'avenue', 'road', 'drive',
            'theodor-heuss', 'stuttgart', '70174',  # Specific to this customer
            'billing address', 'postal code', 'zip code'
        ]

        result_lower = result.lower()
        leaked_terms = []
        for term in forbidden_terms:
            if term in result_lower:
                leaked_terms.append(term)

        assert len(leaked_terms) == 0, f"Address information leaked: {leaked_terms}"

        # Should still contain allowed information
        assert 'leonie' in result_lower, "Customer name should still be present"
        assert 'germany' in result_lower, "Country should still be present (allowed)"

    def test_transaction_agent_no_billing_address_leak(self, db_manager, mock_llm):
        """Test that transaction agent doesn't return billing address."""

        transaction_agent = TransactionAgent(db_manager, mock_llm)
        transaction_agent.set_authenticated_customer(2)  # Leonie Köhler

        # Get the invoice details tool
        tools = transaction_agent.get_tools()
        invoice_tool = None
        for tool in tools:
            if hasattr(tool, 'name') and 'invoice' in tool.name.lower():
                invoice_tool = tool
                break

        assert invoice_tool is not None, "Invoice details tool not found"

        # Get an invoice ID for this customer first
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT InvoiceId FROM Invoice
                WHERE CustomerId = 2
                LIMIT 1
            """)
            invoice_row = cursor.fetchone()

        assert invoice_row is not None, "No invoices found for test customer"
        invoice_id = invoice_row['InvoiceId']

        # Execute the tool with the invoice ID
        result = invoice_tool.invoke({"invoice_id": invoice_id})

        print(f"\nTransaction agent invoice result:")
        print(result)

        # Check that no billing address information is present
        forbidden_terms = [
            'theodor-heuss', 'stuttgart', '70174',  # Specific address components
            'billing address:', 'address:', 'street', 'straße',
            'postal code:', 'zip code:'
        ]

        result_lower = result.lower()
        leaked_terms = []
        for term in forbidden_terms:
            if term in result_lower:
                leaked_terms.append(term)

        assert len(leaked_terms) == 0, f"Billing address information leaked: {leaked_terms}"

        # Should still contain allowed information
        assert 'germany' in result_lower, "Billing country should still be present (allowed)"

    def test_comprehensive_address_protection(self, db_manager):
        """Comprehensive test across multiple customers to ensure no address leaks."""

        test_customers = [
            ('leonekohler@surfeu.de', 2),  # The problematic case
            ('ftremblay@gmail.com', 3),
            ('bjorn.hansen@yahoo.no', 4)
        ]

        for email, customer_id in test_customers:
            print(f"\nTesting address protection for {email}...")

            # Get raw customer data to see what we're protecting
            with db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT Address, City, State, PostalCode
                    FROM Customer
                    WHERE CustomerId = ?
                """, (customer_id,))
                customer_data = cursor.fetchone()

            if customer_data:
                print(f"  Raw address data: {dict(customer_data)}")

                # Simulate what the support agent would return
                mock_llm = MagicMock()
                support_agent = CustomerSupportAgent(db_manager, mock_llm)
                support_agent.set_authenticated_customer(customer_id)

                tools = support_agent.get_tools()
                for tool in tools:
                    if hasattr(tool, 'name') and 'account' in tool.name.lower():
                        result = tool.invoke({})

                        # Check for any address components
                        result_lower = result.lower()
                        address_components = [
                            customer_data['Address'].lower() if customer_data['Address'] else '',
                            customer_data['City'].lower() if customer_data['City'] else '',
                            customer_data['State'].lower() if customer_data['State'] else '',
                            customer_data['PostalCode'].lower() if customer_data['PostalCode'] else ''
                        ]

                        address_components = [comp for comp in address_components if comp]  # Remove empty strings

                        leaked_components = []
                        for component in address_components:
                            if component and component in result_lower:
                                leaked_components.append(component)

                        assert len(leaked_components) == 0, f"Address components leaked for {email}: {leaked_components}"

                        print(f"  ✅ No address leaks detected for {email}")
                        break


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])