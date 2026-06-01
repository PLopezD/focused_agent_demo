"""
PII Redaction Evaluation Script

This script evaluates the effectiveness of PII middleware in redacting
phone numbers, zip codes, and addresses using the "Middleware Dataset" in LangSmith.
"""

import re
import asyncio
import sys
import os
from typing import Dict, List, Any, Tuple
from langsmith import Client
from langsmith.evaluation import evaluate
from pydantic import BaseModel

# Add parent directory to path to import from main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MusicStoreOrchestrator
from helpers.pii_config import PII_PATTERNS

class PIIEvaluationResult(BaseModel):
    """Result of PII evaluation for a single response."""
    phone_redacted: bool = True
    zipcode_redacted: bool = True
    address_redacted: bool = True
    phone_detected_count: int = 0
    zipcode_detected_count: int = 0
    address_detected_count: int = 0
    raw_response: str = ""
    detected_pii: Dict[str, List[str]] = {}

class PIIRedactionEvaluator:
    """Evaluator for PII redaction in agent responses."""

    def __init__(self):
        """Initialize the PII redaction evaluator."""
        self.langsmith_client = Client()
        self.orchestrator = MusicStoreOrchestrator(use_memory=False)

        # Compile regex patterns for efficient matching
        self.phone_pattern = re.compile(
            '|'.join(f'({pattern})' for pattern in PII_PATTERNS["phone_number"]),
            re.IGNORECASE
        )
        self.zipcode_pattern = re.compile(
            '|'.join(f'({pattern})' for pattern in PII_PATTERNS["zipcode"]),
            re.IGNORECASE
        )
        self.address_pattern = re.compile(
            '|'.join(f'({pattern})' for pattern in PII_PATTERNS["address"]),
            re.IGNORECASE
        )

        # Redaction patterns to detect if PII was properly redacted
        self.redaction_patterns = {
            'phone': re.compile(r'\*{3}-\*{3}-\d{4}|\[REDACTED_PHONE\]|\[REDACTED_PHONE_NUMBER\]', re.IGNORECASE),
            'zipcode': re.compile(r'\[REDACTED_ZIPCODE\]|\[REDACTED_ZIP\]', re.IGNORECASE),
            'address': re.compile(r'\[REDACTED_ADDRESS\]', re.IGNORECASE)
        }

    def detect_unredacted_pii(self, text: str) -> PIIEvaluationResult:
        """
        Detect unredacted PII in the response text.

        Args:
            text: The response text to check for PII

        Returns:
            PIIEvaluationResult with detection results
        """
        result = PIIEvaluationResult(raw_response=text)

        # Detect unredacted phone numbers
        phone_matches = self.phone_pattern.findall(text)
        if phone_matches:
            # Flatten the tuple matches from regex groups
            phone_numbers = []
            for match in phone_matches:
                if isinstance(match, tuple):
                    phone_numbers.extend([m for m in match if m])
                else:
                    phone_numbers.append(match)
            result.phone_detected_count = len(phone_numbers)
            result.detected_pii['phone'] = phone_numbers
            result.phone_redacted = False

        # Detect unredacted zip codes
        zipcode_matches = self.zipcode_pattern.findall(text)
        if zipcode_matches:
            # Flatten the tuple matches from regex groups
            zipcodes = []
            for match in zipcode_matches:
                if isinstance(match, tuple):
                    zipcodes.extend([m for m in match if m])
                else:
                    zipcodes.append(match)
            result.zipcode_detected_count = len(zipcodes)
            result.detected_pii['zipcode'] = zipcodes
            result.zipcode_redacted = False

        # Detect unredacted addresses
        address_matches = self.address_pattern.findall(text)
        if address_matches:
            # Flatten the tuple matches from regex groups
            addresses = []
            for match in address_matches:
                if isinstance(match, tuple):
                    addresses.extend([m for m in match if m])
                else:
                    addresses.append(match)
            result.address_detected_count = len(addresses)
            result.detected_pii['address'] = addresses
            result.address_redacted = False

        return result

    def check_redaction_presence(self, text: str) -> Dict[str, bool]:
        """
        Check if redaction markers are present in the text.

        Args:
            text: The response text to check for redaction markers

        Returns:
            Dictionary indicating presence of redaction markers
        """
        return {
            'phone_redaction_present': bool(self.redaction_patterns['phone'].search(text)),
            'zipcode_redaction_present': bool(self.redaction_patterns['zipcode'].search(text)),
            'address_redaction_present': bool(self.redaction_patterns['address'].search(text))
        }

def run_orchestrator_with_message(message: str, email: str = None) -> str:
    """
    Run the orchestrator with a message and optional email for authentication.

    Args:
        message: The user message to send
        email: Optional email for authentication

    Returns:
        The orchestrator's response
    """
    try:
        orchestrator = MusicStoreOrchestrator(use_memory=True)  # Use memory for session

        # If email is provided, authenticate first
        if email:
            # Use a real email from the database for testing
            test_emails = {
                "customer@example.com": "luisg@embraer.com.br",
                "user@test.com": "ftremblay@gmail.com",
                "test@test.com": "leonekohler@surfeu.de"
            }

            actual_email = test_emails.get(email, email)
            auth_response = orchestrator.chat(f"My email is {actual_email}", "test_session")
            print(f"[DEBUG] Auth response: {auth_response[:200]}...")

            # Then send the actual message
            response = orchestrator.chat(message, "test_session")
        else:
            response = orchestrator.chat(message)

        return response
    except Exception as e:
        return f"Error: {str(e)}"

def get_test_customer_data() -> str:
    """
    Get test customer data directly from database to simulate what should be redacted.
    This bypasses the orchestrator to get raw customer data for testing.
    """
    try:
        from database import DatabaseManager

        db = DatabaseManager()
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT FirstName, LastName, Email, Address, City, State, PostalCode, Phone, Company
                FROM Customer
                WHERE Email = 'luisg@embraer.com.br'
                LIMIT 1
            """)
            customer = cursor.fetchone()

            if customer:
                # Format as if returned by support agent
                return f"""Here is your account information:

Name: {customer[0]} {customer[1]}
Email: {customer[2]}
Company: {customer[8]}
Address: {customer[3]}
City: {customer[4]}
State: {customer[5]}
Postal Code: {customer[6]}
Phone: {customer[7]}

Your account is in good standing. Is there anything specific you'd like to update?"""
            else:
                return "Customer not found"

    except Exception as e:
        return f"Database error: {str(e)}"

async def evaluate_pii_redaction(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate PII redaction for a single test case.

    Args:
        inputs: Dictionary containing 'message' and optionally 'email'

    Returns:
        Dictionary with evaluation results
    """
    evaluator = PIIRedactionEvaluator()

    # Get the message and email from inputs
    message = inputs.get('message', '')
    email = inputs.get('email')

    # Run the orchestrator
    response = run_orchestrator_with_message(message, email)

    # Evaluate PII redaction
    pii_result = evaluator.detect_unredacted_pii(response)
    redaction_markers = evaluator.check_redaction_presence(response)

    # Calculate overall scores
    pii_properly_redacted = (
        pii_result.phone_redacted and
        pii_result.zipcode_redacted and
        pii_result.address_redacted
    )

    # Create detailed results
    results = {
        'pii_properly_redacted': pii_properly_redacted,
        'phone_redacted': pii_result.phone_redacted,
        'zipcode_redacted': pii_result.zipcode_redacted,
        'address_redacted': pii_result.address_redacted,
        'phone_detected_count': pii_result.phone_detected_count,
        'zipcode_detected_count': pii_result.zipcode_detected_count,
        'address_detected_count': pii_result.address_detected_count,
        'detected_pii': pii_result.detected_pii,
        'redaction_markers_present': redaction_markers,
        'response': response,
        'input_message': message,
        'input_email': email
    }

    return results

def pii_redaction_binary_evaluator(run, example) -> dict:
    """
    Binary evaluator for PII redaction (pass/fail).

    Args:
        run: The run result from LangSmith
        example: The example from the dataset

    Returns:
        Dictionary with binary evaluation result
    """
    try:
        # Get the response from the run
        response = ""
        if hasattr(run, 'outputs') and run.outputs:
            response = run.outputs.get('response', run.outputs.get('output', ''))
        elif hasattr(run, 'output'):
            response = run.output

        evaluator = PIIRedactionEvaluator()
        pii_result = evaluator.detect_unredacted_pii(response)

        # Check if all PII types are properly redacted
        all_redacted = (
            pii_result.phone_redacted and
            pii_result.zipcode_redacted and
            pii_result.address_redacted
        )

        return {
            "key": "pii_redaction_pass",
            "score": 1 if all_redacted else 0,
            "comment": f"Phone: {'✓' if pii_result.phone_redacted else '✗'}, "
                      f"ZIP: {'✓' if pii_result.zipcode_redacted else '✗'}, "
                      f"Address: {'✓' if pii_result.address_redacted else '✗'}. "
                      f"Detected PII: {pii_result.detected_pii}"
        }
    except Exception as e:
        return {
            "key": "pii_redaction_pass",
            "score": 0,
            "comment": f"Evaluation error: {str(e)}"
        }

def pii_redaction_detailed_evaluator(run, example) -> dict:
    """
    Detailed evaluator for PII redaction with breakdown by PII type.

    Args:
        run: The run result from LangSmith
        example: The example from the dataset

    Returns:
        Dictionary with detailed evaluation results
    """
    try:
        # Get the response from the run
        response = ""
        if hasattr(run, 'outputs') and run.outputs:
            response = run.outputs.get('response', run.outputs.get('output', ''))
        elif hasattr(run, 'output'):
            response = run.output

        evaluator = PIIRedactionEvaluator()
        pii_result = evaluator.detect_unredacted_pii(response)
        redaction_markers = evaluator.check_redaction_presence(response)

        # Calculate individual scores
        phone_score = 1 if pii_result.phone_redacted else 0
        zipcode_score = 1 if pii_result.zipcode_redacted else 0
        address_score = 1 if pii_result.address_redacted else 0
        overall_score = (phone_score + zipcode_score + address_score) / 3

        return {
            "key": "pii_redaction_detailed",
            "score": overall_score,
            "details": {
                "phone_redacted": pii_result.phone_redacted,
                "zipcode_redacted": pii_result.zipcode_redacted,
                "address_redacted": pii_result.address_redacted,
                "phone_detected_count": pii_result.phone_detected_count,
                "zipcode_detected_count": pii_result.zipcode_detected_count,
                "address_detected_count": pii_result.address_detected_count,
                "detected_pii": pii_result.detected_pii,
                "redaction_markers": redaction_markers
            },
            "comment": f"Overall PII Redaction Score: {overall_score:.2f}. "
                      f"Phone: {phone_score}, ZIP: {zipcode_score}, Address: {address_score}"
        }
    except Exception as e:
        return {
            "key": "pii_redaction_detailed",
            "score": 0,
            "details": {"error": str(e)},
            "comment": f"Detailed evaluation error: {str(e)}"
        }

async def run_pii_evaluation():
    """
    Run the PII redaction evaluation using the Middleware Dataset in LangSmith.
    """
    print("🔍 Starting PII Redaction Evaluation...")

    try:
        # Run the evaluation
        results = await evaluate(
            run_orchestrator_with_message,
            data="Middleware Dataset",  # Your LangSmith dataset name
            evaluators=[
                pii_redaction_binary_evaluator,
                pii_redaction_detailed_evaluator
            ],
            experiment_prefix="pii_redaction_eval",
            description="Evaluation of PII redaction middleware for phone numbers, zip codes, and addresses"
        )

        print("✅ PII Redaction Evaluation completed!")
        print(f"📊 Results summary:")
        print(f"   - Experiment ID: {results.experiment_name}")
        print(f"   - Dataset: Middleware Dataset")
        print(f"   - Evaluators: Binary Pass/Fail + Detailed Breakdown")

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise

def run_single_test():
    """
    Run a single test to verify the evaluation logic works.
    """
    print("🧪 Running single test...")

    # Test 1: Raw customer data (to verify PII detection works)
    print("\n=== Test 1: PII Detection on Raw Data ===")
    raw_data = get_test_customer_data()
    print(f"Raw customer data:\n{raw_data}")

    evaluator = PIIRedactionEvaluator()
    pii_result = evaluator.detect_unredacted_pii(raw_data)
    redaction_markers = evaluator.check_redaction_presence(raw_data)

    print(f"\nPII Detection Result on Raw Data:")
    print(f"  - Phone redacted: {pii_result.phone_redacted}")
    print(f"  - ZIP redacted: {pii_result.zipcode_redacted}")
    print(f"  - Address redacted: {pii_result.address_redacted}")
    print(f"  - Phone detected: {pii_result.phone_detected_count}")
    print(f"  - ZIP detected: {pii_result.zipcode_detected_count}")
    print(f"  - Address detected: {pii_result.address_detected_count}")
    print(f"  - Detected PII: {pii_result.detected_pii}")

    # Test 2: Orchestrator response
    print("\n=== Test 2: Orchestrator Response ===")
    test_message = "Show me my account information"
    test_email = "customer@example.com"

    response = run_orchestrator_with_message(test_message, test_email)
    print(f"\nOrchestrator Response:\n{response}")

    pii_result2 = evaluator.detect_unredacted_pii(response)
    redaction_markers2 = evaluator.check_redaction_presence(response)

    print(f"\nPII Detection Result on Orchestrator Response:")
    print(f"  - Phone redacted: {pii_result2.phone_redacted}")
    print(f"  - ZIP redacted: {pii_result2.zipcode_redacted}")
    print(f"  - Address redacted: {pii_result2.address_redacted}")
    print(f"  - Detected PII: {pii_result2.detected_pii}")
    print(f"  - Redaction markers: {redaction_markers2}")

    # Test 3: Test redacted data to verify detection works
    print("\n=== Test 3: Manually Redacted Data ===")
    redacted_data = """Here is your account information:

Name: Luís Gonçalves
Email: [REDACTED_EMAIL]
Company: Embraer
Address: [REDACTED_ADDRESS]
City: São José dos Campos
State: SP
Postal Code: [REDACTED_ZIPCODE]
Phone: ***-***-5555

Your account is in good standing."""

    print(f"Manually redacted data:\n{redacted_data}")

    pii_result3 = evaluator.detect_unredacted_pii(redacted_data)
    redaction_markers3 = evaluator.check_redaction_presence(redacted_data)

    print(f"\nPII Detection Result on Manually Redacted Data:")
    print(f"  - Phone redacted: {pii_result3.phone_redacted}")
    print(f"  - ZIP redacted: {pii_result3.zipcode_redacted}")
    print(f"  - Address redacted: {pii_result3.address_redacted}")
    print(f"  - Redaction markers present: {redaction_markers3}")

def test_pii_middleware_directly():
    """
    Test the PII middleware directly by creating an agent and running a query.
    """
    print("🧪 Testing PII middleware directly...")

    try:
        from database import DatabaseManager
        from agents.support_agent import CustomerSupportAgent
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent

        # Create support agent with PII middleware
        db = DatabaseManager()
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        support_agent = CustomerSupportAgent(db, llm)

        # Set authenticated customer
        support_agent.set_authenticated_customer(1)  # First customer in DB

        # Create agent with PII middleware
        agent_executor = create_agent(
            model=llm,
            tools=support_agent.get_tools(),
            middleware=support_agent.get_pii_middleware()
        )

        # Test with account info request
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            support_agent.get_system_message(),
            SystemMessage(content="Customer ID: 1"),
            HumanMessage(content="Show me my account information")
        ]

        print("Running agent with PII middleware...")
        result = agent_executor.invoke({"messages": messages})

        if result.get("messages"):
            response = result["messages"][-1].content
            print(f"Agent response:\n{response}")

            # Check for PII redaction
            evaluator = PIIRedactionEvaluator()
            pii_result = evaluator.detect_unredacted_pii(response)

            print(f"\nPII Detection Result:")
            print(f"  - Phone redacted: {pii_result.phone_redacted}")
            print(f"  - ZIP redacted: {pii_result.zipcode_redacted}")
            print(f"  - Address redacted: {pii_result.address_redacted}")
            print(f"  - Detected PII: {pii_result.detected_pii}")

    except Exception as e:
        print(f"Error testing middleware directly: {e}")
        import traceback
        traceback.print_exc()

