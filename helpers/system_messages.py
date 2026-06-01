
"""
System messages and automated responses for the Music Store Customer Support.
All user-facing automated messages should be defined here to eliminate bloat from the main app.
"""

SYSTEM_MESSAGES = {
    # Welcome and greeting messages
    "WELCOME": """Welcome to our Music Store Customer Support! 🎵

I can help you with:
• **General music questions** and recommendations
• **Account assistance** (provide your email for personalized help)
• **Order inquiries** (authentication required)
• **Technical support**

How can I assist you today?""",

    "AUTHENTICATION_SUCCESS": "Great! I've authenticated you as {first_name} ({email}). Now I can provide personalized assistance with your account, orders, and music recommendations. How can I help you today?",

    "AUTHENTICATION_FAILED": "I couldn't find an account with email '{email}'. You can still ask general questions about music, but I won't be able to access your account information.",

    # Error messages
    "NO_MESSAGE_RECEIVED": "I didn't receive a valid message to process.",

    "GENERAL_ERROR": "I apologize, but I encountered an error. Please try again or contact support directly.",

    "ERROR_MESSAGES": {
        "music": "I apologize, but I encountered an error while processing your music request. Please try again or contact support.",
        "transaction": "I apologize, but I encountered an error while processing your transaction request. Please try again or contact support.",
        "support": "I apologize, but I encountered an error while processing your support request. Please try again or let me escalate this to a human representative.",
        "tavily": "I apologize, but I encountered an error while searching for information. Please try again.",
        "general": "I apologize, but I encountered an error while processing your request. Please try again or contact support."
    },

    "REQUEST_PROCESSING_ERROR": "I'm sorry, I couldn't process your request.",

    "SEARCH_NO_CLEAR_RESPONSE": "I was able to search for information but didn't get a clear response. Could you please rephrase your question?",

    # Authentication requirement messages
    "AUTH_REQUIRED_ACCOUNT_INFO": "You need to be authenticated to view account information. Please provide your email address.",

    "AUTH_REQUIRED_SUPPORT_REP": "You need to be authenticated to view your support representative. Please provide your email address.",

    "AUTH_REQUIRED_ESCALATION": "You need to be authenticated to escalate issues. Please provide your email address.",

    "AUTH_REQUIRED_ORDER_HISTORY": "You need to be authenticated to view your order history. Please provide your email address.",

    "AUTH_REQUIRED_INVOICE_DETAILS": "You need to be authenticated to view invoice details. Please provide your email address.",

    "AUTH_REQUIRED_SPENDING_SUMMARY": "You need to be authenticated to view your spending summary. Please provide your email address.",

    "AUTH_REQUIRED_RECENT_ORDERS": "You need to be authenticated to view your recent orders. Please provide your email address.",

    # No data found messages
    "CUSTOMER_NOT_FOUND": "Customer account not found.",

    "NO_SUPPORT_REP": "No dedicated support representative assigned to your account.",

    "NO_ORDER_HISTORY": "No order history found for this customer.",

    "NO_PURCHASE_HISTORY": "No purchase history found.",

    "NO_PURCHASE_HISTORY_PREFERENCES": "No purchase history found for music preference analysis.",

    "NO_SPENDING_HISTORY": "No purchase history available for spending analysis.",

    "NO_RECENT_ORDERS": "No recent orders found.",

    "NO_ORDERS_IN_TIMEFRAME": "No orders found in the last {days} days.",

    "INVOICE_NOT_FOUND": "Invoice #{invoice_id} not found or doesn't belong to this customer.",

    # Conversation responses
    "GREETING_AUTHENTICATED": "Hello! How can I help you with your music needs today?",

    "GREETING_UNAUTHENTICATED": "Hello! How can I help you with your music needs today? (Provide your email for personalized account assistance)",

    "THANKS_AUTHENTICATED": "You're welcome! Is there anything else I can help you with your account or music preferences?",

    "THANKS_UNAUTHENTICATED": "You're welcome! Is there anything else I can help you with?",

    "GOODBYE_AUTHENTICATED": "Goodbye{name}! Feel free to come back anytime for account assistance or music recommendations!",

    "GOODBYE_UNAUTHENTICATED": "Goodbye! Feel free to come back if you need any music assistance!",

    # Support information responses
    "STORE_INFO": {
        "hours": "Our digital music store is available 24/7 for purchases and downloads. Customer support is available Monday-Friday 9AM-6PM EST.",
        "refund": "Refund Policy: Digital music purchases can be refunded within 14 days if there was a technical issue preventing download. Contact your support rep for assistance.",
        "formats": "We offer music in MP3 format (320 kbps) compatible with all devices and platforms.",
        "download": "After purchase, you can download your music immediately. Downloads are available for 30 days after purchase.",
        "account": "You can update your account information, billing address, and password through your account settings.",
        "payment": "We accept major credit cards (Visa, MasterCard, American Express) and PayPal.",
        "quality": "All tracks are high-quality MP3s encoded at 320 kbps for excellent sound quality."
    },

    "STORE_INFO_GENERAL": """**Digital Music Store Information:**

• **Store Hours:** 24/7 for purchases, Support: Mon-Fri 9AM-6PM EST
• **Music Format:** High-quality MP3 (320 kbps)
• **Payment:** Major credit cards and PayPal accepted
• **Downloads:** Available immediately after purchase for 30 days
• **Refunds:** 14-day policy for technical issues
• **Account Management:** Update details in your account settings

For specific questions about our policies or technical issues, please let me know how I can help!""",

    # Account information templates
    "ACCOUNT_INFO_HEADER": "**Your Account Information:**\n\n",
    "ACCOUNT_CONTACT_INFO_HEADER": "\n**Contact Information:**\n",
    "SUPPORT_REP_HEADER": "**Your Assigned Support Representative:**\n\n",
    "SUPPORT_REP_FOOTER": "\nFor complex issues that require human assistance, you can contact your dedicated support representative directly.",

    # Escalation messages
    "ESCALATION_HEADER": "**Escalating to Human Support**\n\n",
    "ESCALATION_WITH_REP": "Your case has been escalated to {first_name} {last_name}.\nThey will contact you at your registered email address.\nFor urgent matters, you can reach them directly at: {email}\n",
    "ESCALATION_GENERAL": "Your case has been escalated to our general support team.\nA support representative will contact you within 24 hours.\n",
    "ESCALATION_FOOTER": "\nCase Reference: CASE-{customer_id}-{case_number:04d}\nPlease keep this reference number for your records.",

    # Feature descriptions
        "FEATURES_AUTHENTICATED": """## 🎵 Music Store Support Bot Features
*You are authenticated as: {customer_email}*

**🔐 Your Account**
- View your order history and invoices
- Get personalized music recommendations based on your purchases
- Access your account information and billing details

**🎵 Personalized Music Recommendations**
- Suggestions based on your purchase history
- Genre recommendations from your favorites
- New releases in your preferred styles

**💳 Your Transactions**
- View all your past orders
- Get invoice details and receipts
- Track purchases and payments

**🤝 Account Support**
- Update account information
- Billing and payment assistance
- Technical support for your account

How can I assist you with your account today?""",

    "FEATURES_UNAUTHENTICATED": """## 🎵 Music Store Support Bot Features

**🔐 Authentication & Account Management**
- Secure customer authentication via email
- Access to personalized account information
- Order history and billing support

**🎵 Music Recommendations**
- Personalized music suggestions based on preferences
- Genre-based recommendations (jazz, rock, classical, etc.)
- Artist and album discovery

**💳 Transaction Support**
- Order history lookup
- Invoice and billing inquiries
- Payment and purchase assistance

**🤝 Customer Support**
- General music store inquiries
- Technical support and troubleshooting
- Human escalation when needed

**To access personalized features, please provide your email address.**

How can I assist you today?"""
}
