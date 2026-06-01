"""
Music Recommendation Agent - Specialized for music discovery and personalized recommendations.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from database import DatabaseManager

class MusicRecommendationAgent:
    def __init__(self, db_manager: DatabaseManager, llm: ChatOpenAI):
        self.db = db_manager    
        self.llm = llm

        # Create tool functions with closure over self.db
        @tool
        def get_customer_genres(customer_id: int) -> str:
            """Get customer's preferred music genres based on purchase history."""
            genres = self.db.get_customer_genres(customer_id)
            if not genres:
                return "No purchase history found for music preference analysis."

            result = "Based on your purchase history, your preferred genres are:\n"
            for i, genre in enumerate(genres, 1):
                result += f"{i}. {genre['Genre']} - {genre['PurchaseCount']} purchases, ${genre['TotalSpent']:.2f} spent\n"

            return result

        @tool
        def get_genre_recommendations(customer_id: int, genres: str, limit: int = 10) -> str:
            """Get music recommendations based on specified genres."""
            genre_list = [g.strip() for g in genres.split(',')]
            recommendations = self.db.get_recommendations_by_genre(genre_list, customer_id, limit)

            if not recommendations:
                return f"No new recommendations found for genres: {', '.join(genre_list)}"

            result = f"Here are {len(recommendations)} recommendations based on {', '.join(genre_list)}:\n\n"
            for i, track in enumerate(recommendations, 1):
                result += f"{i}. **{track['TrackName']}** by {track['ArtistName']}\n"
                result += f"   Album: {track['AlbumTitle']}, Genre: {track['Genre']}, Price: ${track['UnitPrice']}\n\n"

            return result

        @tool
        def search_music(query: str, limit: int = 10) -> str:
            """Search for music by track name, artist, or album."""
            results = self.db.search_tracks(query, limit)

            if not results:
                return f"No music found matching '{query}'"

            result = f"Found {len(results)} tracks matching '{query}':\n\n"
            for i, track in enumerate(results, 1):
                duration_min = round(track['Milliseconds'] / 60000, 1) if track['Milliseconds'] else "Unknown"
                result += f"{i}. **{track['TrackName']}** by {track['ArtistName']}\n"
                result += f"   Album: {track['AlbumTitle']}, Genre: {track['Genre']}\n"
                result += f"   Duration: {duration_min} min, Price: ${track['UnitPrice']}\n\n"

            return result

        @tool
        def get_purchase_history(customer_id: int, limit: int = 10) -> str:
            """Get customer's recent music purchases for context."""
            purchases = self.db.get_customer_purchases(customer_id)

            if not purchases:
                return "No purchase history found."

            recent_purchases = purchases[:limit]
            result = f"Your recent purchases ({len(recent_purchases)} most recent):\n\n"

            for i, purchase in enumerate(recent_purchases, 1):
                result += f"{i}. **{purchase['TrackName']}** by {purchase['ArtistName']}\n"
                result += f"   Purchased: {purchase['InvoiceDate']}, Price: ${purchase['UnitPrice']}\n\n"

            return result

        # Store tools as instance attributes
        self.tools = [
            get_customer_genres,
            get_genre_recommendations,
            search_music,
            get_purchase_history
        ]

    def get_system_message(self) -> SystemMessage:
        return SystemMessage(content="""
You are a music recommendation specialist for a digital music store. You help customers discover new music based on their preferences and purchase history.

Your capabilities:
- Analyze customer music preferences from purchase history
- Provide personalized recommendations based on genres they enjoy
- Search for specific tracks, artists, or albums
- Help customers explore new genres similar to their favorites

Guidelines:
- Always be enthusiastic about music and discovery
- Provide detailed information about recommended tracks
- Consider the customer's listening patterns when making suggestions
- Explain why you're recommending specific tracks
- Ask follow-up questions to better understand their preferences
- If they haven't purchased much, help them explore popular genres

When making recommendations, consider:
- Their past purchase genres and frequency
- Similar artists and styles
- New releases in their preferred genres
- Cross-genre recommendations for musical exploration

Always use the available tools to get accurate, personalized data for each customer.
""")

    def get_tools(self):
        """Return the list of tools available to this agent."""
        return self.tools