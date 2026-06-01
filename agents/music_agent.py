"""
Music Recommendation Agent - Specialized for music discovery and personalized recommendations.
"""

from typing import Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from database import DatabaseManager
from helpers.system_messages import SYSTEM_MESSAGES

class MusicRecommendationAgent:
    def __init__(self, db_manager: DatabaseManager, llm: ChatOpenAI):
        self.db = db_manager
        self.llm = llm
        self.authenticated_customer_id = None  # Will be set when processing authenticated requests

        # Create tool functions with closure over self.db
        @tool
        def get_customer_genres(customer_id: int) -> str:
            """Get customer's preferred music genres based on purchase history."""
            genres = self.db.get_customer_genres(customer_id)
            if not genres:
                return SYSTEM_MESSAGES["NO_PURCHASE_HISTORY_PREFERENCES"]

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
                return SYSTEM_MESSAGES["NO_PURCHASE_HISTORY"]

            recent_purchases = purchases[:limit]
            result = f"Your recent purchases ({len(recent_purchases)} most recent):\n\n"

            for i, purchase in enumerate(recent_purchases, 1):
                result += f"{i}. **{purchase['TrackName']}** by {purchase['ArtistName']}\n"
                result += f"   Purchased: {purchase['InvoiceDate']}, Price: ${purchase['UnitPrice']}\n\n"

            return result

        @tool
        def get_personalized_recommendations() -> str:
            """Get personalized album recommendations based on your purchase history and explore new genres."""
            if not self.authenticated_customer_id:
                return "You need to be authenticated to get personalized recommendations. Please provide your email address."

            # Get customer's preferred genres
            customer_genres = self.db.get_customer_genres(self.authenticated_customer_id)

            if not customer_genres:
                # New customer - provide popular recommendations across different genres (limit to 3)
                popular_genres = ["Rock", "Pop", "Jazz"]
                recommendations = []
                for genre in popular_genres:
                    genre_recs = self.db.get_recommendations_by_genre([genre], self.authenticated_customer_id, 1)
                    recommendations.extend(genre_recs)

                if recommendations:
                    result = "**Welcome! Here are 3 popular albums across different genres to get you started:**\n\n"
                    for i, track in enumerate(recommendations[:3], 1):
                        result += f"{i}. **{track['TrackName']}** by {track['ArtistName']}\n"
                        result += f"   Album: {track['AlbumTitle']}, Genre: {track['Genre']}, Price: ${track['UnitPrice']}\n"
                        result += f"   *Popular choice in {track['Genre']} for new listeners*\n\n"

                    result += "*Start with these genres and I'll learn your preferences for better recommendations!*"
                    return result
                else:
                    return "I'm having trouble finding recommendations right now. Try searching for specific artists or genres you like!"

            # Existing customer - provide personalized recommendations
            top_genres = [g['Genre'] for g in customer_genres[:3]]  # Top 3 genres
            all_customer_genres = [g['Genre'] for g in customer_genres]

            # Get customer's purchase history to show which albums led to recommendations
            purchases = self.db.get_customer_purchases(self.authenticated_customer_id)

            # Get recommendations from their favorite genres (limit to 3)
            favorite_recommendations = self.db.get_recommendations_by_genre(top_genres, self.authenticated_customer_id, 3)

            # Get recommendations from genres they haven't explored much (cross-genre discovery, limit to 3)
            all_genres = ["Rock", "Pop", "Jazz", "Classical", "Blues", "Electronic", "Hip Hop", "Country", "R&B", "Alternative"]
            unexplored_genres = [g for g in all_genres if g not in all_customer_genres][:2]
            exploration_recommendations = []
            if unexplored_genres:
                exploration_recommendations = self.db.get_recommendations_by_genre(unexplored_genres, self.authenticated_customer_id, 3)

            # Combine recommendations
            result = "**Your Personalized Music Recommendations:**\n\n"

            # Add favorite genre recommendations with purchase context
            if favorite_recommendations:
                result += f"**Based on your favorite genres ({', '.join(top_genres)}):**\n"

                # Group purchased albums by genre for context
                genre_purchases = {}
                for purchase in purchases:
                    genre = purchase.get('Genre', 'Unknown')
                    if genre in top_genres:
                        if genre not in genre_purchases:
                            genre_purchases[genre] = []
                        album_artist = f"{purchase['AlbumTitle']} by {purchase['ArtistName']}"
                        if album_artist not in genre_purchases[genre]:
                            genre_purchases[genre].append(album_artist)

                for i, track in enumerate(favorite_recommendations, 1):
                    track_genre = track['Genre']
                    result += f"{i}. **{track['TrackName']}** by {track['ArtistName']}\n"
                    result += f"   Album: {track['AlbumTitle']}, Genre: {track['Genre']}, Price: ${track['UnitPrice']}\n"

                    # Show which purchased albums from this genre influenced the recommendation
                    if track_genre in genre_purchases and genre_purchases[track_genre]:
                        purchased_albums = genre_purchases[track_genre][:2]  # Show max 2 examples
                        result += f"   *Recommended because you enjoyed {track_genre}: {', '.join(purchased_albums)}*\n\n"
                    else:
                        result += f"   *Recommended based on your interest in {track_genre}*\n\n"

            # Add exploration recommendations
            if exploration_recommendations:
                result += f"\n**Explore new genres ({', '.join(unexplored_genres)}):**\n"
                for i, track in enumerate(exploration_recommendations, 1):
                    result += f"{i}. **{track['TrackName']}** by {track['ArtistName']}\n"
                    result += f"   Album: {track['AlbumTitle']}, Genre: {track['Genre']}, Price: ${track['UnitPrice']}\n"
                    result += f"   *New genre suggestion to expand your musical horizons*\n\n"

            # Add purchase history context
            result += f"\n*Recommendations based on your {len(customer_genres)} favorite genres. "
            result += f"Your top genre is {customer_genres[0]['Genre']} with {customer_genres[0]['PurchaseCount']} purchases.*"

            return result

        # Store tools as instance attributes
        self.tools = [
            get_customer_genres,
            get_genre_recommendations,
            search_music,
            get_purchase_history,
            get_personalized_recommendations
        ]

    def get_system_message(self) -> SystemMessage:
        auth_context = ""
        if self.authenticated_customer_id:
            auth_context = f"""
IMPORTANT: The customer is authenticated (Customer ID: {self.authenticated_customer_id}).
When they ask for music recommendations, ALWAYS start by using the 'get_personalized_recommendations' tool to provide recommendations based on their purchase history and explore new genres.
This tool automatically provides personalized recommendations without needing customer_id parameters.
"""

        return SystemMessage(content=f"""
You are a music recommendation specialist for a digital music store. You help customers discover new music based on their preferences and purchase history.

{auth_context}

Your capabilities:
- Provide personalized recommendations based on purchase history (for authenticated users)
- Analyze customer music preferences from purchase history
- Search for specific tracks, artists, or albums
- Help customers explore new genres similar to their favorites

Guidelines:
- Always be enthusiastic about music and discovery
- For authenticated users: ALWAYS use 'get_personalized_recommendations' first when they ask for recommendations
- Provide detailed information about recommended tracks
- Consider the customer's listening patterns when making suggestions
- Explain why you're recommending specific tracks
- Ask follow-up questions to better understand their preferences
- Help customers explore both familiar and new genres

When making recommendations, prioritize:
1. For authenticated users: Use their purchase history for personalized suggestions
2. Cross-genre recommendations for musical exploration
3. Similar artists and styles to their preferences
4. Popular tracks in genres they enjoy

Always use the available tools to get accurate, personalized data for each customer.
""")

    def set_authenticated_customer(self, customer_id: int):
        """Set the authenticated customer ID for context-aware tool responses."""
        self.authenticated_customer_id = customer_id

    def get_tools(self):
        """Return the list of tools available to this agent."""
        return self.tools