"""
Database connection and query utilities for Chinook music store.
Includes customer data isolation and security controls.
"""

import sqlite3
from typing import List, Any, Optional
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", "./chinook.db")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def authenticate_customer(self, email: str) -> Optional[dict[str, Any]]:
        """Authenticate customer by email and return customer info."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT CustomerId, FirstName, LastName, Email, Company,
                       City, State, Country, SupportRepId
                FROM Customer
                WHERE Email = ? COLLATE NOCASE
            """, (email,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_customer_purchases(self, customer_id: int) -> List[dict[str, Any]]:
        """Get all purchases for a specific customer."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.InvoiceId, i.InvoiceDate, i.Total,
                       il.Quantity, il.UnitPrice,
                       t.Name as TrackName, t.Composer,
                       al.Title as AlbumTitle,
                       ar.Name as ArtistName,
                       g.Name as Genre
                FROM Invoice i
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                JOIN Track t ON il.TrackId = t.TrackId
                LEFT JOIN Album al ON t.AlbumId = al.AlbumId
                LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
                LEFT JOIN Genre g ON t.GenreId = g.GenreId
                WHERE i.CustomerId = ?
                ORDER BY i.InvoiceDate DESC
            """, (customer_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_customer_genres(self, customer_id: int) -> List[dict[str, Any]]:
        """Get customer's preferred genres based on purchase history."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT g.Name as Genre,
                       COUNT(*) as PurchaseCount,
                       SUM(il.Quantity * il.UnitPrice) as TotalSpent
                FROM Invoice i
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                JOIN Track t ON il.TrackId = t.TrackId
                LEFT JOIN Genre g ON t.GenreId = g.GenreId
                WHERE i.CustomerId = ? AND g.Name IS NOT NULL
                GROUP BY g.GenreId, g.Name
                ORDER BY PurchaseCount DESC, TotalSpent DESC
                LIMIT 5
            """, (customer_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_recommendations_by_genre(self, genres: List[str], customer_id: int, limit: int = 10) -> List[dict[str, Any]]:
        """Get track recommendations based on genres, excluding already purchased tracks."""
        placeholders = ','.join('?' for _ in genres)

        with self.get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT DISTINCT t.TrackId, t.Name as TrackName,
                       al.Title as AlbumTitle, ar.Name as ArtistName,
                       g.Name as Genre, t.UnitPrice
                FROM Track t
                LEFT JOIN Album al ON t.AlbumId = al.AlbumId
                LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
                LEFT JOIN Genre g ON t.GenreId = g.GenreId
                WHERE g.Name IN ({placeholders})
                AND t.TrackId NOT IN (
                    SELECT DISTINCT il.TrackId
                    FROM Invoice i
                    JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                    WHERE i.CustomerId = ?
                )
                ORDER BY RANDOM()
                LIMIT ?
            """, (*genres, customer_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_invoice_details(self, invoice_id: int, customer_id: int) -> Optional[dict[str, Any]]:
        """Get invoice details for a specific customer (security: customer_id validation)."""
        with self.get_connection() as conn:
            # First verify the invoice belongs to the customer
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM Invoice
                WHERE InvoiceId = ? AND CustomerId = ?
            """, (invoice_id, customer_id))

            if cursor.fetchone()[0] == 0:
                return None  # Invoice doesn't belong to this customer

            # Get invoice details
            cursor = conn.execute("""
                SELECT i.InvoiceId, i.InvoiceDate, i.BillingAddress,
                       i.BillingCity, i.BillingState, i.BillingCountry,
                       i.Total,
                       il.Quantity, il.UnitPrice,
                       t.Name as TrackName,
                       al.Title as AlbumTitle,
                       ar.Name as ArtistName
                FROM Invoice i
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                JOIN Track t ON il.TrackId = t.TrackId
                LEFT JOIN Album al ON t.AlbumId = al.AlbumId
                LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
                WHERE i.InvoiceId = ?
            """, (invoice_id,))

            rows = cursor.fetchall()
            if not rows:
                return None

            # Structure the response
            invoice_data = dict(rows[0])
            invoice_data['items'] = []
            for row in rows:
                invoice_data['items'].append({
                    'track_name': row['TrackName'],
                    'album_title': row['AlbumTitle'],
                    'artist_name': row['ArtistName'],
                    'quantity': row['Quantity'],
                    'unit_price': row['UnitPrice']
                })

            return invoice_data

    def search_tracks(self, query: str, limit: int = 20) -> List[dict[str, Any]]:
        """Search tracks by name, artist, or album."""
        search_term = f"%{query}%"

        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT t.TrackId, t.Name as TrackName,
                       al.Title as AlbumTitle, ar.Name as ArtistName,
                       g.Name as Genre, t.UnitPrice, t.Milliseconds
                FROM Track t
                LEFT JOIN Album al ON t.AlbumId = al.AlbumId
                LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
                LEFT JOIN Genre g ON t.GenreId = g.GenreId
                WHERE t.Name LIKE ? COLLATE NOCASE
                   OR al.Title LIKE ? COLLATE NOCASE
                   OR ar.Name LIKE ? COLLATE NOCASE
                ORDER BY ar.Name, al.Title, t.Name
                LIMIT ?
            """, (search_term, search_term, search_term, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_support_rep_info(self, support_rep_id: int) -> Optional[dict[str, Any]]:
        """Get support representative information."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT EmployeeId, FirstName, LastName, Title, Email, Phone
                FROM Employee
                WHERE EmployeeId = ?
            """, (support_rep_id,))

            row = cursor.fetchone()
            return dict(row) if row else None