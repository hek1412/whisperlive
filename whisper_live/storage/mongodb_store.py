"""
MongoDB archival storage for WhisperLive sessions.
Handles cold data storage for completed sessions with full history.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoDBArchiveStore:
    """
    MongoDB-based archival storage for completed sessions.

    Collections:
        sessions - Full session documents with metadata and transcript

    Document structure:
        {
            "_id": session_id,
            "created_at": ISODate,
            "updated_at": ISODate,
            "closed_at": ISODate,
            "status": "closed",
            "config": {...},
            "segments": [{...}],
            "consolidated_blocks": [{...}],
            "speakers": ["Speaker1", "Speaker2"],
            "summary": {...},
            "metadata": {...}
        }
    """

    def __init__(self, mongodb_url: str, database_name: str = "whisperlive"):
        """
        Initialize MongoDB archive store.

        Args:
            mongodb_url: MongoDB connection URL (e.g., mongodb://localhost:27017)
            database_name: Database name (default: whisperlive)
        """
        try:
            self.client = MongoClient(mongodb_url)
            self.db = self.client[database_name]
            self.sessions = self.db.sessions

            # Create indexes
            self._create_indexes()

            # Test connection
            self.client.admin.command('ping')
            logger.info(f"[MongoDBStore] Connected to MongoDB at {mongodb_url}")
        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Create indexes for efficient queries."""
        try:
            # Index on created_at for time-based queries
            self.sessions.create_index([("created_at", DESCENDING)])

            # Index on status for filtering active/closed sessions
            self.sessions.create_index([("status", ASCENDING)])

            # Index on closed_at for archival queries
            self.sessions.create_index([("closed_at", DESCENDING)])

            # Compound index for status + created_at
            self.sessions.create_index([
                ("status", ASCENDING),
                ("created_at", DESCENDING)
            ])

            logger.debug("[MongoDBStore] Indexes created successfully")
        except PyMongoError as e:
            logger.warning(f"[MongoDBStore] Failed to create indexes: {e}")

    def archive_session(self, session_id: str, session_data: Dict) -> bool:
        """
        Archive a complete session to MongoDB.

        Args:
            session_id: Session ID
            session_data: Complete session data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document
            document = {
                "_id": session_id,
                "created_at": session_data.get("created_at", datetime.utcnow()),
                "updated_at": datetime.utcnow(),
                "closed_at": session_data.get("closed_at", datetime.utcnow()),
                "status": session_data.get("status", "closed"),
                "config": session_data.get("config", {}),
                "segments": session_data.get("segments", []),
                "consolidated_blocks": session_data.get("consolidated_blocks", []),
                "speakers": list(session_data.get("speakers", [])),
                "summary": session_data.get("summary"),
                "metadata": session_data.get("metadata", {})
            }

            # Upsert document
            self.sessions.replace_one(
                {"_id": session_id},
                document,
                upsert=True
            )

            logger.info(f"[MongoDBStore] Archived session {session_id}")
            return True

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to archive session {session_id}: {e}")
            return False

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a session from MongoDB.

        Args:
            session_id: Session ID

        Returns:
            Session document or None if not found
        """
        try:
            document = self.sessions.find_one({"_id": session_id})
            if document:
                # Convert ObjectId to string if present
                document["session_id"] = document.pop("_id")
                return document
            return None

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to get session {session_id}: {e}")
            return None

    def get_session_transcript(self, session_id: str) -> Optional[List[Dict]]:
        """
        Get consolidated transcript for a session.

        Args:
            session_id: Session ID

        Returns:
            List of consolidated blocks or None if not found
        """
        try:
            document = self.sessions.find_one(
                {"_id": session_id},
                {"consolidated_blocks": 1}
            )

            if document:
                return document.get("consolidated_blocks", [])
            return None

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to get transcript for {session_id}: {e}")
            return None

    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get summary for a session.

        Args:
            session_id: Session ID

        Returns:
            Summary dictionary or None if not found
        """
        try:
            document = self.sessions.find_one(
                {"_id": session_id},
                {"summary": 1}
            )

            if document:
                return document.get("summary")
            return None

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to get summary for {session_id}: {e}")
            return None

    def update_session_summary(self, session_id: str, summary: Dict) -> bool:
        """
        Update or add summary to an archived session.

        Args:
            session_id: Session ID
            summary: Summary dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.sessions.update_one(
                {"_id": session_id},
                {
                    "$set": {
                        "summary": summary,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"[MongoDBStore] Updated summary for session {session_id}")
                return True
            else:
                logger.warning(f"[MongoDBStore] Session {session_id} not found for summary update")
                return False

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to update summary for {session_id}: {e}")
            return False

    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        List sessions with optional filtering.

        Args:
            status: Filter by status (e.g., "closed")
            limit: Maximum number of sessions to return
            skip: Number of sessions to skip (for pagination)

        Returns:
            List of session documents
        """
        try:
            query = {}
            if status:
                query["status"] = status

            cursor = self.sessions.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)

            sessions = []
            for doc in cursor:
                doc["session_id"] = doc.pop("_id")
                sessions.append(doc)

            return sessions

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from MongoDB.

        Args:
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.sessions.delete_one({"_id": session_id})

            if result.deleted_count > 0:
                logger.info(f"[MongoDBStore] Deleted session {session_id}")
                return True
            else:
                logger.warning(f"[MongoDBStore] Session {session_id} not found for deletion")
                return False

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to delete session {session_id}: {e}")
            return False

    def search_sessions(
        self,
        speaker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search sessions by criteria.

        Args:
            speaker: Filter by speaker name
            start_date: Filter sessions created after this date
            end_date: Filter sessions created before this date
            limit: Maximum number of results

        Returns:
            List of matching session documents
        """
        try:
            query = {}

            if speaker:
                query["speakers"] = speaker

            if start_date or end_date:
                date_query = {}
                if start_date:
                    date_query["$gte"] = start_date
                if end_date:
                    date_query["$lte"] = end_date
                query["created_at"] = date_query

            cursor = self.sessions.find(query).sort("created_at", DESCENDING).limit(limit)

            sessions = []
            for doc in cursor:
                doc["session_id"] = doc.pop("_id")
                sessions.append(doc)

            return sessions

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to search sessions: {e}")
            return []

    def get_statistics(self) -> Dict:
        """
        Get statistics about archived sessions.

        Returns:
            Dictionary with statistics
        """
        try:
            total_sessions = self.sessions.count_documents({})
            closed_sessions = self.sessions.count_documents({"status": "closed"})

            # Get date range
            oldest = self.sessions.find_one(sort=[("created_at", ASCENDING)])
            newest = self.sessions.find_one(sort=[("created_at", DESCENDING)])

            stats = {
                "total_sessions": total_sessions,
                "closed_sessions": closed_sessions,
                "oldest_session": oldest.get("created_at") if oldest else None,
                "newest_session": newest.get("created_at") if newest else None
            }

            return stats

        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close MongoDB connection."""
        try:
            self.client.close()
            logger.info("[MongoDBStore] Connection closed")
        except PyMongoError as e:
            logger.error(f"[MongoDBStore] Error closing connection: {e}")
