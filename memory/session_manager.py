import sqlite3
import uuid
import pytz
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re

class SessionManager:
    def __init__(self, db_path: str = "jarvis_session.db", context_limit: int = 5200):
        self.db_path = db_path
        self.context_limit = context_limit
        self.current_session_id = None
        self.session_timeout_minutes = 30

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                summary TEXT,
                message_count INTEGER DEFAULT 0,
                status TEXT DEFAULT "active",
                total_tokens INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                assistant_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')

        conn.commit()
        conn.close()
        

    def get_or_create_session(self) -> str:
        """Get current active session or create new one"""
        if self.current_session_id and self._is_session_active():
            return self.current_session_id
        
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions (session_id, status) VALUES (?, ?)",
            (session_id, "active")
        )
        conn.commit()
        conn.close()

        self.current_session_id = session_id
        return session_id

    def _is_session_active(self) -> bool:
        if not self.current_session_id:
            print("[DEBUG] No current session.")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
            (self.current_session_id,)
        )
        result = cursor.fetchone()
        conn.close()

        if not result:
            print("[DEBUG] No messages yet; session is active.")
            return True

        try:
            # Parse the timestamp from the DB (stored in UTC)
            last_message_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        
            # Convert the UTC time to IST
            utc_zone = pytz.utc
            ist_zone = pytz.timezone('Asia/Kolkata')
        
            # Set the timestamp to UTC timezone first
            last_message_time = utc_zone.localize(last_message_time)
        
            # Convert it to IST
            last_message_time_ist = last_message_time.astimezone(ist_zone)
        
            # Get the current time in IST
            now_ist = datetime.now(ist_zone)

            # Check if the session is inactive based on the timeout
            inactive = now_ist - last_message_time_ist > timedelta(minutes=self.session_timeout_minutes)
            print(f"[DEBUG] Last message at {last_message_time_ist}, inactive: {inactive}")
            return not inactive
        except Exception as e:
            print(f"[ERROR] Timestamp parsing failed: {e}")
            return True  # default to active if unsure


    def add_message(self, user_query: str, assistant_response: str, tokens_used: int = 0):
        """Add a message to the current session"""
        session_id = self.get_or_create_session()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO messages (session_id, user_query, assistant_response, tokens_used) VALUES (?, ?, ?, ?)",
            (session_id, user_query, assistant_response, tokens_used)  # Fixed: was token_used
        )

        cursor.execute(
            "UPDATE sessions SET message_count = message_count + 1, total_tokens = total_tokens + ? WHERE session_id = ?",
            (tokens_used, session_id)
        )

        conn.commit()
        conn.close()

    def should_end_session(self, user_query: str) -> bool:
        """Determine if session should end based on conversation patterns"""
    
        # Only very explicit ending phrases
        ending_phrases = [
            r'\b(goodbye|bye|see you later|farewell)\s*$',  # Must be at end
            r'\b(that\'s all for now|that\'s it for today)\b',
            r'\b(end session|close session|terminate session)\b',
            r'\b(signing off|logging off)\b'
        ]
    
        # Only very explicit topic changes
        topic_change_phrases = [
            r'\b(let\'s talk about something completely different)\b',
            r'\b(new topic:|different topic:)\b',
            r'\b(changing subjects?:)\b'
        ]
    
        user_lower = user_query.lower().strip()
    
        # Only end if it's a very explicit ending
        for pattern in ending_phrases:
            if re.search(pattern, user_lower):
                return True
    
        # Only end for very explicit topic changes
        for pattern in topic_change_phrases:
            if re.search(pattern, user_lower):
                return True
    
        return False
    
    def get_session_context(self) -> str:
        """Build context for current session with sliding window if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all messages for this session
        cursor.execute(
            "SELECT user_query, assistant_response, tokens_used FROM messages WHERE session_id = ? ORDER BY timestamp",
            (self.current_session_id,)
        )
        messages = cursor.fetchall()
        conn.close()
        
        if not messages:
            return ""
        
        # Calculate total tokens
        total_tokens = sum(msg[2] if msg[2] else 0 for msg in messages)
        
        if total_tokens <= self.context_limit:
            # Include all messages
            return self._format_full_context(messages)
        else:
            # Use sliding window
            return self._format_sliding_window_context(messages, self.current_session_id)
        
    def _format_full_context(self, messages: List[Tuple]) -> str:
        """Format all messages as context"""
        context = "Previous conversation:\n\n"
        for user_query, assistant_response, _ in messages:
            context += f"User: {user_query}\n"
            context += f"Jarvis: {assistant_response}\n\n"
        return context
    
    def _format_sliding_window_context(self, messages: List[Tuple], session_id: str) -> str:
        """Format context using sliding window approach"""
        # Keep last 8 exchanges
        recent_messages = messages[-8:]
        
        context = ""
        
        # Add session summary if available (from RAG or previous part of session)
        if len(messages) > 8:
            context += "[Earlier in this conversation: Discussion covered technical implementation details and optimization strategies]\n\n"
        
        context += "Recent conversation:\n\n"
        for user_query, assistant_response, _ in recent_messages:
            context += f"User: {user_query}\n"
            context += f"Jarvis: {assistant_response}\n\n"
        
        return context
    
    def end_session(self, summary: str = None) -> Dict:
        """End current session and return session data for RAG storage"""
        if not self.current_session_id:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update session status
        cursor.execute(
            "UPDATE sessions SET status = ?, end_time = ?, summary = ? WHERE session_id = ?",
            ('ended', datetime.now().isoformat(), summary, self.current_session_id)
        )
        
        # Get session data for RAG storage
        cursor.execute(
            "SELECT session_id, start_time, end_time, message_count, total_tokens FROM sessions WHERE session_id = ?",
            (self.current_session_id,)
        )
        session_data = cursor.fetchone()
        
        # Get all messages for summary generation
        cursor.execute(
            "SELECT user_query, assistant_response FROM messages WHERE session_id = ? ORDER BY timestamp",
            (self.current_session_id,)
        )
        messages = cursor.fetchall()
        
        conn.commit()
        conn.close()
        
        # Reset current session
        session_id = self.current_session_id
        self.current_session_id = None
        
        return {
            'session_id': session_id,
            'start_time': session_data[1] if session_data else None,
            'end_time': session_data[2] if session_data else None,
            'message_count': session_data[3] if session_data else 0,
            'total_tokens': session_data[4] if session_data else 0,
            'messages': messages,
            'summary': summary
        }
    
    def get_all_messages(self) -> List[Tuple[str, str]]:
        """Return all (user_query, assistant_response) tuples from the current session"""
        if not self.current_session_id:
            return []
    
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_query, assistant_response FROM messages WHERE session_id = ? ORDER BY timestamp",
            (self.current_session_id,)
        )
        messages = cursor.fetchall()
        conn.close()
        return messages


    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        if not self.current_session_id:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_count, total_tokens FROM sessions WHERE session_id = ?",
            (self.current_session_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'session_id': self.current_session_id,
                'message_count': result[0],
                'total_tokens': result[1]
            }
        return {}