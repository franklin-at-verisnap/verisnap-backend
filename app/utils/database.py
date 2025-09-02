"""
Database operations
"""
import sqlite3
import os
from typing import List, Dict, Any, Optional
from config import settings


class DatabaseService:
    """Service for database operations"""
    
    def __init__(self):
        self.db_path = settings.database_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create database if it doesn't exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create captures table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY,
                userid TEXT,
                os TEXT,
                lat TEXT,
                lon TEXT,
                altitude TEXT,
                original_signature TEXT,
                watermarked_signature TEXT,
                relevant_obj_tags TEXT,
                device_time TEXT,
                server_time TEXT,
                t_score TEXT,
                os_score TEXT,
                in_vs_out_score TEXT,
                day_vs_night_score TEXT,
                altitude_score TEXT,
                device_score TEXT,
                barometerData TEXT,
                magnetometerData TEXT,
                heading TEXT,
                vscore TEXT,
                baro_score TEXT,
                magnetometer_score TEXT,
                cheat_penalty TEXT
            )
        ''')
        
        # Create index for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_captures_server_time 
            ON captures (server_time DESC)
        ''')
        
        # Ensure new columns exist (idempotent)
        try:
            cursor.execute("PRAGMA table_info('captures')")
            cols = {row[1] for row in cursor.fetchall()}
            for col in ["baro_score", "magnetometer_score", "cheat_penalty"]:
                if col not in cols:
                    cursor.execute(f"ALTER TABLE captures ADD COLUMN {col} TEXT")
        except Exception as e:
            print(f"[DB] schema check/alter failed: {e}")
        
        conn.commit()
        conn.close()
    
    def insert_capture(self, capture_dict: Dict[str, Any]) -> bool:
        """Insert a new capture record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            insert_query = '''
            INSERT INTO captures (
                id, userid, os, lat, lon, altitude, original_signature, watermarked_signature,
                relevant_obj_tags, device_time, server_time, t_score, os_score,
                in_vs_out_score, day_vs_night_score, altitude_score, device_score,
                barometerData, magnetometerData, heading, vscore,
                baro_score, magnetometer_score, cheat_penalty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            capture_data = (
                str(capture_dict.get('id', '')),
                str(capture_dict.get('userid', '')),
                str(capture_dict.get('os', '')),
                str(capture_dict.get('lat', '')),
                str(capture_dict.get('lon', '')),
                str(capture_dict.get('altitude', '')),
                str(capture_dict.get('original_signature', '')),
                str(capture_dict.get('watermarked_signature', '')),
                str(capture_dict.get('relevant_obj_tags', '')),
                str(capture_dict.get('device_time', '')),
                str(capture_dict.get('server_time', '')),
                str(capture_dict.get('t_score', '')),
                str(capture_dict.get('os_score', '')),
                str(capture_dict.get('in_vs_out_score', '')),
                str(capture_dict.get('day_vs_night_score', '')),
                str(capture_dict.get('altitude_score', '')),
                str(capture_dict.get('device_score', '')),
                str(capture_dict.get('barometerData', '')),
                str(capture_dict.get('magnetometerData', '')),
                str(capture_dict.get('heading', '')),
                str(capture_dict.get('vscore', '')),
                str(capture_dict.get('baro_score', '')),
                str(capture_dict.get('magnetometer_score', '')),
                str(capture_dict.get('cheat_penalty', '')),
            )
            
            cursor.execute(insert_query, capture_data)
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error inserting capture: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def query_records_by_hash(self, avghash: str) -> List[Dict[str, Any]]:
        """Query records by watermarked signature hash"""
        records_list = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql_query = "SELECT * FROM captures WHERE watermarked_signature = ?"
            cursor.execute(sql_query, (avghash,))
            records = cursor.fetchall()
            
            # Convert to dictionaries
            for record in records:
                columns = [col[0] for col in cursor.description]
                record_dict = dict(zip(columns, record))
                records_list.append(record_dict)
            
        except sqlite3.Error as e:
            print("Error querying the database:", e)
        finally:
            if 'conn' in locals():
                conn.close()
        
        return records_list
    
    def get_capture_by_id(self, userid: str, image_id: str) -> Optional[Dict[str, Any]]:
        """Get capture record by user ID and image ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM captures WHERE userid = ? AND id = ?",
                (userid, image_id)
            )
            row = cursor.fetchone()
            
            if row:
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            return None
            
        except sqlite3.Error as e:
            print("Error getting capture:", e)
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_captures_by_user(self, userid: str, page: int = 1, per_page: int = 20) -> List[Dict[str, Any]]:
        """Get captures for a user with pagination"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            offset = (page - 1) * per_page
            cursor.execute(
                "SELECT id, server_time FROM captures "
                "WHERE userid = ? "
                "ORDER BY server_time DESC "
                "LIMIT ? OFFSET ?",
                (userid, per_page, offset)
            )
            rows = cursor.fetchall()
            
            return [{"id": row[0], "server_time": row[1]} for row in rows]
            
        except sqlite3.Error as e:
            print("Error getting user captures:", e)
            return []
        finally:
            if 'conn' in locals():
                conn.close()
