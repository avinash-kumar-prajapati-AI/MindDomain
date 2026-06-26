import sqlite3
import os
from src.config import config
from src.models import PipelineResponse

def get_db_connection():
    db_path = config.storage.sqlite_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT NOT NULL,
            rating TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log(response: PipelineResponse, rating: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (query, answer, source, rating) VALUES (?, ?, ?, ?)",
        (response.raw_query, response.answer, response.source, rating)
    )
    conn.commit()
    conn.close()
