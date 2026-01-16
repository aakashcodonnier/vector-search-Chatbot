
import mysql.connector
from contextlib import contextmanager


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # UPDATE HERE
        database="case_studies_db"
    )
@contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper cleanup."""
    conn = None
    try:
        conn = get_connection()
        yield conn
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()
