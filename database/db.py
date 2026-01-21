
#!/usr/bin/env python3
"""
Database connection module for Dr. Robert Young's semantic search system

This module provides database connection functionality for the semantic search system.
"""

# Third-party imports
import mysql.connector


def get_connection():
    """
    Establish and return a connection to the MySQL database
    
    This function creates a connection to the case studies database with
    the required parameters for the semantic search system.
    
    Returns:
        mysql.connector.connection.MySQLConnection: Database connection object
    """
    return mysql.connector.connect(
        host="localhost",        # Database host
        user="root",           # Database username
        password="",           # Database password (empty assumed for local dev)
        database="case_studies_db"  # Target database name
    )
