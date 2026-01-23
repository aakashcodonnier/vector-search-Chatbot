#!/usr/bin/env python3
"""
Database connection module for Dr. Robert Young's semantic search system

This module provides database connection functionality for the semantic search system.
It manages MySQL connections for storing and retrieving articles with their embeddings.

Configuration:
- Host: localhost (assumes local MySQL server)
- User: root (default MySQL user)
- Password: "" (empty password for local development)
- Database: case_studies_db (specific database for this project)

Note: In production, use environment variables or configuration files
for sensitive database credentials.
"""

# Third-party imports
import mysql.connector  # MySQL database connector


def get_connection():
    """
    Establish and return a connection to the MySQL database
    
    This function creates a connection to the case studies database with
    the required parameters for the semantic search system. The connection
    uses dictionary cursor for easier data manipulation and includes proper
    connection parameters for the project's database structure.
    
    Returns:
        mysql.connector.connection.MySQLConnection: Database connection object
        
    Note:
        - Uses dictionary cursor for named column access
        - Connection parameters are hardcoded for development
        - In production, use connection pooling and environment variables
    """
    return mysql.connector.connect(
        host="localhost",        # Database host (local development)
        user="root",           # Database username (MySQL default)
        password="",           # Database password (empty for local dev)
        database="case_studies_db"  # Target database name for scraped articles
    )
