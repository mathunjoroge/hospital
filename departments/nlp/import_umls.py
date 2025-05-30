import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from config import POSTGRES_DSN  # Ensure POSTGRES_DSN is in config.py

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_rrf_file(file_path, table_name, columns, filter_sab=None):
    """Import an RRF file into a PostgreSQL table."""
    try:
        # Read RRF file (pipe-delimited, no header)
        logger.info(f"Reading {file_path}")
        df = pd.read_csv(file_path, sep='|', header=None, low_memory=False, encoding='utf-8')
        
        # Assign column names based on UMLS documentation
        df.columns = columns
        
        # Filter for SNOMEDCT_US and ICD10CM if specified
        if filter_sab and 'SAB' in df.columns:
            df = df[df['SAB'].isin(filter_sab)]
            logger.info(f"Filtered {file_path} to {len(df)} rows for SABs: {filter_sab}")
        
        # Handle missing values
        df = df.fillna('')
        
        # Convert to list of tuples for bulk insert
        records = [tuple(row) for row in df.to_numpy()]
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**POSTGRES_DSN)
        cursor = conn.cursor()
        
        # Generate SQL insert query
        cols = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(columns))
        query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        
        # Bulk insert
        logger.info(f"Inserting {len(records)} rows into {table_name}")
        execute_values(cursor, query, records, page_size=1000)
        
        conn.commit()
        logger.info(f"Successfully imported {file_path} into {table_name}")
        
    except Exception as e:
        logger.error(f"Failed to import {file_path}: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    # Define column names for each RRF file based on UMLS documentation
    mrconso_columns = [
        'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
        'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
    ]
    mrsty_columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF']
    mrdef_columns = ['CUI', 'AUI', 'ATUI', 'SATUI', 'SAB', 'DEF', 'SUPPRESS', 'CVF']
    
    # Filter for clinical diagnosis vocabularies
    filter_sabs = ['SNOMEDCT_US', 'ICD10CM']
    
    # Import MRCONSO
    import_rrf_file(
        file_path='MRCONSO.RRF',  # Update with your path
        table_name='mrconso',
        columns=mrconso_columns,
        filter_sab=filter_sabs
    )
    
    # Import MRSTY
    import_rrf_file(
        file_path='MRSTY.RRF',  # Update with your path
        table_name='mrsty',
        columns=mrsty_columns
    )
    
    # Import MRDEF
    import_rrf_file(
        file_path='MRDEF.RRF',  # Update with your path
        table_name='mrdef',
        columns=mrdef_columns,
        filter_sab=filter_sabs
    )

if __name__ == "__main__":
    main()