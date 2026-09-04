# convert_markets_fintech_fixed.py
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import re

print("🚀 Starting markets_db migration to Cloudflare D1...")
print("="*50)

# Connect to your local PostgreSQL
conn = psycopg2.connect(
    dbname="markets_db",
    user="markets_user",
    host="localhost"
)
cur = conn.cursor(cursor_factory=RealDictCursor)

# Define your ENUMs for reference
ENUMS = {
    "ExitReason": ["TAKE_PROFIT", "STOP_LOSS", "MANUAL"],
    "KycStatus": ["PENDING", "VERIFIED", "REJECTED", "EXPIRED"],
    "TradeSide": ["BUY", "SELL"],
    "TradeStatus": ["PENDING", "EXECUTED", "CANCELLED", "FAILED"],
    "UserRole": ["USER", "ADMIN", "MODERATOR", "ANALYST"],
    "UserStatus": ["ACTIVE", "INACTIVE", "SUSPENDED", "CLOSED"]
}

print("📋 ENUMs detected:")
for enum_name, values in ENUMS.items():
    print(f"  • {enum_name}: {', '.join(values)}")

# Get all tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name;
""")
tables = [row['table_name'] for row in cur.fetchall()]
print(f"\n📊 Found {len(tables)} tables")

# Get column information for all tables
table_columns = {}
for table in tables:
    cur.execute("""
        SELECT 
            column_name,
            data_type,
            udt_name,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """, (table,))
    table_columns[table] = cur.fetchall()

# Get foreign keys
cur.execute("""
    SELECT
        conrelid::regclass AS table_name,
        confrelid::regclass AS referenced_table,
        conname AS constraint_name,
        pg_get_constraintdef(oid) AS definition
    FROM pg_constraint
    WHERE contype = 'f'
    ORDER BY conrelid::regclass;
""")
foreign_keys = cur.fetchall()
print(f"🔗 Found {len(foreign_keys)} foreign key relationships")

# Get indexes
cur.execute("""
    SELECT
        tablename,
        indexname,
        indexdef
    FROM pg_indexes
    WHERE schemaname = 'public'
    ORDER BY tablename, indexname;
""")
indexes = cur.fetchall()
print(f"📑 Found {len(indexes)} indexes")

# Generate SQLite schema
print("\n📝 Generating SQLite schema...")
sqlite_schema = [
    "-- markets_db migration to Cloudflare D1",
    "-- Generated for fintech application",
    "-- Enable foreign key support (critical for data integrity)",
    "PRAGMA foreign_keys = ON;",
    "",
    "BEGIN TRANSACTION;",
    ""
]

# Create tables
for table in tables:
    print(f"  Processing table: {table}")
    sqlite_schema.append(f"-- Table: {table}")
    sqlite_schema.append(f"CREATE TABLE IF NOT EXISTS \"{table}\" (")
    
    col_defs = []
    primary_key = None
    
    for col in table_columns[table]:
        col_name = col['column_name']
        pg_type = col['udt_name']
        is_nullable = col['is_nullable']
        default = col['column_default']
        
        # Handle ENUMs - convert to TEXT with CHECK constraint later
        if pg_type in ENUMS:
            sqlite_type = 'TEXT'
        else:
            # Map PostgreSQL types to SQLite
            if pg_type in ('int4', 'int8', 'int2', 'integer', 'bigint', 'smallint'):
                sqlite_type = 'INTEGER'
            elif pg_type in ('text', 'varchar', 'bpchar', 'char'):
                sqlite_type = 'TEXT'
            elif pg_type in ('bool', 'boolean'):
                sqlite_type = 'INTEGER'
            elif pg_type in ('numeric', 'decimal', 'float4', 'float8'):
                sqlite_type = 'REAL'
            elif pg_type in ('timestamp', 'timestamptz', 'date', 'time'):
                sqlite_type = 'TEXT'
            elif pg_type in ('json', 'jsonb'):
                sqlite_type = 'TEXT'
            elif pg_type in ('bytea'):
                sqlite_type = 'BLOB'
            else:
                sqlite_type = 'TEXT'
        
        # Check if this is the primary key
        if col_name == 'id':
            primary_key = col_name
            col_def = f'    "{col_name}" {sqlite_type} PRIMARY KEY'
            if sqlite_type == 'INTEGER':
                col_def += ' AUTOINCREMENT'
        else:
            col_def = f'    "{col_name}" {sqlite_type}'
        
        # Add NOT NULL constraint
        if is_nullable == 'NO' and col_name != 'id':
            col_def += ' NOT NULL'
        
        # Handle defaults
        if default and col_name != 'id':
            if 'nextval' in str(default):
                pass
            elif default == 'true':
                col_def += " DEFAULT 1"
            elif default == 'false':
                col_def += " DEFAULT 0"
            elif 'now()' in str(default):
                col_def += " DEFAULT CURRENT_TIMESTAMP"
            elif default.startswith("'") and default.endswith("'"):
                col_def += f" DEFAULT {default}"
            elif default.isdigit():
                col_def += f" DEFAULT {default}"
        
        col_defs.append(col_def)
    
    sqlite_schema.append(',\n'.join(col_defs))
    sqlite_schema.append(");\n")
    
    # Add CHECK constraints for ENUM columns
    for col in table_columns[table]:
        col_name = col['column_name']
        pg_type = col['udt_name']
        if pg_type in ENUMS:
            enum_values = "', '".join(ENUMS[pg_type])
            check_name = f"check_{table}_{col_name}"
            sqlite_schema.append(f"CREATE TRIGGER {check_name}_insert")
            sqlite_schema.append(f"BEFORE INSERT ON \"{table}\"")
            sqlite_schema.append(f"WHEN NEW.\"{col_name}\" NOT IN ('{enum_values}') AND NEW.\"{col_name}\" IS NOT NULL")
            sqlite_schema.append("BEGIN")
            sqlite_schema.append(f"    SELECT RAISE(ABORT, 'Invalid {col_name} value. Must be one of: {enum_values}');")
            sqlite_schema.append("END;")
            sqlite_schema.append("")
            
            sqlite_schema.append(f"CREATE TRIGGER {check_name}_update")
            sqlite_schema.append(f"BEFORE UPDATE ON \"{table}\"")
            sqlite_schema.append(f"WHEN NEW.\"{col_name}\" NOT IN ('{enum_values}') AND NEW.\"{col_name}\" IS NOT NULL")
            sqlite_schema.append("BEGIN")
            sqlite_schema.append(f"    SELECT RAISE(ABORT, 'Invalid {col_name} value. Must be one of: {enum_values}');")
            sqlite_schema.append("END;")
            sqlite_schema.append("")

# Add indexes
if indexes:
    sqlite_schema.append("-- Indexes for performance")
    for idx in indexes:
        idx_def = idx['indexdef']
        idx_def = re.sub(r'CREATE INDEX (\w+) ON public\.(\w+)', r'CREATE INDEX IF NOT EXISTS \1 ON "\2"', idx_def)
        idx_def = idx_def.replace('USING btree', '')
        sqlite_schema.append(idx_def + ';')
    sqlite_schema.append("")

# Add foreign key constraints
if foreign_keys:
    sqlite_schema.append("-- Foreign key relationships preserved")
    for fk in foreign_keys:
        sqlite_schema.append(f"-- FK: {fk['constraint_name']} on {fk['table_name']} references {fk['referenced_table']}")

sqlite_schema.append("\nCOMMIT;")

# Write schema file
with open('markets_db_schema.sqlite', 'w') as f:
    f.write('\n'.join(sqlite_schema))

print(f"\n✅ Schema file created: markets_db_schema.sqlite")

# Export data
print("\n📤 Exporting data...")
with open('markets_db_data.sqlite', 'w') as f:
    f.write("-- markets_db data\nBEGIN TRANSACTION;\n\n")
    
    # Export in order to respect foreign keys
    export_order = ['User'] + [t for t in tables if t != 'User']
    
    for table in export_order:
        if table not in tables:
            continue
            
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cur.fetchone()['count']
        print(f"  {table}: {count} rows")
        
        if count == 0:
            continue
            
        cur.execute(f'SELECT * FROM "{table}"')
        rows = cur.fetchall()
        
        for row in rows:
            cols = []
            vals = []
            for key, value in row.items():
                cols.append(f'"{key}"')
                if value is None:
                    vals.append('NULL')
                elif isinstance(value, bool):
                    vals.append('1' if value else '0')
                elif isinstance(value, (int, float)):
                    vals.append(str(value))
                elif isinstance(value, (dict, list)):
                    # JSON data - escape single quotes properly
                    json_str = json.dumps(value)
                    escaped_json = json_str.replace("'", "''")
                    vals.append(f"'{escaped_json}'")
                else:
                    # String data - escape single quotes
                    str_val = str(value).replace("'", "''")
                    vals.append(f"'{str_val}'")
            
            insert = f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({", ".join(vals)});\n'
            f.write(insert)
        
        f.write('\n')
    
    f.write("COMMIT;\n")

print("✅ Data export complete!")

# Combine schema and data
with open('markets_db_complete.sqlite', 'w') as outfile:
    with open('markets_db_schema.sqlite', 'r') as schema:
        outfile.write(schema.read())
    with open('markets_db_data.sqlite', 'r') as data:
        outfile.write(data.read())

print(f"\n🎉 Complete SQLite file created: markets_db_complete.sqlite")
print(f"\n📊 Summary:")
print(f"  • Tables: {len(tables)}")
print(f"  • ENUMs: {len(ENUMS)}")
print(f"  • Foreign Keys: {len(foreign_keys)}")
print(f"  • Indexes: {len(indexes)}")

conn.close()