import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import os

# Configurações do banco de destino (PostgreSQL)
user = os.environ["DB_USER"]
password = os.environ["DB_PASS"]
host = os.environ["DB_HOST"]
port = os.environ["DB_PORT"]
dbname = os.environ["DB_NAME"]

pg_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}" 

pg_engine = create_engine(pg_url)
# Conectar ao SQLite
sqlite_conn = sqlite3.connect("/home/ec2-user/crypto_hype.db")

# Iterar e carregar em blocos
chunk_size = 50000
offset = 0
total_inserted = 0

while True:
    query = f"""
        SELECT * FROM ohlcv
        LIMIT {chunk_size} OFFSET {offset}
    """
    chunk = pd.read_sql_query(query, sqlite_conn)

    if chunk.empty:
        break

    chunk.to_sql("ohlcv", con=pg_engine, index=False, if_exists="append")
    offset += chunk_size
    total_inserted += len(chunk)
    print(f"{total_inserted} registros inseridos...")

sqlite_conn.close()
print(f"\n✅ Importação concluída: {total_inserted} registros inseridos no total.")




