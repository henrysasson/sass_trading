import psycopg2

# Conecta ao banco padrão "postgres"
conn = psycopg2.connect(
    host="sass-crypto-db.c7q26iqyun16.us-east-2.rds.amazonaws.com",
    user="hsasson",
    password="Maxbra104#",  # ← substitua aqui
    port=5432,
    dbname="postgres"
)

conn.autocommit = True
cur = conn.cursor()

try:
    cur.execute("CREATE DATABASE crypto_data")
    print("Banco 'crypto_data' criado com sucesso.")
except psycopg2.errors.DuplicateDatabase:
    print("Banco 'crypto_data' já existe.")
finally:
    cur.close()
    conn.close()


