import sqlite3
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH =  BASE_DIR / "aml.db"

def get_connection():
	return sqlite3.connect(DB_PATH)



def get_db():
	conn = get_connection()
	try:
		yield conn
	finally:
		conn.close()

def create_db_table(conn):
	cursor  = conn.cursor()
	cursor.execute("""
	CREATE TABLE IF NOT EXISTS transactions ( 
 	id INTEGER PRIMARY KEY AUTOINCREMENT,
	timestamp TEXT,	
	Sender_account INTEGER,
        Receiver_account INTEGER,
        Amount REAL,
        Payment_currency TEXT,
        Received_currency TEXT,
        Sender_bank_location TEXT,
        Receiver_bank_location TEXT,
        Payment_type TEXT,
        fraud_probability REAL,
        alert INTEGER) 
	""")
	conn.commit()
	

def get_senders_history(conn, senders):
	if not senders:
		return pd.DataFrame()
	
	placeholders =  ",".join(["?"] * len(senders))
	
	query = f"""
	SELECT Amount, Sender_account, Receiver_account, timestamp
	FROM transactions
	WHERE Sender_account IN ({placeholders})
	ORDER BY timestamp DESC
	"""	
	df = pd.read_sql_query(query, conn, params = senders)

	return df

def save_transaction(conn, tx, fraud_probability, alert):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO transactions (
            timestamp,
            Sender_account,
            Receiver_account,
            Amount,
            Payment_currency,
            Received_currency,
            Sender_bank_location,
            Receiver_bank_location,
            Payment_type,
            fraud_probability,
            alert
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tx["timestamp"],
        tx["Sender_account"],
        tx["Receiver_account"],
        tx["Amount"],
        tx["Payment_currency"],
        tx["Received_currency"],
        tx["Sender_bank_location"],
        tx["Receiver_bank_location"],
        tx["Payment_type"],
        float(fraud_probability),
        int(alert)
    ))

if __name__ == "__main__" :
	
	create_db_table()
 
