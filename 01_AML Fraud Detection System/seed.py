from datetime import datetime, timedelta
import requests 
import random
from api.db import get_connection, create_db_table

def generate_random_db_transactions(n=200):
	Sender_account = ["5459041199","8046287266","4097215471","8779641997"]
	Receiver_account = ["6609117934","5645102341","7658096331","3423799664"]
	Payment_currency = ["UK pounds","Moroccan dirham"]
	Received_currency = ["UK pounds","Pakistani rupee"]
	Payment_type = ["Debit card","Cheque","ACH","Credit Card","Cross-border"]
	Sender_bank_location = ["UK", "Spain","Italy"]
	Receiver_bank_location = ["UK", "USA", "UAE"]
	base_time = datetime.now()
	data = []
	for _ in range(n):
		Amount = round(random.expovariate(1/100),2)
		sender = random.choice(Sender_account)
		receiver = random.choice(Receiver_account)
		p_curr = random.choice(Payment_currency)
		r_curr = random.choice(Received_currency)
		payment_t = random.choice(Payment_type)
		sender_b_l = random.choice(Sender_bank_location)
		re_b_l = random.choice(Receiver_bank_location)
		timestampdt = base_time - timedelta(minutes = random.randint(1,5000))
		timestamp = timestampdt.strftime("%Y-%m-%d %H:%M:%S")
		fraud_probability = None
		alert  =  None
		data.append((time, date, timestamp, sender, receiver, Amount,
		p_curr, r_curr, sender_b_l, re_b_l, payment_t,
		fraud_probability, alert))

	return data

def inserting_db(n = 200):

    create_db_table()

    conn = get_connection()
    cur = conn.cursor()

    data = generate_random_db_transactions(n)

    cur.executemany("""
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
        VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

    print(f"Seeded {n} historical transactions into DB")


url = "http://127.0.0.1:8000/predict-batch"

def generate_transaction_api():
        return {
                "Sender_account" :
         random.choice([
        "5459041199","8046287266","4097215471","8779641997"]),
        "Amount" : round(random.expovariate(1/100),2),
        "timestamp": (
    datetime.now() - timedelta(minutes=random.randint(1, 500))
).strftime("%Y-%m-%d %H:%M:%S"),
        "Receiver_account" : random.choice(["6609117934","5645102341",
"7658096331","3423799664"]),
        "Payment_currency" : random.choice(["UK pounds","Moroccan dirham"]),
        "Received_currency" :random.choice(["UK pounds","Pakistani rupee"]),
        "Payment_type": random.choice(["Debit card","Cheque","ACH","Credit Card"]),
        "Sender_bank_location" :random.choice(["UK", "Spain","Italy"]),
        "Receiver_bank_location" : random.choice(["UK", "USA", "UAE"])
        }

def send_transactions_to_api(n=10):
		txs = [generate_transaction_api() for _ in range(n) ]
		response= requests.post(url, json = txs)
		
		print(response.text)
		
if __name__ == "__main__":
	

	send_transactions_to_api(10)
	
