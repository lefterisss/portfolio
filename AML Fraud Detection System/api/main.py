from fastapi import FastAPI, Depends,HTTPException
from pydantic import BaseModel
from transferred_encoder import Encoder
import joblib
import json
import logging
import sys
import __main__


from features import build_features
from db import (
    get_db,
    create_db_table,
    get_senders_history,
    save_transaction
)


app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__main__.Encoder = Encoder

if "__mp_main__" in sys.modules:
    sys.modules["__mp_main__"].Encoder = Encoder


model = joblib.load("../artifacts/model.pkl")
encoder = joblib.load("../artifacts/encoder.pkl")
senderbehaviour = joblib.load("../artifacts/senderbehaviour.pkl")

with open("../artifacts/threshold.json") as f:
    THRESHOLD = json.load(f)["threshold"]

with open("../artifacts/features.json") as f:
        feature_info = json.load(f)


FEATURE_COLUMNS  = feature_info["feature_columns"]

with open("../artifacts/risky_flags.json") as f:
    risky_flags = json.load(f)

class Transaction(BaseModel):
    Sender_account : str
    Receiver_account: str
    timestamp: str
    Amount : float
    Payment_currency : str
    Received_currency : str
    Sender_bank_location : str
    Receiver_bank_location: str
    Payment_type: str

@app.post("/predict-batch")
def predict_batch(txs: list[Transaction], conn=Depends(get_db)):
    try:
        logger.info(f"START batch prediction: {len(txs)} transactions")

        create_db_table(conn)

        tx_dicts = [tx.model_dump() for tx in txs]
        senders = [tx["Sender_account"] for tx in tx_dicts]

        all_history_df = get_senders_history(conn, senders)
        results = []

        for i, tx_dict in enumerate(tx_dicts):
            sender = tx_dict["Sender_account"]

            logger.info(f"Processing transaction {i + 1} for sender {sender}")

            sender_history = all_history_df[
                all_history_df["Sender_account"] == sender
            ].copy()

            features_df = build_features(
                tx=tx_dict,
                history_df=sender_history,
                senderbehaviour=senderbehaviour,
                risky_flags=risky_flags
            )

            encoded_df = encoder.transform(features_df)

            X = encoded_df.reindex(
                columns=FEATURE_COLUMNS,
                fill_value=0
            )

            prob = model.predict_proba(X)[0, 1]
            alert = int(prob >= THRESHOLD)

            logger.info(f"prob={prob:.10f}, alert={alert}")

            save_transaction(conn, tx_dict, prob, alert)

            results.append({
                "Sender_account": sender,
                "Receiver_account": tx_dict["Receiver_account"],
                "Amount": tx_dict["Amount"],
                "fraud_probability": float(prob),
                "threshold": float(THRESHOLD),
                "alert": alert
            })

        conn.commit()
        logger.info("Batch commit DONE")

        return {
            "n_transactions": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"ERROR in /predict-batch: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


