import numpy as np
import pandas as pd


def build_features(tx, history_df, senderbehaviour, risky_flags):
	df = pd.DataFrame([tx]).copy()

	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["Log_Amount"] = np.log1p(df["Amount"])
	df["hour"] = df["timestamp"].dt.hour
	df["day_of_week"] = df["timestamp"].dt.weekday
	df["weekend"] = df["day_of_week"].isin([5,6]).astype(int)

	df["high_risk_hour"] = df["hour"].isin([5,6,7]).astype(int)
	df["high_risk_day"] = df["day_of_week"].isin([2,3,4]).astype(int)
	
	high_risk_payment_types = [ 
		"Cash Deposit",
		"Cash Withdrawal",
		"Cross-border"
	]

	df["high_risk_payment_type"] = (
		df["Payment_type"].isin(high_risk_payment_types).astype(int)
	)

	df["combined_risk_flag"] = (
		df["high_risk_hour"] * df["high_risk_day"] * df["high_risk_payment_type"]
	)
	for col, info in risky_flags.items():
		bin_edges = info["bin_edges"]

		df["amount_decile"] = pd.cut(
			df["Amount"],
			bins = bin_edges,
			labels = False,
			include_lowest = True
		)

		df["amount_decile"] = df["amount_decile"].fillna(0).astype(int)

		risky_pairs = set(
			(str(v), int(dec)) for v, dec in info["risky_pairs"]
		)

		pairs = list(zip(df[col].astype(str),df["amount_decile"].astype(int)))

		df[f"{col}_combo_risk_flag"] = [
			int(pair in risky_pairs) for pair in pairs
		]
	
	sender = tx["Sender_account"]

	if sender in  senderbehaviour.index:
		profile = senderbehaviour.loc[sender]
		
		df["sender_transaction_count"] = profile["sender_transaction_count"]
		df["sender_mean_Amount"] = profile["sender_mean_Amount"]
		df["sender_std_Amount"]	=  profile["sender_std_Amount"]
		df["receiver_bank_loc_n"] = profile["receiver_bank_loc_n"]

	else:
		df["sender_transaction_count"] = 0
		df["sender_mean_Amount"] = df["Amount"].iloc[0]
		df["sender_std_Amount"] = 1
		df["receiver_bank_loc_n"] = 0


	if history_df is None  or len(history_df) == 0:

		df["sum_amount_last_1hr"] = 0
		df["sum_amount_last_24hr"] = 0
		df["tx_count_last_1hr"] = 0
		df["tx_count_last_24hr"] = 0
		df["unique_receivers_last_1hr"] = 0
		df["unique_receivers_last_24hr"] = 0
		df["sender_hist_mean_amount"] = df["sender_mean_Amount"]
		df["amount_to_sender_mean_ratio"] = (
    df["Amount"] / (df["sender_hist_mean_amount"] + 1e-9))
	else:
		history_df = history_df.copy()
		history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
		
		current_time = df.loc[0, "timestamp"]
		
		past_1hr = history_df[
			(history_df["timestamp"] < current_time ) & 
			(history_df["timestamp"] >= current_time - pd.Timedelta(hours = 1))
		]
		
		past_24hr = history_df[
			(history_df["timestamp"] >= current_time - pd.Timedelta(hours = 24))

		]
		
		df["sum_amount_last_1hr"] = past_1hr["Amount"].sum()
		df["sum_amount_last_24hr"] = past_24hr["Amount"].sum()

		df["tx_count_last_1hr"] = len(past_1hr)
		df["tx_count_last_24hr"] = len(past_24hr)

		df["unique_receivers_last_1hr"] = past_1hr["Receiver_account"].nunique()
		df["unique_receivers_last_24hr"] = past_24hr["Receiver_account"].nunique()

		df["sender_hist_mean_amount"] = history_df["Amount"].mean()

		df["amount_to_sender_mean_ratio"] = (
			df["Amount"] / (df["sender_hist_mean_amount"] + 1e-9)
		)

	return df
