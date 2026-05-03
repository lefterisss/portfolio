# 📊 AML Fraud Detection System 

## **Βriefly Introduction**: 

This project implements **an end-to-end Anti-Money Laundering (AML) fraud detection system**, covering:

- Feature engineering on transactional data
- Supervised machine learning model training
- Statistical evaluation (precision, recall, validation strategy)
- Production-style deployment via FastAPI
- Real-time batch inference pipeline

The system simulates how financial institutions analyze incoming transactions and flag potentially suspicious activity.

## Architecture

### High-Level Flow

- Raw transactions  <br />

- Feature Engineering (features.py)  <br />
        
- Encoding (OneHotEncoder) <br />
        
- Model (Decision Tree) <br />
        
- Probability + Threshold <br />
        
- Alert Decision <br />
        
- Stored in Database (SQLite)


## System Components

**Training Pipeline** (Kaggle / local)

- Data preprocessing <br />   
- Feature engineering <br />
- Model training <br />
- Threshold optimization <br />
- Artifact export <br />  


**API Layer** (FastAPI)

- Receives transactions

- Builds features in real-time

- Applies trained model

- Returns predictions

- Stores results

- Database (SQLite)

- Stores transaction history

- Enables behavioral features
