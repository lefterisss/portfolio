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

## Tech Stack

### Core

- Python </br>
- Pandas/NumPy
- Scikit-learn

### API

- FastAPI
- Uvicorn
- Pydantic

### Stoage 

- SQLite

### ML Artifacts

- joblib(model, encoder)
- JSON (threshold, featuresm risk flags)

## Training Phase

### Objective

Train a model to estimate:

                        P(fraud∣transaction features)


### Model

- Decision Tree Classifier
- GridSearchCV for hyperparameter tuning
- Optimization focused on **precision**

### Data Split Strategy

- Time-based split:

  - Train /Validation / Test

- Prevents data  leakage

### Feature Engineering

Includes:

- Time-based features (hour, day, weekend) </br>
- Behavioral features (sender history) </br>
- Statistical aggregates (rolling windows) </br>
- Risk flags (high-risk combinations) </br>
- Log-transformed amounts </br>

    Time-based features (hour, day, weekend)

    Behavioral features (sender history)

    Statistical aggregates (rolling windows)

    Risk flags (high-risk combinations)

    Log-transformed amounts

### Model Performance

Metric               Value 

**Precision**            0.34

**Recall**               0.30

#### Interpretation 

- The model detects some fraud patterns </br>
- It misses a portion of fraudulent transactions </br>
- Performance is affected by class imbalance </br>


## Class Imbalance impact

Fraud is rare:
                      P(fraud)≪P(normal)

This leads to: 

 - Conservative predictions
 - Few high-risk regions in feature space
 - Difficuly detecting rare patterns

## API Inference Pipeline

 ### Endpoint

 **POST /predict-batch**

 **Accepts**:
[
 {
 "Sender_account": "...",
    "Receiver_account": "...",
    "Amount": ...,
    "timestamp": "...",
    ...
}
]
 
