# AML-Fraud Detection-Analysis 

## **Βriefly Introduction**: 
This dataset is highly **imbalanced**, with legitimate trasnactions vastly outnumbering fraudulent ones. 
Such imbalance poses significant challenges for model evaluation, as the extremely small proportion of laundering cases can lead to low values for metrics such as precision and recall.
To address this issue, this project explores the use of unsupervised anomaly detection, specifically using the Isolation Forest model. Unlike supervised classifiers, anomaly detection methods do not rely on labeled fraud examples during training. Instead, they learn the general structure of normal transaction behaviour and identify observations that deviate significantly from this pattern.
## Isolation Forest:
The model is an unsupervised learning detector which assigns an anomaly score to each transaction. Isolation Forest produces anomaly scores rather than calibrated fraud probabilities. 
Consequently, the model optimizes detection of statistical rarity

## DATASET DESCRIPTION

 We should get more familiar with our Dataset. The dataset is about Financial Transactions. It consists data on Amount being transferred,as well as Payment Type,Sender/Receiver bank Account,Location.

**Note:**The dataset can be found at the following url.
(Να βαλω τα credits του dataset επειδη το πηρα copy)
                                                                 
 
## PROBLEM DEFINITION
This project does prioritization of suspicious AML transactions using anomaly scores.
**======================================================================================================================
# PIPELINE
The pipeline follows these steps :

Data Loading <br />
Exploratory Data Analysis <br />
Train/Test Split <br />
Feature Engineering <br />
Encoding <br />
Correlation <br />
Train/Evaluation <br />

**Exploratory Data Analysis**

It focuses on:

i) Visualise the probability distribution to understand our classes.
ii) Check the feature **Amount**. The reason i decided it to do only on this
because it has a strong signal of abnomal behaviour. Visualise the distribution,
scaling it and observe the distribution.
iii)Boxplot to understand the correlation of Log Amount with our class. The 50% of the values are
inside the box.
iv) Conditional Probability between the amount and the **Is_laundering** class which has been cut to 10 equally sized groups.

Train/Test Split
Split the time based on a cut-off value and not randomly splitting because our need to have a time order.

## Feature Enginnering

Divided up to three different classes(Basic Feature Engineering, Sender_behavioural_features
- In Basic Feature Engineering:
  We analyse the time and the Amount to create different features with more signal info for our DataFrame
- Sender Behavioural Features:
  i)Making analysis for each sender with **groupby** and the **aggregates** so we have an account behaviour analysis(eg The count aggregate answers how active is the sender). That analysis is suitable for situations such as Smurfing and for Fraud detection.
- Time Feature Engineering:
  i)This feature helps detect potential structuring behavior by analyzing transactions within short time windows rather than evaluating each transaction in isolation.
  ii) train_df_small["Amount"].quantile(0.2), means that a threshold based on the 20th percentile of the transaction amount distribution was used to identify small transactions. This data-driven cutoff captures the lower tail of the distribution, where structuring behavior often occurs.
 iii)We merged again the Df (combined_df) to include the new feature to both of them. To do the rolling(), it was required to do set_index(
"timestamp") so the functions to be applied to **a moving time window**(10min).

### Encoding
Because of our categorical features : **Payment_currency**, **Received_currency**, **Sender_bank_location**, **Receiver_bank_location**, **Payment_type**,
to include it them in our model we do fit() to learn the categorical values and enc.transform(df[self.cat_cols]) ,to transform them into a numeric form. 
With **encoded_df = pd.DataFrame(encoded,columns = feature_names,index = df.ind**ex), we included the numerical columns on a DataFrame and after we put it side by side with our df to have the new suitable columns.

### Correlation
  Explore how much the feature correlate to each other.
  The code inside the **Correlation** class does the following:
  i)We pick the main numeric columns we wanted to focus.
  Thats why we didnt use the df.select_dtypes(include = "number") because it would include features such as then **encoding one** which have **dummy values**.
  ii) Calculation of the correlation matrix via corr() to corr_df[existing_cols]
  iii)Heatmap to visualize the corr results.
  iv)Relation of the numerical features with the class **"Is_laundering"**. 
  The **(num_df.corr()[target_col].drop(target_col)
                                .sort_values(key=abs, ascending=False))** it only examines the correlation of **each feature** with only the **target_col**. This is not important in this project because Isolation Forest is not trained with the class values. But for an overall EDA understanding we apply it.
  

## OUR MODEL - ISOLATION FOREST
We start by: <br />
i) Making our X_train , X_test with a list of features which we define as an argument (to have the opportunity to use again this class function). <br />
ii)Training our model with X_train. <br />
iii)Using the Isolation Forest function score_samples on X_train, X_test to have the anomaly scores. 
iv) train_scores = -score <br />
    test_scores = -score2, means the biggest number the more suspicious. <br />
v) train_scores = how much suspicious are the past (X_train) transactions.
### THRESHOLDING
i) With `np.quantile(train_Scores,0.99)` we alert the 0.1% transactions as more suspicious.
ii) Comparing `test_scores >= threshold` we make a policy to label as **is_laundering** (main_y_predict) the one's with test_score bigger than the threshold
iii) Generating the confusion matrix to start investigating the number of TN(True negatives), FP(false positives), FN(false negatives) , TP(true positives).
  
## Evaluation of the model


