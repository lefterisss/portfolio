# AML-Fraud Detection-Analysis 

## **Βriefly Introduction**: 
This dataset is highly **imbalanced**, with legitimate trasnactions vastly outnumbering fraudulent ones. 
Such imbalance poses significant challenges for model evaluation, as the extremely small proportion of laundering cases can lead to low values for metrics such as precision and recall.
To address this issue, this project explores the use of unsupervised anomaly detection, specifically using the Isolation Forest model. Unlike supervised classifiers, anomaly detection methods do not rely on labeled fraud examples during training. Instead, they learn the general structure of normal transaction behaviour and identify observations that deviate significantly from this pattern.
## Isolation Forest:
The model is an unsupervised learning detector which assigns an anomaly score to each transaction. Isolation Forest produces anomaly scores rather than calibrated fraud probabilities. 
Consequently, the model optimizes detection of statistical rarity

## DATASET DESCRIPTION

 We should get more familiar with our Dataset. The dataset is about Financial Transactions.It consists data on Amount being transferred,as well as Payment Type,Sender/Receiver bank Account,Location.

**Note:**The dataset can be found at the following url.
(Να βαλω τα credits του dataset επειδη το πηρα copy)
                                                                 
 
## PROBLEM DEFINITION
This project does prioritization of suspicious AML transactions using anomaly scores.
**======================================================================================================================
# PIPELINE
The pipeline follows these steps :

Data Loading
Exploratory Data Analysis
Train/Test Split
Feature Engineering
Encoding
Correlation
Train/Evaluation
==========================================================================================================================================

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

Feature Enginnering

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

  
  

normalize=True)`

`print(frequency)`

Update the categorical column with value transformations

`data['area_encoded'] = data['area'].map(frequency)`

`print(data[['area','area_encoded']])`

Drop method for the area column

`data_new= data.drop(["area"],axis=1)`

Note: The moment we've done our values as numerical the ordinary column area is removed because of the categorical content 

Plots to understand the relationship between the variables.Also in diagonal is a distribution of each variable

`sns.pairplot(data_new)`

`plt.show()`

We can see that some variables have strong linear relationship between them. Its important to understand this as a first step to understand a potentially problem with our estimated values .


## MULTICOLLINEARITY AND CORRELATION MATRIX

The correlation matrix expresses the relationship of two predictors. This matrix involves a row and columns table in which each cell represents the correlation coefficient of the two variables. The correlation coefficient ranges from -1 to 1 with -1 indicating perfect negative relationship and 1 perfect positive relationship. The visualization of the correlation matrix is an important step before the building of the machine learning model because you can gain a better understanding about what’s most important for your model.

**The Variance Inflation Factor** quantifies how much the variance of a specific regression coefficient for a feature is inflated due to multicollinearity with other features.

So by running the `data_new.corr()` function we can observe that each cell shows a value between 0 and 1. We put a threshold to 0.8 value to find the problematic pairs which make the model less accurate.


`problematic_pairs = correlation_matrix[(correlation_matrix.abs()> threshold) & (correlation_matrix.abs()<1)]`

`print(problematic_pairs)`

So the value such as the pair between active enteprises and births which is 0.99373 implies a very strong relationship between them that we have to act because it suggests that they contain similar (redundant) information.

This observation is important to identify later problems that occurs to our coefficients of the model, thus making it less accurate.

## Evaluation of the model

A factor which implies how accurate is our model is R_2. This is the coefficient of determination a measure of the goodness of fit of a model. A value of 1 indicates that the regression predictions perfectly fit the data. The formula of r is calculated as 

                                                            
                                  R2= SSTotal​SSExplained​​ = 1−SSTotal​SSResidual​​

Formula definitions:

SSExplained​/SSTotal​ is the proportion of the total variance in y that can be "explained" by X2,X3,…,XpX2​,X3​,…,Xp​ (predictors)
R2R2 ranges from 0 to 1:

    If R2=0R2=0: None of the variance in X1X1​ is explained by X2,X3,…,XpX2​,X3​,…,Xp​; all the variance is left in SSResidualSSResidual​.
    If R2=1R2=1: All of the variance in X1X1​ is explained by X2,X3,…,XpX2​,X3​,…,Xp​, meaning SSResidual=0SSResidual​=0, and X1X1​ can be perfectly predicted from the other predictors.
    
