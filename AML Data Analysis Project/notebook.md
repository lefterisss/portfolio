# AML-Fraud Detection-Analysis 

## **Βriefly Introduction**: 
This dataset is highly imbalanced, with legitimate trasnactions vastly outnumbering fraudulent ones. 
Such imbalance poses significant challenges for model evaluation, as the extremely small proportion of laundering cases can lead to low values for metrics such as precision and recall.
To address this issue, this project explores the use of unsupervised anomaly detection, specifically using the Isolation Forest model. Unlike supervised classifiers, anomaly detection methods do not rely on labeled fraud examples during training. Instead, they learn the general structure of normal transaction behaviour and identify observations that deviate significantly from this pattern.
## Isolation Forest: The model is an unsupervised learning detector which assigns an anomaly score to each transaction. Isolation Forest produces anomaly scores rather than calibrated fraud probabilities. 
Consequently, the model optimizes detection of statistical rarity

## DATASET DESCRIPTION

 We should get more familiar with our Dataset. The dataset is about Financial Transactions.It consists data on Amount being transferred,as well as Payment Type,Sender/Receiver bank Account,Location.

**Note:**The dataset can be found at the following url.
(Να βαλω τα credits του dataset επειδη το πηρα copy)
                                                                 
 
 PROBLEM DEFINITION
This project explores whether unsupervised anomaly detection can effectively prioritize suspicious AML transactions.
======================================================================================================================
# PROJECT STEPS

## Import the necessary libraries

**For numerical calculations**

`import numpy as np`

to handle the dataset and use of a dataframe

`import pandas as pd`

for plotting

`import matplotlib.pyplot as plt`

`import seaborn as sns`

`import warnings`

`warnings.filterwarnings('ignore')`


## EXPLORATORY DATA ANALYSIS

In this step we explore the dataset to have a better understanding. It provides useful insights of our dataset.

Have a comprehended picture of our data

`Data.head()`

`data.shape()`

Understand what kinds of data we have

`data.dtype()`

Addressing the Issue of Missing Values

Checking for missing values .

`print(data.isnull().sum())`

column transformations 

Because our Linear Model is valid only with numerical values we transform the categorical to numerical values 

Find the numerical columns and visualize them

`numer_col = [col for col in data.columns if data[col].dtype!='O']`

`print(numer_col)`

Observe the first 5 rows of the numerical columns

`data[numer_col].head()`

find the percentage of each categorical value 

`frequency= data['area'].value_counts(normalize=True)`

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
    
