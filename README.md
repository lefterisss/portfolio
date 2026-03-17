# Multiple-Linear-Regression-analysis
This project is about prediction on a target variable and Residual Analysis to evaluate  the model


## Multiple Linear Regression:

**Goal of the Project:** The aim of the project is to apply a linear regression model for prediction purposes on a target variable 

## Table of Contents

**Definition of Multiple Linear Regression**

Our linear model is a mathematical representation that describes the relationship between a target variable and multiple predictors (independent variables). It assumes that the target variable is influenced by the predictors through a linear combination of their effects.

The mathematical equation is is expressed as:

                                y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​+ϵ
                                
 The **independent variable** is y which we aim to predict the values of it. 
 
 **Dependent variables :** At the right of the equation are the multiple predictors multiplied by a unique coefficient β.These are characteristics( also known as features) of a phenomenon.
 
 **Coefficient term=** Is an estimate of the change in the dependent variable  y, that results from a change in the independent variable. In other words it shows when a change happens in a predictor how much change the y variable.
 
**Error term:** he model includes an error term to account for the variability in y that cannot be explained by the predictors.

**Model Calculations and mathematical concepts:** The estimations and other calculations are based in Matrix equations. Our Multiple linear model can be written in matrix form as :

                                             Y = B*X + ε 

X = design matrix. It consist as columns our predictors and the rows are the overall observations (of the dataset). 

## DATASET DESCRIPTION

 We should get more familiar with our Dataset. The dataset is about Business Demographics and Survival Rates, Borough.It consists data on enterprise births, deaths, active enterprises across boroughs.
 
**Data features/Characteristics:** We should describe our dataset for better understanding. 
Our Features are:
                    Births= Number of births in a given year
                    year= The specific year of the phenomenon.
                    deaths= Number of deaths in a given year
                    active enterprise= A time series number of active enterprises.
                    Birth percentage= The rate of the births in a given year
                    
**Note:** The  original dataset consisted of two data subsets, but we selected one for our purpose. We removed the irrelevant (for our purpose) column of survival rates.
                                                                  
                                                                 
The dataset can be found at the following url.
 [London Datastore](https://data.london.gov.uk/dataset/business-demographics-and-survival-rates-borough)
 
 
 PROBLEM DEFINITION
 A machine learning model is built to predict the outcome of the target variable based on two or more predictors of our business demographics dataset.
Our goal is to estimate the outcome of the birth_rates based on the dependent variables (the feature we explained earlier). The selection of the model is the OLS model from the library statsmodels.api which we will apply.We mention it cause there is also other model building class such as LinearRegression which serves different purposes but have lot similarities.

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

## FEATURE SCALING & MODEL ASSUMPTIONS 

Although linear regression does not strictly require feature scaling, large differences in magnitude between variables (e.g. year vs active_enterprises) can affect numerical stability and interpretation of coefficients.

In this project, scaling was not applied explicitly, but this choice is justified because:

The model is interpreted mainly through coefficients and statistical tests

The main goal is understanding relationships rather than optimizing predictive performance

## MULTICOLLINEARITY AND CORRELATION MATRIX

The correlation matrix expresses the relationship of two predictors. This matrix involves a row and columns table in which each cell represents the correlation coefficient of the two variables. The correlation coefficient ranges from -1 to 1 with -1 indicating perfect negative relationship and 1 perfect positive relationship. The visualization of the correlation matrix is an important step before the building of the machine learning model because you can gain a better understanding about what’s most important for your model.

**The Variance Inflation Factor** quantifies how much the variance of a specific regression coefficient for a feature is inflated due to multicollinearity with other features.

So by running the `data_new.corr()` function we can observe that each cell shows a value between 0 and 1. We put a threshold to 0.8 value to find the problematic pairs which make the model less accurate.


`problematic_pairs = correlation_matrix[(correlation_matrix.abs()> threshold) & (correlation_matrix.abs()<1)]`

`print(problematic_pairs)`

So the value such as the pair between active enteprises and births which is 0.99373 implies a very strong relationship between them that we have to act because it suggests that they contain similar (redundant) information.

This observation is important to identify later problems that occurs to our coefficients of the model, thus making it less accurate.

### Why Multicollinearity is a Problem

High multicollinearity leads to:</br>
i) Unstable coefficient estimates</br>
ii) Large standard errors</br>
iii) Difficulty in interpreting the effect of individual predictors</br>
In this dataset:

**active_enterprises**, **births**, and **deaths** are highly **correlated**
→ they represent similar underlying economic activity

Thus, removing births improves model stability.

## Evaluation of the model

A factor which implies how accurate is our model is R_2. This is the coefficient of determination a measure of the goodness of fit of a model. A value of 1 indicates that the regression predictions perfectly fit the data. The formula of r is calculated as 

                                                            
                                  R2= SSTotal​SSExplained​​ = 1−SSTotal​SSResidual​​

Formula definitions:

SSExplained​/SSTotal​ is the proportion of the total variance in y that can be "explained" by X2,X3,…,XpX2​,X3​,…,Xp​ (predictors)
R2R2 ranges from 0 to 1:

    If R2=0R2=0: None of the variance in X1X1​ is explained by X2,X3,…,XpX2​,X3​,…,Xp​; all the variance is left in SSResidualSSResidual​.
    If R2=1R2=1: All of the variance in X1X1​ is explained by X2,X3,…,XpX2​,X3​,…,Xp​, meaning SSResidual=0SSResidual​=0, and X1X1​ can be perfectly predicted from the other predictors.

 ## Model Diagnostics

After fitting the model, diagnostic checks were performed:

1. Residual Analysis

Standardized residuals were used to detect outliers

Observations with |residual| > 2 were considered problematic

2. Leverage

Measures how far an observation is from the mean of predictors

High leverage points can disproportionately affect the model

Threshold used:

leverage threshold=2pn
leverage threshold=
n
2p
	​

3. Cook’s Distance

Combines leverage and residual information

Identifies influential observations
