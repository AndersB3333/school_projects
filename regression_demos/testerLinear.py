# Testing
# 4.2. testerLinear.py

# Dependencies
import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import read_csv

from linearmodels import *

real_estate = csvDataSet("real_estate.csv", scaled=True)

# Demonstration
# Uploading the dataset, and declaring independent and dependent values
df = read_csv('real_estate.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Instantiating an OLS model, and fitting the model
lin_reg = LinearRegression(has_intercept=True)
lin_reg.fit(X_train, y_train)

# transforming the linear regression models
X_transformed = lin_reg.transform(X_test)

print(
    f"Explained sum of squares is: {np.sum((X_transformed - np.mean(y_test)) ** 2)}")


# Linear Regression Model 4
reg4 = LinearRegression()
reg4.linearModel(real_estate, "full", "y ~ b1*x1")
reg4.optimize()
reg4.summary()

# Creating another instance of Linear Regression model 3 on the dataset.
reg4_model_plot = diagnosticPlot(reg4)

# Plotting the Linear Regression with y-values vs mu.
reg4_model_plot.plot(real_estate.y, reg4.predict(real_estate.x))
