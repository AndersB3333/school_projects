# Testing
# 4.1. testerLogistic.py
# Dependencies

import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from linearmodels import *

import statsmodels.api as sm
spector_data = sm.datasets.spector.load()
# if pandas dataset in statsmodels (some versions of statsmodels):
spector_y = spector_data.endog.values
spector_x = spector_data.exog.values
# if numpy arrays in statsmodels (some versions of statsmodels)
#spector_y = spector_data.endog
#spector_x = spector_data.exog

spector = DataSet(spector_x, spector_y)

# Splitting the dataset.
spector.train_test(testSize=0.3, randomState=12345)

# Logistic Regression Model 1
# Creating an instance of the first regression model.
reg1 = LogisticRegression()
# Specifying dataobject, covariates and the training set to use in the
# model.
reg1.linearModel(spector, 'train', "y ~ b0 + b1*x1")

# Fitting the beta parameter's.
reg1.optimize()
reg1.summary()

# Creating an instance of diagnosticPlot with Linear Regression
# as input.
reg1_model_plot = diagnosticPlot(reg1)

yTe = spector.y_te.reshape(-1, 1)
# Predicting the Logistic Regression 1 model on the test set.
x_te = spector.x_te
y_pred = reg1.predict(x_te)
# Plotting the ROC curve on the test set.
reg1_model_plot.plot(yTe, y_pred)

# Logistic Regression Model 2
reg2 = LogisticRegression()
reg2.linearModel(spector, "train", "y ~ b0 + b1*x1 + b2*x2")
reg2.optimize()
reg2.summary()

# Predicting the Logistic Regression 2 model on the test set.
y_pred = reg2.predict(x_te)
# Creating another instance of diagnosticplot.
reg2_model_plot = diagnosticPlot(reg2)
# Plotting the ROC curve on the test set.
reg2_model_plot.plot(yTe, y_pred)

# Logistic Regression Model 3
reg3 = LogisticRegression()
reg3.linearModel(spector, "train", "y ~ b0 + b1*x1 + b2*x2 + b3*x3")
reg3.optimize()
reg3.summary()

# Predicting the Logistic Regression 3 model on the test set.
y_pred = reg3.predict(x_te)
# Creating another instance of diagnosticplot.
reg3_model_plot = diagnosticPlot(reg3)
# Plotting ROC curve on the test set.
reg3_model_plot.plot(yTe, y_pred)
