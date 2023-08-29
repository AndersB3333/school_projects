import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression


class LogisticRegression():
    """A logistic regression class that uses sigmoid function to calculate the weights.
    Does not include a constant."""
    # The constructor contains one empty variable which is the weights
    model_type = "Logistic Regression"

    def __init__(self, has_constant=False):
        self._weights = None
        self._has_constant = has_constant

    def cost(self, weights, x, y):
        """Determining the cost function of the model (using sigmoid function)"""
        y_pred = 1 / (1 + np.exp(-np.dot(x, weights)))
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # Fitting the weights
    def fit(self, x, y):
        if self._has_constant:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        self._weights = np.zeros((x.shape[1], 1))
        m = minimize(self.cost, self._weights, args=(x, y))
        self._weights = m.x

    # Predicting the values
    def predict(self, x):
        if self._has_constant:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        y_pred = 1 / (1 + np.exp(-np.dot(x, self._weights)))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred


# Demo
# Loading a classification dataset
spector_data = sm.datasets.spector.load()
spector_y = spector_data.endog.values
spector_x = spector_data.exog.values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(spector_x, spector_y)

# Instantiating a logistic regression object
log_reg = LogisticRegression(has_constant=True)
# Fitting the weights of the model
log_reg.fit(X_train, y_train)
# Predicting the target variables based on new data
y_pred = log_reg.predict(X_test)
# Printing the confusion matrix
print(confusion_matrix(y_test, y_pred))
