import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from linearmodels import *


class LM:
    # Specify the model, choose DataSet object to get data from,
    #  and whether you want to regress on the entire set of data or a
    #  part of it.
    #  @param dataobj the DataSet object to get data from
    #  @param part specify which part of the DataSet you want to
    #  perform regression on. Takes inputs ("full" and "train").
    #  @param regression specifies the regression in the form
    #  "y ~ b0 + b1*x1 + b2*x3" if you want a regression with a constant,
    #  the first variable and the third variable.
    #
    def linearModel(self, dataobj, part, regression):
        # The instance variable _constantAdded affect the print method and
        # the array that chooses variables from the x-array.
        self._constantAdded = False
        # Check if first row in x-array is constant and non-zero
        firstRow = dataobj.x[0, :].reshape(1, dataobj.x.shape[1])
        # Does first row have 0 diff between max and min value?
        hasConstantRow = np.ptp(firstRow, axis=1) == 0
        # Is the first row non-zero?
        hasConstantRow &= np.all(firstRow != 0.0, axis=1)
        if hasConstantRow:
            self._constantAdded = True

        # Specify instance variable for use in other methods
        self._model = regression
        # Find the tilde position in the regression string
        tildePos = self._model.find("~")
        # From the covariates part of the string (after the ~), split by "+"
        # into a list
        covariates = self._model[tildePos+1:].split(sep="+")
        # Empty list for x-variables
        self._variables = []
        # Iterate over each element in covariates list
        for addend in covariates:
            addend = addend.strip()
            # start search from right-hand side
            i = len(addend) - 1
            while addend[i].isdigit():
                i -= 1
            # take out the integer part of covariate
            variable = int(addend[i+1:])
            if variable == 0:
                dataobj.add_constant()
                self._constantAdded = True
                self._variables.append(variable)
            elif variable > dataobj.x.shape[0]:
                raise Exception("Covariate out of range")
            else:
                self._variables.append(variable)
        # _variables is used to know the variable numbers, _regrPos
        # is an array that specifies the variable's position in the
        # x-array.
        self._regrPos = self._variables
        if not self._constantAdded:
            self._regrPos = [i-1 for i in self._variables]

        # Instance variable to store beta estimates
        self._params = np.zeros(len(self._variables))

        # Determines which observations of the DataSet to use depending
        # on user input.
        if part == "full":
            self._x = dataobj.x
            self._y = dataobj.y
        elif part == "train":
            self._x = dataobj.x_tr
            self._y = dataobj.y_tr
        else:
            raise Exception("'part' must be specified ('full' or 'train')")

    # The model's deviance given parameters. Should be overridden in
    #  subclasses
    #  @param params the beta parameters to test for
    #  @param x the x-array
    #  @param y the y observations
    #

    def fit(self, params, x, y):
        raise NotImplementedError

    # The model's estimate for mu. Should be overridden
    #  @param x the x variables to make an estimation for. e.g., x_tr, x_te,
    #  or x from the DataSet object
    #
    def predict(self, x):
        raise NotImplementedError

    @property
    # Get fitted parameter beta
    #  @return the fitted parameters in an array
    #
    def params(self):
        return self._params

    # Numerical minimization with scipy
    #  @param init_val initial guess of parameter values
    #
    def optimize(self, init_val=1):
        x = self._x[self._regrPos, :]
        y = self._y
        len_params = x.shape[0]
        init_params = np.repeat(init_val, len_params)
        results = minimize(self.fit, init_params, args=(x, y))
        self._params = results['x']

    @property
    # Get the string representation of the specified model
    #
    def model(self):
        try:
            return self._model
        except:
            print("No model is specified")

    # Calculates the appropriate statistic to show model performance
    #  Should be overridden
    def diagnosis(self):
        raise NotImplementedError

    # Print out a string result for specified model if it's specified.
    #  If the model is not yet fitted, the method returns 0 values for
    #  parameters.
    #
    def __repr__(self):
        try:
            temp = "y ~"
            for i in range(len(self._variables)):
                if self._constantAdded and i == 0:
                    temp = temp + " {}".format(self.params[0])
                    continue
                sign = " + "
                if not self._constantAdded and i == 0:
                    sign = " "
                if self._params[i] < 0:
                    sign = " - "
                temp = temp + sign+"{}*x{}".format(abs(self._params[i]),
                                                   self._variables[i])
            print(temp)
        except NameError:
            print("I am a Linear Model")

    # Prints out the model specified in linearModel, the fitted
    #  parameters, and the model accuracy in the following format:
    # ----------------------------------------
    # y ~ b0 + b1*x1
    # R-squared:      0.724
    # b0:            -7.233
    # b1:             2.080
    # ----------------------------------------
    #

    def summary(self):
        diagLabel = "R-squared:"
        if self.model_type == "Logistic Regression":
            diagLabel = "AUC:"
        rsq = self.diagnosis()
        temp = f"{'-'*40}\n{self._model:40}\n{diagLabel:11}"\
            f"{rsq:10.3f}{'':19}\n"
        for i in range(len(self._variables)):
            temp += f"b{str(self._variables[i]) + ':':10s}"\
                    f"{self._params[i]:10.3f}{'':19}\n"
        temp += "-"*40
        print(temp)


class LinearRegression(LM):
    def __init__(self, has_intercept=False):
        """Instantiates an empty OLS linear regression model."""
        self._betas = None
        self._intercept = None
        self.has_intercept = has_intercept

    def y_intercept(self):
        """Implements a constant in the regression model."""
        self.has_intercept = True

    def fit(self, x, y):
        """Storing the fitted betas. If has_intercept instance variable is
        set to true, intercept beta will be stored elsewhere"""
        if self.has_intercept == True:
            x.shape
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

        XTX_inv = np.linalg.inv(x.T @ x)
        self._betas = np.dot(XTX_inv, (x.T @ y))

        if self.has_intercept == True:
            self._intercept = self._betas[0]
            self._betas = self._betas[1:]

    def fit_transform(self, x, y):
        """Fitting and returning the solved linear equation."""
        self.fit(x, y)
        if self.has_intercept == True:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            return x @ np.concatenate((np.array(lin_reg._intercept, ndmin=1), np.array(self._betas)))
        return x @ self._betas

    def transform(self, x):
        """Returns the trained betas on new observations."""
        if self.has_intercept == True:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            return x @ np.concatenate((np.array(lin_reg._intercept, ndmin=1), np.array(self._betas, ndmin=1)))
        return x @ self._betas


# Demonstration
# Uploading the dataset, and declaring independent and dependent values
df = pd.read_csv('real_estate.csv', header=None)
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
