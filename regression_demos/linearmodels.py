# Dependencies
#  The classes defined below rely on the following libraries
import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Defines the linear models class. A linear model have a dependent
#  variable y and independent variables contained in an array of x's.
#  From the y and x-inputs, the linear model can be fitted with parameters
#  to find the relation that best describes the data.
#


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


# A Linear Regression class that creates a Linear Regression instance.
#  The linear regression fits parameters by minimizing the model's
#  deviance.
#
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
        if self._intercept == True:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

        XTX_inv = np.linalg.inv(x.T @ x)
        self._betas = np.dot(XTX_inv, (x.T @ y))

        if self.has_intercept == True:
            self._intercept = self._betas[0]
            self._betas = self._betas[1:]

    def fit_transform(self, x, y):
        """Fitting and returning the solved linear equation."""
        self.fit(x, y)
        return x @ self._betas

    def transform(self, x):
        """Returns the trained betas on new observations."""
        if self.has_intercept == True:
            return x @ np.concatenate((np.array(self._intercept, ndmin=1), np.array(self._betas)))
        return x @ self._betas

# Defines the LogisticRegression class


class LogisticRegression(LM):
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

    def fit(self, x, y):
        """Fitting the weights"""
        if self._has_constant:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        self._weights = np.zeros((x.shape[1], 1))
        m = minimize(self.cost, self._weights, args=(x, y))
        self._weights = m.x

    # Predicting the values
    def predict(self, x):
        """Predicting the values"""
        if self._has_constant:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        y_pred = 1 / (1 + np.exp(-np.dot(x, self._weights)))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

# Defines the DataSet class. A dataset has a dependent variable y and
#  covariates x. The DataSet can be split in a training and testing
#  component for model fitting purposes.
#


class DataSet:
    # Constructs a dataset object from an x and y input.
    #  @param x the covariates of the dataset
    #  @param y the dependent variable of the dataset
    #  @param transposed False if the rows correspond to observations and
    #  columns correspond to variables.
    #  @param scaled True if you want to scale x-values to values in the
    #  interval 0-1.
    #
    def __init__(self, x, y, scaled=False, transposed=False):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Data must be numpy arrays")
        if transposed:
            self._x = x
        else:
            self._x = x.T
        if scaled == False:
            scaler = MinMaxScaler()
            temp = scaler.fit_transform(self._x.T)
            self._x = temp.T
        self._y = y.reshape(1, -1)

    # Separates the input data in a testing and training set for
    #  model fitting purposes. Training and test sets are stored as
    #  instance variables in the object.
    #  @param testSize the fraction of observations that should be in
    #  the test set
    #  @param randomState a random seed for result reproducibility
    #
    def train_test(self, testSize, randomState):
        """Splitting into train and test sets. The function stores
        the values inside the class."""
        random.seed(randomState)
        # extract positions from random.sample()
        num_obs = self._x.shape[1]
        # positions to extract for training set
        self._trainSet = random.sample(
            range(num_obs), int(num_obs * (1-testSize)))
        # remaining positions for test set
        self._testSet = [i for i in range(num_obs) if not i in self._trainSet]

    @property
    # Returns the covariates of the input data
    #  @return covariates in transposed form
    #
    def x(self):
        return self._x

    @property
    # Returns the dependent variable of the input data
    #  @return independent variables
    #
    def y(self):
        return self._y

    @property
    # Returns the training set portion of the input covariate data
    #  @return training set x
    #
    def x_tr(self):
        return self._x[:, self._trainSet]

    @property
    # Returns the training set portion of the input dependent variable
    #  @return training set y
    #
    def y_tr(self):
        return self._y[:, self._trainSet]

    @property
    # Returns the testing set portion of the input covariate data
    #  @return testing set x
    #
    def x_te(self):
        return self._x[:, self._testSet]

    @property
    # Returns the testing set portion of the input dependent variable
    #  @return testing set y
    #
    def y_te(self):
        return self._y[:, self._testSet]

# Defines the csvDataSet class, a subclass of the DataSet class. The
#  csvDataSet y and x values from a csv file and constructs a DataSet
#  object.
#


class csvDataSet(DataSet):
    # Constructs a DataSet object from a csv file. The dependent variable
    #  values must be in the first row (if transposed) or the first column
    #  otherwise.
    #  @param filename the csv file to read
    #  @param transposed False if the rows correspond to observations and
    #  columns correspond to variables.
    #  @param scaled True if you want to scale values
    #
    def __init__(self, filename, transposed=False, scaled=False):
        from csv import reader
        with open(filename, "r", newline="") as infile:
            csvReader = reader(infile)
            row1 = next(csvReader)
            data = np.array(row1).astype(float)
            for row in csvReader:
                row = np.array(row).astype(float)
                data = np.vstack((data, row))
        if transposed:
            y = data[0, :]
            x = data[1:, :]
        else:
            y = data[:, 0]
            x = data[:, 1:]
        super().__init__(x, y, scaled, transposed)


class diagnosticPlot():
    # The constructor is instantiated if the input is either a logistic
    # Regression, or a linear regression
    def __init__(self, reg_model):
        if not isinstance(reg_model, (LogisticRegression, LinearRegression)):
            raise TypeError(
                'The argument is not a Logistic Regression or Linear Regression instance.')
        self._mod_type = reg_model.model_type

    def plot(self, y, mu):
        # Checking to see if the model type is Logistic Regression.
        if self._mod_type == 'Logistic Regression':
            from sklearn import metrics
            fpr, tpr, tresholds = metrics.roc_curve(y, mu)
            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                              estimator_name='ROC curve')
            display.plot()
            plt.show()
        # Checking to see if the model type is Linear Regression.
        elif self._mod_type == 'Linear Regression':
            # Size of the graph
            fig, ax = plt.subplots(figsize=(8, 8))

            # Scatter plot of the real values vs mu
            ax.scatter(y, mu, color='#426ded', s=20)

            # Adjusting the axes.
            # Finding the max values for equal scaling, and adding a bit of
            # extra space to not make the last values not too close to the limit.
            max_val = int(np.ceil(np.max([np.max(mu), np.max(y)])) * 1.05)

            # testing to see if the min value is below 0.
            min_val = int(np.ceil(np.min([np.min(mu), np.min(y)])) * 1.05)

            # If value is lower than 20, the scale will be adjusted to
            # a little below the lowest value.
            if min_val < 20:
                lowest_scale = min_val - 10
            else:
                lowest_scale = 0

            # X-axis
            ax.set_xlabel('Y Values')
            plt.xlim([lowest_scale, max_val])

            # Y-axis title
            ax.set_ylabel('Mu Values')
            plt.ylim([lowest_scale, max_val])
            # Set grid and background color
            ax.set_facecolor('#e9ecf5')
            ax.grid(color='grey', axis='y', linestyle='solid', linewidth=.15)

            # Changing the borders to borders for design purposes.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)

            # The title.
            plt.title('Y vs µ', size=18, pad=20)

            plt.show()
        else:
            # Error raised if the model type conditions are not met.
            raise NameError("The model has to be either a 'Logistic"
                            "Regression', or a 'Linear Regression' model")
