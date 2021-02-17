# Import packages
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sn


def find_svm(train, train_target, valid, valid_target, max_c=100):
    """
    Finds the best value for the c parameter of the SVM.

    :param train: train dataset except the target column.
    :param train_target: target column of the train dataset.
    :param valid: valid dataset except the target column.
    :param valid_target: target column of the valid dataset.
    :param max_c: maximum value of c explored. default_value=100
    :return: the best value of c.
    """
    # List of the errors achieved
    error = []
    # Feed the model with training sets and train
    for counter in range(1, max_c + 1):
        print(f'Testing SVM with C = {counter}.')
        # Instantiate a Linear Support Vector Classifier
        # todo change
        model_SVR = LinearSVR(C=counter, max_iter=5000, loss='squared_epsilon_insensitive', dual=False)
        # Fit the model according to the given training data
        model_SVR.fit(train, train_target)
        # Predict target variable for samples in the valid dataset
        valid_predictions = model_SVR.predict(valid)
        # Compute and save the error achieved
        error.append(mean_squared_error(valid_target, valid_predictions))
    # Find the best value of c according to the errors achieved
    best = np.argmin(error) + 1
    print(f'The best value for the c parameter is {best}')
    return best


def support_vector_machine(train, train_target, test, test_target, c, plot=True):
    """
    Trains a support vector machine to predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param c: value of the c parameter of the SVM.
    :param plot: if True the results are plotted as well.
    :return: the features importance.
    """
    # Instantiate a Linear Support Vector Classifier
    # todo change support vector machine
    model_SVR = LinearSVR(C=c, max_iter=5000)
    # Fit the model according to the given training data
    model_SVR.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_SVR.predict(test)
    # Compute and print the R2_score
    r_squared = r2_score(test_target, test_predictions)
    print(f'R-squared on test set: {r_squared:.4f}')
    if plot:
        sn.set()
        plt.figure(figsize=(25, 10))
        plt.plot(test_target.index.values, test_target.values, label='Actual data')
        plt.plot(test_target.index.values, test_predictions, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Total Energy (kW)')
        plt.tight_layout()
        plt.show()
    return model_SVR
