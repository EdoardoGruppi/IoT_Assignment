# Import packages
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Modules.visualization import plot_results
from xgboost import XGBRegressor, XGBRFRegressor
from Modules.utilities import compute_metrics


def support_vector_machine(train, train_target, test, test_target, plot=True, cv=5):
    """
    Trains a support vector machine using cross validation to find the optimal parameters. Than the model is used to
    predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param plot: if True the results are plotted as well. default_Value=True
    :param cv: determines the number of folds used in the cross-validation splitting strategy. default_Value=5
    :return: the features importance.
    """
    # List of parameters to evaluate
    params = {'C': [1, 5, 10, 50, 100]}
    # Instance of the model
    model_SVR = LinearSVR(max_iter=10000, loss='squared_epsilon_insensitive', dual=False)
    grids = GridSearchCV(model_SVR, params, cv=cv, verbose=3, n_jobs=-1)
    # Search the best values
    grids.fit(train, train_target)
    print(f'Best estimator: {grids.best_estimator_}')
    # Train the best estimator
    model_SVR = grids.best_estimator_
    model_SVR.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_SVR.predict(test)
    # Compute and print metrics
    compute_metrics(test_target, test_predictions, model='SVM')
    if plot:
        plot_results(test_predictions, test_target)
    return model_SVR


def xgb_regressor(train, train_target, test, test_target, plot=True, cv=5):
    """
    Trains a XGBRegressor model to predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param plot: if True the results are plotted as well. default_Value=True
    :param cv: determines the number of folds used in the cross-validation splitting strategy. default_Value=5
    :return: the features importance.
    """
    # List of parameters to evaluate
    params = {'n_estimators': [1, 3, 5], 'max_depth': [4, 8, 12], 'learning_rate': [0.01, 0.1, 1]}
    # Instantiate a XGBRegressor object
    model_XGB = XGBRegressor()
    grids = GridSearchCV(model_XGB, params, cv=cv, verbose=3, n_jobs=-1)
    # Fit the model according to the given training data
    grids.fit(train, train_target)
    print(f'Best estimator: {grids.best_estimator_}')
    # Train the best estimator
    model_XGB = grids.best_estimator_
    model_XGB.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_XGB.predict(test)
    # Compute and print metrics
    compute_metrics(test_target, test_predictions, model='XGB')
    if plot:
        plot_results(test_predictions, test_target)
    return model_XGB


def xgb_random_forest_regressor(train, train_target, test, test_target, plot=True, cv=5):
    """
    Trains a XGB Random Forest Regressor model to predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param plot: if True the results are plotted as well. default_Value=True
    :param cv: determines the number of folds used in the cross-validation splitting strategy. default_Value=5
    :return: the features importance.
    """
    # List of parameters to evaluate
    params = {'n_estimators': [1, 3, 5], 'max_depth': [4, 8, 12], 'learning_rate': [0.01, 0.1, 1]}
    # Instantiate a XGBRFRegressor object
    model_XGB_RF = XGBRFRegressor()
    grids = GridSearchCV(model_XGB_RF, params, cv=cv, verbose=3, n_jobs=-1, refit=True)
    # Fit the model according to the given training data
    grids.fit(train, train_target)
    print(f'Best estimator: {grids.best_estimator_}')
    # Train the best estimator
    model_XGB_RF = grids.best_estimator_
    model_XGB_RF.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_XGB_RF.predict(test)
    # Compute and print metrics
    compute_metrics(test_target, test_predictions, model='XGB_RF')
    if plot:
        plot_results(test_predictions, test_target)
    return model_XGB_RF
