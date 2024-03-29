# Import packages
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV
from Modules.visualization import plot_results
from xgboost import XGBRegressor, XGBRFRegressor
from Modules.utilities import compute_metrics
from lightgbm import LGBMRegressor
from Modules.utilities import residuals_properties
from sklearn.ensemble import RandomForestRegressor
import warnings


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
    params = {'C': [1, 50, 100], 'kernel': ['poly', 'rbf']}
    # Instance of the model
    model_SVR = SVR()
    grids = GridSearchCV(model_SVR, params, cv=cv, verbose=1, n_jobs=-1)
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
    params = {'n_estimators': [400, 500], 'max_depth': [2, 3, 5], 'learning_rate': [0.1]}
    # Instantiate a XGBRegressor object
    model_XGB = XGBRegressor()
    grids = GridSearchCV(model_XGB, params, cv=cv, verbose=1, n_jobs=-1)
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


def random_forest_regressor(train, train_target, test, test_target, plot=True, cv=5):
    """
    Trains a Random Forest Regressor model to predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param plot: if True the results are plotted as well. default_Value=True
    :param cv: determines the number of folds used in the cross-validation splitting strategy. default_Value=5
    :return: the features importance.
    """
    # List of parameters to evaluate
    params = {'n_estimators': [50]}
    # Instantiate a Random Forest object
    model_RF = RandomForestRegressor(oob_score=True, random_state=33)
    grids = GridSearchCV(model_RF, params, cv=cv, verbose=1, n_jobs=-1)
    # Fit the model according to the given training data
    grids.fit(train, train_target)
    print(f'Best estimator: {grids.best_estimator_}')
    # Train the best estimator
    model_RF = grids.best_estimator_
    model_RF.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_RF.predict(test)
    # Compute and print metrics
    compute_metrics(test_target, test_predictions, model='RF')
    if plot:
        plot_results(test_predictions, test_target)
    return model_RF


def light_gbm_regressor(train, train_target, test, test_target, plot=True, cv=5):
    """
    Trains a Light GBM Regressor model to predict a target variable.

    :param train: train dataset.
    :param train_target: target column related to the training dataset.
    :param test: test dataset.
    :param test_target: target column related to the test dataset.
    :param plot: if True the results are plotted as well. default_Value=True
    :param cv: determines the number of folds used in the cross-validation splitting strategy. default_Value=5
    :return: the features importance.
    """
    warnings.filterwarnings("ignore")
    # List of parameters to evaluate
    params = {'n_estimators': [400, 500], 'max_depth': [2, 5], 'learning_rate': [0.1],
              'min_child_samples': [10, 20], 'num_leaves': [50, 200]}
    # Instantiate a LGBMRegressor object
    model_LGBM = LGBMRegressor(class_weight='balanced')
    grids = GridSearchCV(model_LGBM, params, cv=cv, verbose=1, n_jobs=-1, scoring='r2')
    # Fit the model according to the given training data
    grids.fit(train, train_target)
    print(f'Best estimator: {grids.best_estimator_}')
    # Train the best estimator
    model_LGBM = grids.best_estimator_
    model_LGBM.fit(train, train_target)
    # Predict target variable for samples in the test dataset
    test_predictions = model_LGBM.predict(test)
    # Compute and print metrics
    compute_metrics(test_target, test_predictions, model='LGBM')
    if plot:
        plot_results(test_predictions, test_target)
    return model_LGBM


def model_residuals(model, data, data_target, model_name='Model'):
    """
    Computes and plots the residuals of the model obtained on the data passed.

    :param model: model object used to make predictions.
    :param data: data used to predict a target variable.
    :param data_target: true values of the target variable.
    :param model_name: string to identify the model. default_value='Model'
    :return:
    """
    # Make predictions
    data_predictions = model.predict(data)
    # Compute the residuals
    residuals = data_target - data_predictions
    # Display the residuals properties
    residuals_properties(residuals, model=model_name)
