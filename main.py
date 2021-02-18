# Import packages
from pandas import read_csv
import os
from Modules.config import *
from Modules.utilities import *
from Modules.visualization import *
from Modules.data_preparation import *
from Modules.acquisition import download_dataset
from Models.model import *
from Modules.ML_interpretability import *

# DATA ACQUISITION =====================================================================================================
# The dataset can be downloaded from the following link https://bit.ly/37pTa0f or using the download_dataset()
# function provided that gets data from the IBM SQL Db2 dataset.
# dataframe = download_dataset()
# dataframe.to_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')

# DATA PREPROCESSING ===================================================================================================
dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
dataframe = process_dataframe(dataframe)
dataframe = transform_categorical(dataframe, 'Light')
dataframe = dataframe.resample(resampling).interpolate()
# get_info(dataframe)
dataframe = get_time_details(dataframe)
train, valid, test = dataset_division(dataframe, valid_size=0, test_size=0.05)
dataframe = train.copy()

# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================
# plot_correlation_matrix(dataframe)
# plot_distribution(dataframe, x='Total Energy', hue='Month')
# _ = detect_univariate_outlier(dataframe.iloc[:, :2], cap=None, nan=None)
# scatter_plot_matrix(dataframe.iloc[:, :5])
# plot_distributions(dataframe.iloc[:, :14], columns=5)
# box_plot(dataframe, dataframe.columns[:14])
# hue_scatter_plot_matrix(dataframe, dataframe.columns[:4], ['Month', 'Hour'])
# detect_seasonality(dataframe, target, 'Hour', 'Week Day')

# target_column = dataframe[target]
# plot_series(target_column)
# plot_distribution(dataframe, target)
# plot_distribution(dataframe, x=target, hue='Hour')
# plot_auto_correlation(target_column, lags=60)
# decompose_series(target_column, period=100, mode='additive')

# granger_test(dataframe, target_column=target, max_lag=4)
# check_stationarity(target_column)

columns_to_remove = ['Day Of Year', 'Week Of Year', 'Dew Point']
train = train.drop(columns_to_remove, axis=1)
valid = valid.drop(columns_to_remove, axis=1)
test = test.drop(columns_to_remove, axis=1)
# Since the optimal parameters are found using Cross validation the validation set is extracted dynamically from the
# training set. In this case the data are prepared using the transform_dataset_cv function.
train, train_target, test, test_target = transform_dataset_cv(train=train, test=test, target_column=target,
                                                              reduction=False, n_components=0.93)

# DATA INFERENCE AND ML INTERPRETABILITY ===============================================================================
# Find the best value for the c parameter of a SVM
# model_SVR = support_vector_machine(train=train, train_target=train_target, test=test, test_target=test_target, cv=5)
# features_importance(model_SVR.coef_, train.columns)
# plot_partial_dependencies(model_SVR, test, column='0')
# plot_two_ways_pdp(model_SVR, test, [('0', '1')])
# plot_ice(model_SVR, test, column='0')
# surrogate_tree(model_SVR, test, max_depth=4)
# plot_lime(model_SVR, test, instance=25)

model_XGB = xgb_regressor(train, train_target, test, test_target, cv=5)
plot_shap(model_XGB, test, instance=25, feature='0', dataset=True)
model_XGB_RF = xgb_random_forest_regressor(train, train_target, test, test_target, cv=5)
plot_shap(model_XGB_RF, test, instance=25, feature='0', dataset=True)
