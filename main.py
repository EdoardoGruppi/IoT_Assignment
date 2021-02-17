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
dataframe = dataframe.resample('5min').interpolate()
# get_info(dataframe)
dataframe = get_time_details(dataframe)
train, valid, test = dataset_division(dataframe, valid_size=0.05, test_size=0.05)
dataframe = train.copy()

# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================
# plot_correlation_matrix(dataframe)
# plot_distribution(dataframe, x='Total Energy', hue='Month')
# dataframe = detect_univariate_outlier(dataframe.iloc[:, :2], cap='z_score', nan=None)
# scatter_plot_matrix(dataframe.iloc[:, :5])
# plot_distributions(dataframe.iloc[:, :20], columns=5)
# box_plot(dataframe, dataframe.columns[:5])
# hue_scatter_plot_matrix(dataframe, dataframe.columns[:2], ['Month'])
# detect_seasonality(dataframe, dataframe.columns[0], 'Hour', 'Week Day')

# target_column = dataframe['Total Energy']
# plot_series(target_column)
# plot_distribution(dataframe, 'Total Energy')
# plot_distribution(dataframe, x='Total Energy', hue='Hour')
# plot_auto_correlation(target_column, lags=60)
# decompose_series(target_column, period=100, mode='additive')
# granger_test(dataframe.iloc[:, :8], target_column='Total Energy', max_lag=4)
# check_stationarity(target_column)

# todo hypothesis testing

# columns_to_remove = []
# train = train.drop(columns_to_remove, axis=1)
# valid = valid.drop(columns_to_remove, axis=1)
# test = test.drop(columns_to_remove, axis=1)
train, train_target, valid, valid_target, test, test_target = transform_dataset(train=train, valid=valid, test=test,
                                                                                target_column='Total Energy',
                                                                                reduction=True, n_components=0.93)

# DATA INFERENCE AND ML INTERPRETABILITY ===============================================================================
# Find the best value for the c parameter of a SVM
c_value = find_svm(train, train_target, valid, valid_target, max_c=100)
model_SVR = support_vector_machine(train=train, train_target=train_target, test=test,
                                   test_target=test_target, c=c_value)
features_importance(model_SVR.coef_, train.columns)
plot_partial_dependencies(model_SVR, test, column='0')
plot_two_ways_pdp(model_SVR, test, [('0', '1')])
plot_ice(model_SVR, test, column='0')

