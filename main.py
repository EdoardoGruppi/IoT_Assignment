# Import packages
from pandas import read_csv
import os
from Modules.config import *
from Modules.utilities import *
from Modules.visualization import *
from Modules.data_preparation import *

# DATA ACQUISITION =====================================================================================================


# DATA PREPROCESSING ===================================================================================================
dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
dataframe = remove_features(dataframe)
# get_info(dataframe)
train, valid, test = dataset_division(dataframe, valid_size=0.05, test_size=0.05)
dataframe = train.copy()
dataframe = get_time_details(dataframe)

# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================
# plot_correlation_matrix(dataframe)
# plot_distribution(dataframe, x='Total Energy', hue='Month')
# dataframe = detect_univariate_outlier(dataframe.iloc[:, :2], cap='z_score', nan=None)
# scatter_plot_matrix(dataframe.iloc[:, :5])
# plot_distributions(dataframe.iloc[:, :20], columns=5)
# box_plot(dataframe, dataframe.columns[:5])
# hue_scatter_plot_matrix(dataframe, dataframe.columns[:2], ['Month'])
# detect_seasonality(dataframe, dataframe.columns[0], 'Hour', 'Week Day')

target_column = dataframe['Total Energy']
# plot_series(dataframe[target_column])
# plot_auto_correlation(dataframe[target_column], lags=60)
# decompose_series(target_column, period=100, mode='additive')
# granger_test(dataframe.iloc[:, :8], target_column='Total Energy', max_lag=4)
# check_stationarity(target_column)

# todo hypothesis testing
# columns_to_remove = []
# train = train.drop(columns_to_remove, axis=1)
# valid = valid.drop(columns_to_remove, axis=1)
# test = test.drop(columns_to_remove, axis=1)
# train, valid, test = transform_dataset(train, valid, test, 'Total Energy', reduction=False)


# DATA INFERENCE =======================================================================================================
