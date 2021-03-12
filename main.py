# Import packages
from pandas import read_csv
import os
from Modules.utilities import *
from Modules.visualization import *
from Modules.data_preparation import *
from Modules.acquisition import download_dataset
from Models.model import *
from Modules.ML_interpretability import *

# DATA ACQUISITION =====================================================================================================
# The dataset can be downloaded from the following link https://bit.ly/37pTa0f or using the download_dataset()
# function provided that gets data from the IBM SQL Db2 dataset.
dataframe = download_dataset()
dataframe.to_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')

# DATA PREPROCESSING ===================================================================================================
# Read the csv file saved earlier
dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
# Rename all the columns. Some features are combined and some variables are dropped
dataframe = process_dataframe(dataframe)
# Resample the dataframe computing the mean of the interval selected. This helps a lot in the processing of the data.
dataframe = dataframe.resample(resampling).mean()
# Print meaningful information about the dataset such as the feature types or their statistics
get_info(dataframe)
# Enlarge the dataframe creating from the time index time-based columns ('Month', 'Year', 'Week Day' and so on)
dataframe = get_time_details(dataframe)
# Divide the dataset between the train, valid and test sets. Valid is 0 if cross validation is used to train models
train, valid, test = dataset_division(dataframe, valid_size=0, test_size=0.03)
# Copy the train dataframe on which to work for data visualization
dataframe = train.copy()

# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================
# Plot the correlation matrix of the dataframe
plot_correlation_matrix(dataframe)
# Plot a pie chart representing the consumption of every room of the house
pie_chart(dataframe, ['Kitchen', 'Living Room', 'Furnace', 'Outside', 'Home Office'], title='Rooms Consumption')
# PLot a distribution of the total monthly power consumption
plot_distribution(dataframe, x='Total Power', hue='Month')
# Detect univariate outlier in the time series passed
_ = detect_univariate_outlier(dataframe.iloc[:, :2], cap=None, nan=None)
# Plot the scatter plot matrix of some of the columns of the dataframe. Not all the columns are considered to
# facilitate the reading of the plot.
scatter_plot_matrix(dataframe.iloc[:, :5])
# Display a single plot that comprises n subplots, each displaying a specific series distribution
plot_distributions(dataframe.iloc[:, :13], columns=5)
# Show a comparison of the box plots of all the variables
box_plot(dataframe, dataframe.columns[:13], figsize=(30, 6))
# The following function plots one or more scatter plot matrices colouring the data points according to the specific
# time they are measured.
hue_scatter_plot_matrix(dataframe, dataframe.columns[:4], ['Month', 'Hour'])
# Display a seasonal plot to detect recurrent patterns
detect_seasonality(dataframe, target, 'Hour', 'Week Day', figsize=(12, 7))

# Consider only target variable of the project and plot it.
target_column = dataframe[target]
plot_series(target_column)
# Display the distribution of the target series in different ways and according to the time intervals considered
plot_distribution(dataframe, target)
plot_distribution(dataframe, x=target, hue='Hour')
# Plot the auto-correlation functions of the target column
plot_auto_correlation(target_column, lags=60)
# Decompose series to separate the trend, seasonality and residuals
decompose_series(target_column, period=100, mode='additive')

# Use the granger test to find some sort of meaningful relationships between the variables
granger_test(dataframe.iloc[:, :17], target_column=target, max_lag=8, test='ssr_chi2test')
# Check if the series is stationary
check_stationarity(dataframe)
check_single_stationarity(target_column)

# Remove the columns that are not useful
columns_to_remove = ['Day Of Year', 'Week Of Year', 'Kitchen', 'Living Room', 'Furnace', 'Outside', 'Home Office']
train = train.drop(columns_to_remove, axis=1)
valid = valid.drop(columns_to_remove, axis=1)
test = test.drop(columns_to_remove, axis=1)
# Since the optimal parameters are found using Cross validation the validation set is extracted dynamically from the
# training set. In this case the data are prepared using the transform_dataset_cv function.
train, train_target, test, test_target = transform_dataset_cv(train=train, test=test, target_column=target)

# DATA INFERENCE AND ML INTERPRETABILITY ===============================================================================
cv = 5
model_XGB = xgb_regressor(train, train_target, test, test_target, cv=cv, plot=False)
model_RF = random_forest_regressor(train, train_target, test, test_target, cv=cv, plot=False)
model_LGBM = light_gbm_regressor(train, train_target, test, test_target, cv=cv, plot=True)

# Select the best model
model = model_LGBM
# Compute the feature importance
features_importance(model.feature_importances_, train.columns)
# Plot the partial dependence plots for one variable...
plot_partial_dependencies(model, test, column=['Dew Point'])
plot_partial_dependencies(model, test, column=['Pressure'])
plot_partial_dependencies(model, test, column=['Temperature'])
plot_partial_dependencies(model, test, column=['Wind Bearing'])
plot_partial_dependencies(model, test, column=['Wind Speed'])
plot_partial_dependencies(model, test, column=['Humidity'])
plot_partial_dependencies(model, test, column=['Precipitation'])
plot_partial_dependencies(model, test, column=['Hour'])
# ...and for two variables.
plot_two_ways_pdp(model, test, [('Pressure', 'Humidity')])
plot_two_ways_pdp(model, test, [('Temperature', 'Humidity')])
plot_two_ways_pdp(model, test, [('Temperature', 'Hour')])
plot_two_ways_pdp(model, test, [('Pressure', 'Dew Point')])
plot_two_ways_pdp(model, test, [('Temperature', 'Wind Speed')])
plot_two_ways_pdp(model, test, [('Pressure', 'Hour')])
plot_two_ways_pdp(model, test, [('Pressure', 'Temperature')])
plot_two_ways_pdp(model, test, [('Pressure', 'Precipitation')])
plot_two_ways_pdp(model, test, [('Day Moment', 'Temperature')])
# Ice plots
plot_ice(model, test, features=['Temperature'])
plot_ice(model, test, features=['Dew Point'])
# Surrogate Tree
surrogate_tree(model, test, max_depth=5)
# Shap plots
plot_shap(model, test, instance=96, dataset=True)
# Lime plot
plot_lime(model, test, instance=96)
# Model residuals
model_residuals(model, train, train_target)
