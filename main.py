# Import packages
from pandas import read_csv
import os
from Modules.config import *
from Modules.utilities import *
from Modules.visualization import *

# DATA ACQUISITION =====================================================================================================


# DATA PREPROCESSING ===================================================================================================
dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
dataframe = remove_features(dataframe)
# get_info(dataframe)
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
# todo Remove features for multicollinearity

# PREPARE DATASET ======================================================================================================


# DATA INFERENCE =======================================================================================================
