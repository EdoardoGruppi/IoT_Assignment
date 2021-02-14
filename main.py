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
# todo remove features for multicollinearity
plot_correlation_matrix(dataframe)
plot_distribution(dataframe, x='Total Energy', hue='Month')


# DATA INFERENCE =======================================================================================================

