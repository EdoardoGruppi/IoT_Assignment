# Import packages
from pandas import read_csv
import os
from Modules.config import *
from Modules.utilities import *

# DATA ACQUISITION =====================================================================================================


# DATA PREPROCESSING ===================================================================================================
dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
dataframe = remove_features(dataframe)
get_info(dataframe)
print(dataframe.head())
print(dataframe.tail())
# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================


# DATA INFERENCE =======================================================================================================

