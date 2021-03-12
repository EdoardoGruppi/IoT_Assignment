# Import packages
from pandas import to_datetime, date_range, factorize
from numpy import mean, abs, sqrt, corrcoef
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import qqplot


def get_info(dataframe):
    """
    Returns information about the dataframe passed.

    :param dataframe: dataframe to analyse.
    """
    print('\nDataframe info ' + '=' * 70)
    print(dataframe.info())
    print('\nDataframe head ' + '=' * 70)
    print(dataframe.head())
    print('\nDataframe stats ' + '=' * 70)
    print(dataframe.describe())
    print('\nDataframe cleaning ' + '=' * 70)
    print(f'Total number of missing values: {dataframe.isnull().sum().sum()}')


def transform_categorical(dataframe, column):
    """
    Transforms the string values of a categorical variable in numeric values. This may be helpful to avoid errors
    when using reduction algorithms such as PCA, T-Sne and so on and so forth.

    :param dataframe: the dataframe containing the column to modify.
    :param column: the column to modify within the dataframe.
    :return: the new dataframe
    """
    dataframe[column], _ = factorize(dataframe[column])
    return dataframe


def process_dataframe(dataframe):
    """
    Elaborates the dataframe given renaming all the attributes, deleting some features and setting a datetime index.

    :param dataframe: dataframe to transform.
    :return: the new dataframe.
    """
    # New features names
    columns_names = ['Time', 'Total Power', 'Generated Energy', 'House Overall', 'Dishwasher', 'Furnace 1',
                     'Furnace 2', 'Home Office', 'Fridge', 'Wine Cellar', 'Garage Door', 'Kitchen 1', 'Kitchen 2',
                     'Kitchen 3', 'Barn', 'Well', 'Microwave', 'Living Room', 'Solar', 'Temperature', 'Light',
                     'Humidity', 'visibility', 'summary', 'apparentTemperature', 'Pressure', 'Wind Speed', 'cloudCover',
                     'Wind Bearing', 'Precipitation', 'Dew Point', 'precipProbability']
    # Old names of the features
    dataframe.columns = columns_names
    # Find the initial time from which the measurements start
    start_date = to_datetime(dataframe['Time'][0], unit='s').strftime('%Y-%m-%d %H:%M')
    # The observations are measured every one minute
    dataframe['Time'] = date_range(start_date, periods=len(dataframe), freq='min')
    # Set the time column as the index of the dataframe
    dataframe.index = dataframe['Time']
    # The power consumption of the kitchens is joined in a single variable
    dataframe['Kitchen'] = dataframe[['Kitchen 1', 'Kitchen 2', 'Kitchen 3', 'Fridge', 'Microwave', 'Wine Cellar',
                                      'Dishwasher']].sum(axis=1)
    # The power consumption of the furnaces is joined in a single variable
    dataframe['Furnace'] = dataframe[['Furnace 1', 'Furnace 2']].sum(axis=1)
    # The power consumed outside the house is joined as well
    dataframe['Outside'] = dataframe[['Well', 'Garage Door', 'Barn']].sum(axis=1)
    # The variable House Overall is equal to the Total Energy variable. Then it is not necessary.
    # Remove the columns that are not used
    columns_to_remove = ['Time', 'visibility', 'summary', 'apparentTemperature', 'cloudCover', 'precipProbability',
                         'Kitchen 1', 'Kitchen 2', 'Kitchen 3', 'House Overall', 'Generated Energy', 'Solar', 'Fridge',
                         'Microwave', 'Furnace 1', 'Furnace 2', 'Wine Cellar', 'Dishwasher', 'Barn', 'Well',
                         'Garage Door', 'Light']
    dataframe = dataframe.drop(columns_to_remove, axis=1)
    return dataframe


def get_time_details(dataframe):
    """
    Adds columns to the dataframe defining more clearly time details.

    :param dataframe: dataframe to elaborate.
    :return: the new dataframe.
    """
    # Get the datetime index once
    index = dataframe.index
    # Retrieve details on time
    dataframe['Week Day'] = index.weekday
    dataframe['Hour'] = index.hour
    dataframe['Day'] = index.day
    dataframe['Month'] = index.month
    dataframe['Day Of Year'] = index.dayofyear
    dataframe['Week Of Year'] = index.isocalendar().week
    dataframe['Day Moment'] = dataframe['Hour'].apply(day_moment)
    return dataframe


def day_moment(hour):
    """
    :returns the moment of the day related to the hour selected.

    :param hour: integere in range [0,24).
    """
    # Morning
    if 3 < hour < 12:
        return 1
    # Afternoon
    if 11 < hour < 17:
        return 2
    # Evening
    if 16 < hour < 22:
        return 3
    # Night
    else:
        return 4


def compute_metrics(true_values, predictions, model='Model'):
    """
    Computes metrics (MAPE, RMSE, CORR, R2, MAE, MPE, MSE) to evaluate models performance.

    :param true_values: the true values of the test dataset.
    :param model: string defining the model used. default_value='Model'
    :param predictions: the predicted values.
    :return:
    """
    # Compute errors
    errors = true_values - predictions
    # Compute and print metrics
    mse = mean(errors ** 2)
    mae = mean(abs(errors))
    rmse = sqrt(mse)
    mape = mean(abs(predictions - true_values) / abs(true_values))
    mpe = mean((predictions - true_values) / true_values)
    corr = corrcoef(predictions, true_values)[0, 1]
    r_squared = 1 - (sum(errors ** 2) / sum((true_values - mean(true_values)) ** 2))
    print(f'\n{model} results by manual calculation:\n',
          f'- MAPE: {mape:.4f} \n - RMSE: {rmse:.4f} \n - CORR: {corr:.4f} \n - R2:   {r_squared:.4f}\n',
          f'- MAE:  {mae:.4f} \n - MPE:  {mpe:.4f} \n - MSE:  {mse:.4f}')


def residuals_properties(residuals, model='Model'):
    """
    Computes statistical values and displays plots to evaluate how the models fitted the training dataset. The residuals
    in a time series model are what is left over after fitting a model.

    :param model: string to identify the model. default_value='Model'
    :param residuals: residuals of the model.
    :return:
    """
    # Compute mean, median, skewness, kurtosis and durbin statistic
    mean_value = residuals.mean()
    median = np.median(residuals)
    # skewness = 0 : same weight in both the tails such as a normal distribution.
    skew = stats.skew(residuals)
    # Kurtosis is the degree of the peak of a distribution.
    # 3 it is normal, >3 higher peak, <3 lower peak
    kurtosis = stats.kurtosis(residuals)
    # Values between 0 and 2 indicate positive and values between 2 and 4 indicate negative auto-correlation.
    durbin = durbin_watson(residuals)
    # Anderson-Darling test null hypothesis: the sample follows the normal distribution
    anderson = stats.normaltest(residuals)[1]
    print(f'{model} residuals information:\n - Mean: {mean_value:.4f} \n - Median: {median:.4f} \n - Skewness: '
          f'{skew:.4f} \n - Kurtosis: {kurtosis:.4f}\n - Durbin: {durbin:.4f}\n - Anderson p-value: {anderson:.4f}')
    # Create plots
    sn.set()
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.3))
    # Compute standardized residuals
    residuals = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)
    # First picture: q-q plot
    # Keep only not NaN residuals.
    residuals_non_missing = residuals[~(np.isnan(residuals))]
    qqplot(residuals_non_missing, line='s', ax=axes[0])
    axes[0].set_title('Normal Q-Q')
    # Second picture: simple plot of standardized residuals
    x = np.arange(0, len(residuals), 1)
    sn.lineplot(x=x, y=residuals, ax=axes[1])
    axes[1].set_title('Standardized residual')
    # Third picture: comparison between residual and gaussian distribution
    kde = stats.gaussian_kde(residuals_non_missing)
    x_lim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(x_lim[0], x_lim[1])
    axes[2].plot(x, stats.norm.pdf(x), label='Normal (0,1)', lw=2)
    axes[2].plot(x, kde(x), label='Residuals', lw=2)
    axes[2].set_xlim(x_lim)
    axes[2].legend()
    axes[2].set_title('Estimated density')
    # Last pictures: residuals auto-correlation plots
    plot_acf(residuals, ax=axes[3], lags=30)
    plot_pacf(residuals, ax=axes[4], lags=30)
    fig.tight_layout()
    plt.show()
