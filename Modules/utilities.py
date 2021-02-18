# Import packages
from pandas import to_datetime, date_range, factorize
from numpy import mean, abs, sqrt, corrcoef


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
    columns_names = ['Time', 'Total Energy', 'Generated Energy', 'House Overall', 'Dishwasher', 'Furnace 1',
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
    dataframe['Kitchen'] = dataframe['Kitchen 1'] + dataframe['Kitchen 2'] + dataframe['Kitchen 3'] + dataframe[
        'Fridge'] + dataframe['Microwave'] + dataframe['Wine Cellar'] + dataframe['Dishwasher']
    # The power consumption of the furnaces is joined in a single variable
    dataframe['Furnace'] = dataframe['Furnace 1'] + dataframe['Furnace 2']
    # The power consumed outside the house is joined as well
    dataframe['Outside'] = dataframe['Well'] + dataframe['Garage Door'] + dataframe['Barn']
    # The variable House Overall is equal to the Total Energy variable. Then it is not necessary.
    # Remove the columns that are not used
    columns_to_remove = ['Time', 'visibility', 'summary', 'apparentTemperature', 'cloudCover', 'precipProbability',
                         'Kitchen 1', 'Kitchen 2', 'Kitchen 3', 'House Overall', 'Generated Energy', 'Solar', 'Fridge',
                         'Microwave', 'Furnace 1', 'Furnace 2', 'Wine Cellar', 'Dishwasher', 'Barn', 'Well',
                         'Garage Door']
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
    return dataframe


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
          f'- MAPE: {mape:.4f} \n - RMSE: {rmse:.4f} \n - CORR: {corr:.4f} \n - R2: {r_squared:.4f}\n',
          f'- MAE: {mae:.4f} \n - MPE: {mpe:.4f} \n - MSE: {mse:.4f}')
