# Import packages
from pandas import to_datetime, date_range
from Modules.utilities import *


def light_process_dataframe(df):
    """
    Elaborates the dataframe given renaming all the attributes, deleting some features and setting a datetime index.

    :param df: dataframe to transform.
    :return: the new dataframe.
    """
    dataframe = df.copy()
    # New features names
    columns_names = ['Time', 'Total Power', 'Generated Energy', 'House Overall', 'Dishwasher', 'Furnace 1',
                     'Furnace 2', 'Home Office', 'Fridge', 'Wine Cellar', 'Garage Door', 'Kitchen 1', 'Kitchen 2',
                     'Kitchen 3', 'Barn', 'Well', 'Microwave', 'Living Room', 'Solar', 'Temperature', 'Light',
                     'Humidity', 'visibility', 'summary', 'apparentTemperature', 'Pressure', 'Wind Speed', 'cloudCover',
                     'Wind Bearing', 'Precipitation', 'Dew Point', 'precipProbability']
    # Old names of the features
    dataframe.columns = columns_names
    # Find the initial time from which the measurements start
    start_date = to_datetime(dataframe['Time'].iloc[0], unit='s').strftime('%Y-%m-%d %H:%M')
    # The observations are measured every one minute
    dataframe['Time'] = date_range(start_date, periods=len(dataframe), freq='min')
    # Set the time column as the index of the dataframe
    dataframe.index = dataframe['Time']
    # The power consumption of the kitchens is joined in a single variable
    dataframe['Kitchen'] = dataframe[['Kitchen 1', 'Kitchen 2', 'Kitchen 3']].sum(axis=1)
    # The power consumption of the furnaces is joined in a single variable
    dataframe['Furnace'] = dataframe[['Furnace 1', 'Furnace 2']].sum(axis=1)
    # The variable House Overall is equal to the Total Energy variable. Then it is not necessary.
    # Remove the columns that are not used
    columns_to_remove = ['Time', 'visibility', 'summary', 'apparentTemperature', 'cloudCover', 'precipProbability',
                         'Kitchen 1', 'Kitchen 2', 'Kitchen 3', 'House Overall', 'Generated Energy', 'Solar',
                         'Furnace 1', 'Furnace 2', 'Light']
    dataframe = dataframe.drop(columns_to_remove, axis=1)
    return dataframe


def hard_process_dataframe(df):
    """
    Elaborates the dataframe given renaming all the attributes, deleting some features and setting a datetime index.

    :param df: dataframe to transform.
    :return: the new dataframe.
    """
    dataframe = df.copy()
    # New features names
    columns_names = ['Time', 'Total Power', 'Generated Energy', 'House Overall', 'Dishwasher', 'Furnace 1',
                     'Furnace 2', 'Home Office', 'Fridge', 'Wine Cellar', 'Garage Door', 'Kitchen 1', 'Kitchen 2',
                     'Kitchen 3', 'Barn', 'Well', 'Microwave', 'Living Room', 'Solar', 'Temperature', 'Light',
                     'Humidity', 'visibility', 'summary', 'apparentTemperature', 'Pressure', 'Wind Speed', 'cloudCover',
                     'Wind Bearing', 'Precipitation', 'Dew Point', 'precipProbability']
    # Old names of the features
    dataframe.columns = columns_names
    # Find the initial time from which the measurements start
    start_date = to_datetime(dataframe['Time'].iloc[0], unit='s').strftime('%Y-%m-%d %H:%M')
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


def resampling_dataframe(dataframe, resampling):
    dataframe = dataframe.resample(resampling).mean()
    return get_time_details(dataframe)


def pie_chart(dataframe, columns):
    pie_values = [dataframe[col].sum() for col in columns]
    return pie_values


def create_options(array):
    new_array = []
    for item in array:
        new_array.append({'label': item, 'value': item})
    return new_array
