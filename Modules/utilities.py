# Import packages
from pandas import to_datetime, date_range
import numpy as np


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


def remove_features(dataframe):
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
    # Remove the columns that are not used
    columns_to_remove = ['Time', 'visibility', 'summary', 'apparentTemperature', 'cloudCover', 'precipProbability']
    return dataframe.drop(columns_to_remove, axis=1)


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
