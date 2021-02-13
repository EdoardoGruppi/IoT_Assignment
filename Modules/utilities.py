# Import packages
from pandas import DatetimeIndex, to_datetime, date_range


def get_info(dataframe):
    print('\nDataframe info ' + '=' * 30)
    print(dataframe.info())
    # todo types of variables


def remove_features(dataframe):
    columns_names = ['time', 'Total Energy ', 'Generated Energy', 'House Overall', 'Dishwasher', 'Furnace 1',
                     'Furnace 2', 'Home Office', 'Fridge', 'Wine Cellar', 'Garage Door', 'Kitchen 1', 'Kitchen 2',
                     'Kitchen 3', 'Barn', 'Well', 'Microwave', 'Living Room', 'Solar', 'Temperature', 'Light',
                     'Humidity', 'visibility', 'summary', 'apparentTemperature', 'Pressure', 'Wind Speed', 'cloudCover',
                     'Wind Bearing', 'Precipitation', 'Dew Point', 'precipProbability']
    dataframe.columns = columns_names
    start_date = to_datetime(dataframe['time'][0], unit='s').strftime('%Y-%m-%d %H:%M')
    dataframe['time'] = date_range(start_date, periods=len(dataframe), freq='min')
    dataframe = dataframe.set_index(DatetimeIndex(dataframe['time']))
    columns_to_remove = ['time', 'visibility', 'summary', 'apparentTemperature', 'cloudCover', 'precipProbability']
    return dataframe.drop(columns_to_remove, axis=1)
