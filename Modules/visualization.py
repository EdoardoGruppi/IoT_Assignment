# Import packages
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from Modules.config import *
from pmdarima.arima import ADFTest, ndiffs
from pandas import Series, concat
from scipy.stats import pearsonr


def plot_correlation_matrix(dataframe, figsize=(10, 10)):
    """
    Computes and plots the correlation matrix using the heatmap function of seaborn.

    :param dataframe: dataframe to analyse.
    :param figsize: size of the figure plotted. default_value=(10, 10)
    :return:
    """
    # Compute the correlation matrix
    corr = dataframe.corr()
    sn.set()
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sn.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sn.heatmap(corr, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_distribution(dataframe, x, hue=None):
    """
    Plots the distribution and the box-plots of one variable in relation to another feature of the observations.

    :param dataframe: dataframe containing the features.
    :param x: main feature.
    :param hue: second feature. default_value=None
    :return:
    """
    sn.set()
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    sn.kdeplot(data=dataframe, x=x, hue=hue, ax=axes[0])
    sn.boxplot(y=x, data=dataframe, x=hue, ax=axes[1])
    plt.tight_layout()
    plt.show()


def detect_univariate_outlier(dataframe, cap=None, nan=False):
    """
    Detects univariate outliers for each attribute belonging to the dataframe passed. The function displays three
    distinct plots (boxplot, scatter plot with iqr outliers, scatter plot after calculating z-score) where outliers are
    displayed with red dots. Once the outliers are detected the function can cap them (according to the maximum value
    that does not correspond to outliers with either iqr score or Z-score) or replace their values with NaN.

    :param dataframe: dataset to analyse.
    :param cap: if 'iqr' or 'z_score' operates on the outliers detected with iqr score and z-score respectively.
        default_value=None
    :param nan: if it is True the outlier selected according to the cap parameter are replaced by NaN, otherwise they
        are capped with the maximum non-outlier value. default_value=False
    :return:
    """
    # Outliers rows are not deleted. Alternatives considered are: doing nothing, capping or replacing their values.
    for column in dataframe:
        data = dataframe[column]
        # Create the figure
        sn.set()
        fig, axes = plt.subplots(1, 3, figsize=(25, 6), gridspec_kw={'width_ratios': [1, 7, 7]})
        sn.boxplot(data=data, ax=axes[0])
        # Compute iqr outliers and their position
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        minimum = q1 - 1.5 * (q3 - q1)
        maximum = q3 + 1.5 * (q3 - q1)
        # Save the position of the iqr outliers in the dataframe
        outlier_iqr_loc = dataframe.index[np.where((data < minimum) | (data > maximum))]
        sn.scatterplot(x=dataframe.index, y=data, hue=(data < minimum) | (data > maximum), s=2, ax=axes[1])
        axes[1].legend(loc='upper right')
        # Compute Z-score outliers and their position
        z = np.abs(stats.zscore(data))
        # Save the position of the z-score outliers in the dataframe
        outlier_z_loc = dataframe.index[np.where(z > 3)]
        sn.scatterplot(x=dataframe.index, y=z, hue=z > 3, s=2, ax=axes[2])
        axes[2].legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        # Change the outlier values if required and if there are outliers
        if cap:
            outliers_dict = {"z_score": outlier_z_loc, "iqr": outlier_iqr_loc}
            outliers = outliers_dict[cap]
            if len(outliers) > 0:
                if nan:
                    # Replace with Nan values the z-score outliers
                    data[outliers] = float('nan')
                else:
                    # Keep all the elements that does not correspond to outliers
                    dropped_outlier_dataset = data[np.setdiff1d(data.index, outliers)]
                    # Set all the outliers values as the maximum value of the non outliers data points
                    data[outliers] = np.max(dropped_outlier_dataset)
            # Save the changed values in the dataframe passed
            dataframe[column] = data
    return dataframe


def plot_distributions(dataframe, columns=3):
    """
    Plot the distributions of all the variables within the dataframe given.

    :param dataframe: dataframe to visualize.
    :param columns: number of plots for each column in the figure.
    :return:
    """
    sn.set()
    fig = plt.figure(figsize=(20, 12))
    rows = int(np.ceil(dataframe.shape[1] / columns))
    # Add a subplot for each series in the dataframe
    for i, column in enumerate(dataframe.columns):
        fig.add_subplot(rows, columns, i + 1)
        # If the variable is categorical...
        if dataframe.dtypes[column] == np.object:
            g = sn.countplot(y=column, data=dataframe)
            substrings = [label.get_text()[:18] for label in g.get_yticklabels()]
            g.set(yticklabels=substrings)
        else:
            sn.kdeplot(data=dataframe, x=column)
    # Format the figure
    plt.tight_layout()
    plt.show()


def scatter_plot_matrix(dataframe):
    """
    Plots the scatter plot matrix of the attributes available in the dataframe given.

    :param dataframe: dataframe to analyse.
    """
    sn.set()
    # Scatter plot matrix
    plt.figure(figsize=(20, 20))
    g = sn.pairplot(dataframe, plot_kws=dict(s=0.5), diag_kind='hist', diag_kws=dict(kde=True, bins=50))
    g.map_upper(pearson_corr)
    plt.tight_layout()
    plt.show()


def hue_scatter_plot_matrix(dataframe, columns, hue=None):
    """
    Plots a scatter plot matrix where the data points are coloured according to the day, the weekday, the month or
    the year in which they are measured.

    :param dataframe: dataframe to visualize. If hue is not None it must be in a tidy format.
    :param columns: columns to consider in both the plots.
    :param hue: define how to separate the data points in the scatter plots. It can be 'Day', 'Month', 'Year', 'Quarter'
        and/or 'WeekDay'. default_value=None
    :return:
    """
    sn.set()
    # Plot the pairwise joint distributions
    if hue is not None:
        for label in hue:
            x = dataframe[columns].join(dataframe[label])
            sn.pairplot(x, hue=label, plot_kws=dict(s=0.3))
    else:
        sn.pairplot(dataframe[columns])
    plt.tight_layout()
    plt.show()


def box_plot(dataframe, columns, figsize=(35, 8)):
    """
    Plots the univariate box-plots of each attribute of the dataset passed.

    :param dataframe: dataframe to analyse.
    :param columns: columns to consider in both the plots.
    :param figsize: size of the figure plotted. default_value=(35, 8)
    :return:
    """
    sn.set()
    figure = plt.figure(figsize=figsize)
    for index, col in enumerate(columns):
        figure.add_subplot(1, len(columns), index + 1)
        g = sn.boxplot(y=col, data=dataframe, linewidth=3.5)
        g.set(xlabel=None, ylabel=None)
        g.set_title(col, fontdict={'fontsize': 20})
        figure.tight_layout()
    plt.show()


def detect_seasonality(dataframe, y_axis, x_axis, hue, figsize=(20, 10)):
    """
    Displays a seasonal plot to detect recurrent patterns in data throughout years, months, days, etc.

    :param dataframe: input dataframe.
    :param y_axis: target column to visualize.
    :param x_axis: variable that specify positions on the x axes.
    :param hue: grouping variable that will produce lines with different colors. To detect some sort of seasonality
        it should be one of the variables related to the time.
    :param figsize: size of the figure plotted. default_value=(20, 10)
    :return:
    """
    sn.set(font_scale=1)
    plt.subplots(1, 1, sharey='all', figsize=figsize)
    g = sn.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue, legend='full')
    g.set(xlabel=None, ylabel=None)
    g.set_xlabel('Hour', fontdict={'fontsize': 20})
    g.set_ylabel('Total Power', fontdict={'fontsize': 20})
    plt.tight_layout()
    plt.show()


def plot_auto_correlation(series, lags=None):
    """
    Plots the auto correlation functions of the series provided.

    :param series: time series to analyse.
    :param lags: an int or array of lag values, used on horizontal axis. default_value=None
    :return:
    """
    sn.set()
    fig, axes = plt.subplots(1, 2, sharey='all', figsize=(20, 7))
    plot_acf(series, ax=axes[0], lags=lags)
    plot_pacf(series, ax=axes[1], lags=lags)
    fig.tight_layout()
    plt.show()


def plot_series(series):
    """
    Simple functions to plot the boxplot and the distribution of a time series.

    :param series: series to analyse.
    :return:
    """
    sn.set()
    fig, ax = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 7]})
    sn.boxplot(data=series, linewidth=2, ax=ax[0])
    sn.lineplot(data=series, linewidth=2, ax=ax[1])
    plt.tight_layout()
    plt.show()


def decompose_series(series, period=None, mode='multiplicative'):
    """
    Decomposes a series using moving averages.

    :param series: time series to decompose.
    :param period: period of the series. Required if x is not a pandas dataframe. default_value=None
    :param mode: type of seasonal component ('additive', 'multiplicative'). default_value='multiplicative'
    :return:
    """
    sn.set()
    # Decompose the series using the seasonal_decompose function from the stats-model library
    result = seasonal_decompose(series, model=mode, period=period)
    # Plot the results all together
    fig, axes = plt.subplots(ncols=1, nrows=4, sharex='all', figsize=(30, 10))
    result.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    result.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    result.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    result.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    fig.tight_layout()
    plt.show()


def pie_chart(dataframe, columns, title='Pie Chart'):
    """
    Plots a pie chart using the columns chosen from the dataframe passed.

    :param dataframe: dataframe from which the information is retrieved.
    :param columns: list of columns to visualize in the pie chart.
    :param title: title displayed in the chart. default_value='Pie Chart'
    :return:
    """
    sn.set()
    # Get the sum of all the values of each column
    pie_values = [dataframe[col].sum() for col in columns]
    # This variable specifies the fraction of the radius with which to offset each wedge
    explode = np.zeros(shape=len(columns))
    # In this case only the first element is detached from the main body of the pie
    explode[0] = 0.1
    # Plot the pie chart
    plt.pie(pie_values, labels=columns, explode=explode, shadow=True, autopct='%1.2f%%', startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def granger_test(dataframe, target_column, max_lag=1, test='ssr_ftest', figsize=(20, 10)):
    """
    Performs the granger test on the time series passed. Null hypothesis: the second time series, x2, does NOT Granger
    cause the time series of interest, x1. Grange causality means that past values of x2 have a statistically
    significant effect on the current value of x1.

    :param dataframe: dataset to analyse.
    :param target_column: time series x1.
    :param max_lag: maximum number of lags to consider. default_value=5
    :param test: which test to keep between the four computed by the stat function. default_value='ssr_ftest'
    :param figsize: size of the figure plotted. default_value=(20, 10)
    :return:
    """
    print('\nExecuting Granger Test...')
    results = []
    # Select all the columns in the dataframe except the target column
    columns = [col for col in dataframe.columns if col != target_column]
    # For every column different by the target column compute the granger test.
    for col_name in columns:
        # The granger test results are returned as dictionary
        dictionary = grangercausalitytests(dataframe[[target_column, col_name]], maxlag=max_lag)
        # For every tuple (max = number of lags) in the dictionary, save only the obtained p-value
        results.append(Series([item[0][test][1] for item in dictionary.values()], name=col_name))
    # Create a dataframe with the results achieved
    results = concat(results, axis=1)
    sn.set(font_scale=1)
    plt.figure(figsize=figsize)
    # Display the results collected by means of a heatmap
    sn.heatmap(results, annot=True, linewidths=2, cmap='GnBu_r', cbar=False, square=True, fmt='.2f',
               annot_kws={'c': 'k'}, vmax=1, vmin=-0.5)
    plt.tight_layout()
    plt.show()
    # Print the results as well
    print(f'\n{results}')


def check_single_stationarity(time_series):
    """
    Performs the Augmented Dickey-Fuller test wherein the null hypothesis is: data is not stationary. Adopting an alpha
    value of 0.05, the null hypothesis will be rejected only when the confidence is greater than 95%. This function also
    returns the differencing order of the series.

    :param time_series: series to analyse with the ADF test.
    """
    # Make sure that the original time series is not modified
    series = time_series.copy()
    sn.set()
    # Plot the original time-series along with the final result
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    sn.lineplot(x=series.index, y=series.values, ax=axes[0])
    axes[0].set_title('Original series')
    print('\nResults of Dickey-Fuller Test:')
    # Significance level to reject the null hypothesis is set to 0.05
    adf_test = ADFTest(alpha=0.05)
    # Differencing order
    diff_order = 0
    # Series after differencing. At the beginning it is equal to the original series
    diff_series = series.copy()
    while True:
        # Compute the ADF test. It returns the p-value and if the differencing is needed
        results, should = adf_test.should_diff(series)
        print(f'Differencing order: {diff_order} - P-value: {results:.4f} - Stop: {not should}')
        # Should is a boolean to understand if the series need the differencing
        if should:
            # If it is not already stationary, apply the differencing of one order above
            diff_order += 1
            diff_series = series.diff(periods=diff_order).bfill()
        else:
            break
    # Plot the stationary series.
    sn.lineplot(x=diff_series.index, y=diff_series.values, ax=axes[1])
    axes[1].set_title(f'Trend-stationary series after differencing (diff.order: {diff_order})')
    plt.tight_layout()
    plt.show()


def check_stationarity(dataframe):
    """
    Performs the Augmented Dickey-Fuller test on all the series constituting the dataframe given.

    :param dataframe: dataframe to analyse with the ADF test.
    """
    # Make sure that the original time series is not modified
    data = dataframe.copy()
    print('\nResults of Dickey-Fuller Test:')
    print('{:<15}{:<15}{:<10}{:<10}'.format('Column', 'Stationary', 'P-value', 'Order'))
    # Significance level to reject the null hypothesis is set to 0.05
    adf_test = ADFTest(alpha=0.05)
    # Cycle across each column of the dataframe
    columns = dataframe.columns
    for column in columns:
        # If the series is already stationary the differencing order is equal to 0
        order = 0
        # Compute the ADF test. It returns the p-value and if the differencing is needed
        results, should = adf_test.should_diff(data[column])
        # If the series must be differenced
        if should:
            # The differencing order needed to transform the series in stationary is computed by ndiffs()
            order = ndiffs(data[column], alpha=0.05)
        print(f'{column:<15}{not should:<15}{results:<10.5f}{order:<10}')


def pearson_corr(input_1, input_2, **kws):
    """
    Function to print the correlation between two variables on the scatter plot matrix.

    :param input_1: first variable.
    :param input_2: second variable.
    :param kws: additional arguments.
    """
    # Compute the pearson correlation between the two variables
    result, _ = pearsonr(input_1, input_2)
    # Create the string to print on the plots
    label = r'$\rho$ = ' + str(round(result, 3))
    # Retrieve the ax of the figure
    ax = plt.gca()
    # Print the string built in the upper right corner of the picture
    ax.annotate(label, xy=(.65, .9), size=10, xycoords=ax.transAxes)


def plot_results(test_predictions, test_target):
    """
    Plots the predictions made in comparison with the true values.

    :param test_predictions: predictions on the test dataset.
    :param test_target: true values.
    """
    # Plot comparison between forecasting results and predictions
    sn.set(font_scale=1.5)
    f, ax = plt.subplots(figsize=(30, 10))
    ax.plot(test_target.index, test_target, color='b', lw=1.5, label='Ground Truth')
    ax.plot(test_target.index, test_predictions, color='coral', lw=1.3, label='Predictions')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Joint plot between the true and predicted values
    sn.set(font_scale=1)
    g = sn.jointplot(x=test_predictions, y=test_target.values, kind="reg", color="b")
    g.set_axis_labels('Predictions', 'Observations')
    plt.tight_layout()
    plt.show()
