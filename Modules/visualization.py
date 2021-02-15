# Import packages
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_correlation_matrix(dataframe):
    """
    Computes and plots the correlation matrix using the heatmap function of seaborn.

    :param dataframe: dataframe to analyse.
    :return:
    """
    # Compute the correlation matrix
    corr = dataframe.corr()
    sn.set()
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
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
        ax = fig.add_subplot(rows, columns, i + 1)
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
    fig = plt.figure(figsize=(20, 20))
    g = sn.pairplot(dataframe, plot_kws=dict(s=0.5), diag_kind='hist', diag_kws=dict(kde=True, bins=50))
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


def box_plot(dataframe, columns):
    """
    Plots the univariate box-plots of each attribute of the dataset passed.

    :param dataframe: dataframe to analyse.
    :param columns: columns to consider in both the plots.
    :return:
    """
    sn.set()
    figure = plt.figure(figsize=(16, 8))
    for index, col in enumerate(columns):
        figure.add_subplot(1, len(columns), index + 1)
        sn.boxplot(y=col, data=dataframe, linewidth=3.5)
        figure.tight_layout()
    plt.show()


def detect_seasonality(dataframe, y_axis, x_axis, hue):
    """
    Displays a seasonal plot to detect recurrent patterns in data throughout years, months, days, etc.

    :param dataframe: input dataframe.
    :param y_axis: target column to visualize.
    :param x_axis: variable that specify positions on the x axes.
    :param hue: grouping variable that will produce lines with different colors. To detect some sort of seasonality
        it should be one of the variables related to the time.
    :return:
    """
    sn.set()
    plt.subplots(1, 1, sharey='all', figsize=(20, 10))
    sn.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue, legend='full')
    plt.tight_layout()
    plt.show()
