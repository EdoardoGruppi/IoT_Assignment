# Import packages
import seaborn as sn
import matplotlib.pyplot as plt


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
    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(20, 12))
    sn.kdeplot(data=dataframe, x=x, hue=hue, ax=axes[0])
    sn.boxplot(y=x, data=dataframe, x=hue, ax=axes[1])
    plt.tight_layout()
    plt.show()
