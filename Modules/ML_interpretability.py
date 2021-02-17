# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.inspection import plot_partial_dependence
from pdpbox import pdp


def features_importance(feature_importance, names):
    """
    Shows the feature importance ranking.

    :param feature_importance: feature importance values extracted from the model.
    :param names: names of all the features.
    """
    # Level of importance is relative to the maximum importance
    feature_importance = np.absolute(feature_importance / feature_importance.max())
    # Sort the feature importance values in a descending order
    feature_importance, names = zip(*sorted(zip(feature_importance, names)))
    # Display the features importance
    sn.set()
    plt.figure(figsize=(15, 10))
    plt.title('Features Importance')
    # Bar chart plot
    plt.barh(range(len(names)), feature_importance, align='center')
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.show()


def plot_partial_dependencies(model, test, column):
    """
    Plots a one way partial dependence plot with the variables in the test dataset. Partial dependence plots show how
    a particular variable or predictor affects the model's predictions.

    :param model: the model considered. The partial dependence plot is calculated only after the model has been fit.
    :param test: test dataset.
    :param column: variables studied.
    :return:
    """
    sn.set()
    plt.figure()
    plot_partial_dependence(model, test, column, n_jobs=-1)
    # Format the figure
    plt.tight_layout()
    plt.show()


def plot_two_ways_pdp(model, test, columns):
    """
    Plots a two ways partial dependence plot with the variables given through the argument columns. Two-ways Partial
    dependence plots show how a pair of variables or predictors affects the model's predictions.

    :param model: the model considered. The partial dependence plot is calculated only after the model has been fit.
    :param test: test dataset.
    :param columns: variables studied. It must be in the form [(var1, var2)].
    :return:
    """
    sn.set()
    plt.figure()
    plot_partial_dependence(model, test, columns, n_jobs=-1)
    plt.tight_layout()
    plt.show()


def plot_ice(model, test, column):
    """
    ICE plots is used to display how the prediction target changes when a feature varies at each instances. CE looks
    locally so that the model prediction is measured for each observation when the feature value changes. PDP can be
    regarded as an average of the lines of ICE plot. The ICE curves  plotted with the functions provided by the
    package pdpbox are centered to make it easier to spot the differences between curves at different instances.

    :param model: the model considered. The partial dependence plot is calculated only after the model has been fit.
    :param test: test dataset.
    :param column: variables studied.
    """
    pdp_feature = pdp.pdp_isolate(model, test, model_features=test.columns, feature=column)
    pdp.pdp_plot(pdp_feature, column, plot_lines=True, frac_to_plot=100)
    plt.tight_layout()
    plt.show()
