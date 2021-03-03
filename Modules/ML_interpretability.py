# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.inspection import plot_partial_dependence
from sklearn.tree import DecisionTreeRegressor, plot_tree
from pdpbox import pdp
from lime.lime_tabular import LimeTabularExplainer
from shap import TreeExplainer, force_plot, dependence_plot, summary_plot


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
    plt.figure(figsize=(12, 8))
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

    :param model: the model considered. The ice plot is calculated only after the model has been fit.
    :param test: test dataset.
    :param column: variables studied.
    """
    pdp_feature = pdp.pdp_isolate(model, test, model_features=test.columns, feature=column)
    pdp.pdp_plot(pdp_feature, column, plot_lines=True, frac_to_plot=100)
    plt.tight_layout()
    plt.show()


def surrogate_tree(model, test, max_depth=5, random_state=10):
    """
    Adopts a surrogate tree to explain the inner working of a so-called black box model.

    :param model: the model considered. The surrogate tree is adopted only after the model has been fit.
    :param test: test dataset.
    :param max_depth: maximum depth of the tree. default_value=5
    :param random_state: random state of the tree. default_value=10
    :return:
    """
    # Make the predictions with the "black-box" model
    predictions = model.predict(test)
    # Defining the interpretable decision tree model
    surrogate_model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    # Fitting the surrogate decision tree model
    surrogate_model.fit(test, predictions)
    # Unfortunately there is a bug caused by the conflict between seaborn and sklearn. The only solution is to plot the
    # the decision tree under a gray background. Otherwise, the arrows are not displayed. The background of the image
    # once saved can be modified using an external lightweight software such as paint
    plt.style.use('grayscale')
    plt.figure(figsize=(20, 10))
    plot_tree(surrogate_model, feature_names=test.columns, filled=True, rounded=True, fontsize=8)
    plt.show()
    # Set the previous style
    sn.set()


def plot_shap(model, test, instance=None, feature=None, dataset=False):
    """
    Displays shap plots to explain a black box model.

    :param model: the model considered. The shap plots are calculated only after the model has been fit.
    :param test: test dataset.
    :param instance: instance of the test dataset to explain. default_value=None
    :param feature: feature of the test dataset to explain. default_value=None
    :param dataset: if True the entire dataset is taken into account. default_value=False
    :return:
    """
    # Make an explainer on the model given. Not all the models are supported
    explainer = TreeExplainer(model)
    # Compute SHAP values
    shap_values = explainer.shap_values(test)
    # If not None explain single prediction
    if instance is not None:
        force_plot(explainer.expected_value, shap_values[instance, :], test.iloc[instance, :], matplotlib=True)
    # If not None explain single feature
    if feature is not None:
        fig, ax = plt.subplots(figsize=(13, 10))
        dependence_plot(feature, shap_values, test, ax=ax)
    # If True explain the entire dataset
    if dataset:
        summary_plot(shap_values, test)


def plot_lime(model, test, instance):
    """
    Plots lime for regression.

    :param model: the model considered. The lime plot is calculated only after the model has been fit.
    :param test: test dataset.
    :param instance: instance of the test dataset to explain.
    """
    print('\nLIME Explanation results:')
    # Fit the explainer object
    explainer = LimeTabularExplainer(np.array(test), feature_names=test.columns, verbose=True, mode='regression')
    # Explain the instance required
    explanation = explainer.explain_instance(test.iloc[instance], model.predict)
    # Plot the results. The figure represents the magnitude of the influence of each feature when it is greater or
    # smaller than a certain value. The red color means negatively, the green the opposite.
    # explanation.as_pyplot_figure()
    plot_explanation(explanation.as_list())
    plt.show()
    print(f'Score:{explanation.score}')


def plot_explanation(lime_explanation):
    """
    Plots the lime explanation generated by the plot_lime function.

    :param lime_explanation: lime explanation expressed as list.
    """
    sn.set(font_scale=0.7)
    plt.subplots()
    # Retrieve the pairs of names and values of the lime explanation
    values = [item[1] for item in lime_explanation]
    names = [item[0] for item in lime_explanation]
    # Reverse the order from the largest value to the smallest
    values.reverse()
    names.reverse()
    # Define the colours of each pair according to the value field
    colors = ['green' if value > 0 else 'red' for value in values]
    pos = np.arange(len(lime_explanation)) + .5
    # Plot tje bar chart
    plt.barh(pos, values, align='center', color=colors)
    plt.yticks(pos, names)
    plt.title('Local explanation')
    plt.tight_layout()
    plt.show()
