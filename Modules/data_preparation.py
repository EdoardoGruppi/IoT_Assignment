# Import packages
from pandas import concat, DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE


def dataset_division(dataframe, valid_size=0.05, test_size=0.05):
    """
    Divides the dataset received in three parts: train, validation and test. The validation and test size is expressed
    in percentage.

    :param dataframe: dataset to segment.
    :param valid_size: dimension of the validation set as percentage. default_value=0.05
    :param test_size: dimension of the test set as percentage. default_value=0.05
    :return: the three parts obtained after the splitting.
    """
    # Compute the number of observation dedicated to training and validation
    valid_samples = round(dataframe.shape[0] * valid_size)
    train_samples = round(dataframe.shape[0] * (1 - valid_size - test_size))
    # Divide the dataset in consecutive parts given that it is a time-series
    train = dataframe.iloc[:train_samples, :]
    valid = dataframe.iloc[train_samples:(train_samples + valid_samples), :]
    test = dataframe.iloc[(train_samples + valid_samples):, :]
    return train, valid, test


def transform_dataset(train, valid, test, target_column, algorithm='pca', n_components=0.95, kernel='rbf', perplexity=5,
                      reduction=True):
    """
    Transforms the dataset computing several operations. Firstly, it excludes by any computation the time series to
    predict. Secondly, if dimension reduction is required before applying the algorithm selected (pca, kernel pca,
    t-sne) it normalizes the input in the range [0,1]. Normalization before dimension reduction is crucial. Then, in
    both the cases (reduction requested or not requested) it brings all the features to the same scale, i.e. the scale
    of the series to predict.

    :param train: train dataset.
    :param valid: validation dataset.
    :param test: test dataset.
    :param target_column: column of the dataframe to predict.
    :param algorithm: dimensionality reduction algorithm to apply ('pca','pca_kernel','t-sne'). It is valid only if
        reduction=True. default_value='pca'
    :param n_components: number of components to keep after computing one of the algorithms. It can also be used to
        express instead the variance to kept. default_value=2
    :param kernel: valid only if reduction is True and algorithm is 'kernel_pca'. default_value='rbf'
    :param perplexity: valid only if reduction is True and algorithm is 't-sne'. default_value=5
    :param reduction: if True the dimensionality reduction is applied. default_value=False
    :return: train, validation and test set normalized and if required also reduced.
    """
    # Dictionary of the dimensionality reduction algorithms
    function = {'pca': PCA(n_components=n_components),
                'kernel_pca': KernelPCA(n_components=n_components, kernel=kernel),
                't-sne': TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)}
    # Exclude the main column from the computations
    train_data = train.drop([target_column], axis=1)
    valid_data = valid.drop([target_column], axis=1)
    test_data = test.drop([target_column], axis=1)
    # Apply dimensionality reduction exclusively when it is True
    if reduction:
        # Select the algorithm
        model = function[algorithm]
        # Normalization before applying dimensionality reduction
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)
        # Apply dimensionality reduction
        train_data = model.fit_transform(train_data)
        valid_data = model.transform(valid_data)
        test_data = model.transform(test_data)
    # Rescale all the other features to develop on the same range of the target column
    scaler = MinMaxScaler(feature_range=(train[target_column].min(), train[target_column].max()))
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    # Re-create the dataframes from the array returned by the various algorithms involved
    train = concat([DataFrame(train_data, index=train.index), train[target_column]], axis=1)
    valid = concat([DataFrame(valid_data, index=valid.index), valid[target_column]], axis=1)
    test = concat([DataFrame(test_data, index=test.index), test[target_column]], axis=1)
    # Transform in string all the names of the columns to avoid conflicts later.
    train.columns = [str(col) for col in train.columns]
    valid.columns = [str(col) for col in valid.columns]
    test.columns = [str(col) for col in test.columns]
    return train, valid, test



#
#
# def residuals_properties(residuals):
#     """
#     Computes statistical values and displays plots to evaluate how the models fitted the training dataset. The residuals
#     in a time series model are what is left over after fitting a model.
#     :param residuals: residuals of the model.
#     :return:
#     """
#     # Compute mean, median, skewness, kurtosis and durbin statistic
#     residuals = residuals[1:]
#     mean = residuals.mean()
#     median = np.median(residuals)
#     # skewness = 0 : same weight in both the tails such as a normal distribution.
#     # skewness > 0 : more weight in the left tail of the distribution. Long right tail. Median before mean.
#     # skewness < 0 : more weight in the right tail of the distribution. Long left tail. Median after mean.
#     skew = stats.skew(residuals)
#     # Kurtosis is the degree of the peak of a distribution.
#     # 3 it is normal, >3 higher peak, <3 lower peak
#     kurtosis = stats.kurtosis(residuals)
#     # Durbin-Watson statistic equal to  2.0 means no auto-correlation.
#     # Values between 0 and 2 indicate positive and values between 2 and 4 indicate negative auto-correlation.
#     durbin = durbin_watson(residuals)
#     # Shapiro-Wilk quantifies how likely it is that the data was drawn from a Gaussian distribution.
#     # Null hypothesis: the sample is normally distributed
#     shapiro = stats.shapiro(residuals)[1]
#     # Anderson-Darling test null hypothesis: the sample follows the normal distribution
#     anderson = stats.normaltest(residuals)[1]
#     print(f'Residual information:\n - Mean: {mean:.4f} \n - Median: {median:.4f} \n - Skewness: {skew:.4f} '
#           f'\n - Kurtosis: {kurtosis:.4f}\n - Durbin: {durbin:.4f}',
#           f'\n - Shapiro p-value: {shapiro:.4f}\n - Anderson p-value: {anderson:.4f}')
#     # Create plots
#     sn.set()
#     fig, axes = plt.subplots(1, 5, figsize=(25, 5.3))
#     # Compute standardized residuals
#     residuals = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)
#     # First picture: q-q plot
#     # Keep only not NaN residuals.
#     residuals_non_missing = residuals[~(np.isnan(residuals))]
#     qqplot(residuals_non_missing, line='s', ax=axes[0])
#     axes[0].set_title('Normal Q-Q')
#     # Second picture: simple plot of standardized residuals
#     x = np.arange(0, len(residuals), 1)
#     sn.lineplot(x=x, y=residuals, ax=axes[1])
#     axes[1].set_title('Standardized residual')
#     # Third picture: comparison between residual and gaussian distribution
#     kde = stats.gaussian_kde(residuals_non_missing)
#     x_lim = (-1.96 * 2, 1.96 * 2)
#     x = np.linspace(x_lim[0], x_lim[1])
#     axes[2].plot(x, stats.norm.pdf(x), label='Normal (0,1)', lw=2)
#     axes[2].plot(x, kde(x), label='Residuals', lw=2)
#     axes[2].set_xlim(x_lim)
#     axes[2].legend()
#     axes[2].set_title('Estimated density')
#     # Last pictures: residuals auto-correlation plots
#     plot_acf(residuals, ax=axes[3], lags=30)
#     plot_pacf(residuals, ax=axes[4], lags=30)
#     fig.tight_layout()
#     plt.show()
#
#

