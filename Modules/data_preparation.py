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
    # Target column
    train_target = train[target_column]
    valid_target = valid[target_column]
    test_target = test[target_column]
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
    # Rescale all the other features to develop on the same range of the target column and...
    # Re-create the dataframes from the array returned by the various algorithms involved
    scaler = MinMaxScaler(feature_range=(train[target_column].min(), train[target_column].max()))
    train = DataFrame(scaler.fit_transform(train_data), index=train.index)
    valid = DataFrame(scaler.fit_transform(valid_data), index=valid.index)
    test = DataFrame(scaler.fit_transform(test_data), index=test.index)
    # Transform in string all the names of the columns to avoid conflicts later.
    train.columns = [str(col) for col in train.columns]
    valid.columns = [str(col) for col in valid.columns]
    test.columns = [str(col) for col in test.columns]
    return train, train_target, valid, valid_target, test, test_target
