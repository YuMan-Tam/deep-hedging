from sklearn import model_selection


def train_test_split(data=None, test_size=None):
    """Split simulated data into training and testing sample."""
    xtrain = []
    xtest = []
    for x in data:
        tmp_xtrain, tmp_xtest = model_selection.train_test_split(
            x, test_size=test_size, shuffle=False)
        xtrain += [tmp_xtrain]
        xtest += [tmp_xtest]
    return xtrain, xtest
