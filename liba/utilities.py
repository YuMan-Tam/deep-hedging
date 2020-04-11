from sklearn import model_selection
import numpy as np

# Split simulated data into training and testing sample.
def train_test_split(data = None, test_size=None):
    xtrain = []
    xtest = []
    for x in data:
        tmp_xtrain, tmp_xtest = model_selection.train_test_split(x, test_size=test_size, shuffle=False)
        xtrain += [tmp_xtrain]
        xtest += [tmp_xtest]
    return xtrain, xtest
    
# Reshape (n,1) numpy array to (n,) numpy array
def reshape_1D(array = None):
    return array.reshape(len(array))