import tensorflow.keras.backend as K


def Entropy(wealth=None, w=None, loss_param=None):
    _lambda = loss_param

    # Entropy (exponential) risk measure
    return (1/_lambda)*K.log(K.mean(K.exp(-_lambda*wealth)))
