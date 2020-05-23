import tensorflow.keras.backend as K


def CVaR(wealth = None, w = None, loss_param = None):
    alpha = loss_param
    # Expected shortfall risk measure
    return K.mean(w + (K.maximum(-wealth-w,0)/(1.0-alpha)))
