import tensorflow.keras.backend as K
import tensorflow as tf

@tf.function
def CVaR(wealth = None, w = None, loss_param = None):
	alpha = loss_param
	# Expected shortfall risk measure
	return K.mean(w + (K.maximum(-wealth-w,0)/(1.0-alpha)))

@tf.function
def Entropy(wealth = None, w = None, loss_param = None):
	_lambda = loss_param
	# Entropy (exponential) risk measure
	return (1/_lambda)*K.log(K.mean(K.exp(-_lambda*wealth)))