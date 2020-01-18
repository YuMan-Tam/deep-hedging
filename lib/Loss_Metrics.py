import tensorflow.keras.backend as K
import tensorflow as tf

class Loss_Metrics:
	def __init__(self, wealth = None, w = None):
		self.wealth=wealth
		self.w=w
		
	def CVaR(self,loss_param = None):
		alpha = loss_param
		
		# Expected shortfall risk measure
		return K.mean(self.w + (K.maximum(-self.wealth-self.w,0)/(1.0-alpha)))
	
	def Entropy(self,loss_param = None):
		_lambda = loss_param
		# Entropy (exponential) risk measure
		return (1/_lambda)*K.log(K.mean(K.exp(-_lambda*self.wealth)))
