from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
				Lambda, Add, Dot, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
from tensorflow.keras.activations import tanh, relu, linear
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from loss_metrics import Entropy, CVaR

intitalizer_dict = { 
	"he_normal": he_normal(),
	"zeros": Zeros(),
	"he_uniform": he_uniform(),
	"truncated_normal": TruncatedNormal()
}

bias_initializer=he_uniform()

def Deep_Hedging_Model(N = None, d = None, m = None, \
	risk_free = None, dt = None, initial_wealth = 0.0, epsilon = 0.0, \
	strategy_type = None, use_batch_norm = None, kernel_initializer = "he_uniform", \
	activation_dense = "relu", activation_output = "linear", final_period_cost = False, \
	output_type = "gui", loss_param = None):
		
	# State variables.
	prc = Input(shape=(1,), name = "prc_0")
	information_set = Input(shape=(1,), name = "information_set_0")

	inputs = [prc, information_set]
	layers = [None for _ in range((N+1)*d)]
	for j in range(N+1):            
		if j < N:
			# The information set is P_{t} and strategy_{t-1}
			if j == 0:
				helper1 = information_set
			else:
				if strategy_type is "recurrent":
					helper1 = Concatenate()([information_set,strategy])
				elif strategy_type is "simple":
					helper1 = information_set
				
			# d hidden layers (each with m neurons) for the hedging strategy.
			for i in range(d):
				if i == 0:
					layers[i+(j)*d] = Dense(m,
							 kernel_initializer=kernel_initializer,
							 bias_initializer=bias_initializer,
							 use_bias=(not use_batch_norm), 
							 name = "dense_" + str(i)+ "_" + str(j))(helper1)
					
					if use_batch_norm:
						# Batch normalization.
						layers[i+(j)*d] = BatchNormalization(momentum = 0.99, trainable=True, \
																	name= "BatchNorm_" + str(i)+ "_" + str(j) \
													)(layers[i+(j)*d], training=True)
					
					if activation_dense is "leaky_relu":
						strategyhelper = LeakyReLU()(layers[i+(j)*d])
					else:
						strategyhelper = Activation(activation_dense)(layers[i+(j)*d])
					
				elif i != d-1:
					layers[i+(j)*d] = Dense(m,
							 kernel_initializer=kernel_initializer,
							 bias_initializer=bias_initializer,
							 use_bias=(not use_batch_norm),
							 name = "dense_" + str(i)+ "_" + str(j))(strategyhelper)
					
					if use_batch_norm:
						# Batch normalization                        
						layers[i+(j)*d] = BatchNormalization(momentum = 0.99, trainable=True, \
																	name= "BatchNorm_" + str(i)+ "_" + str(j)
													)(layers[i+(j)*d], training=True)
													
					if activation_dense is "leaky_relu":
						strategyhelper = LeakyReLU()(layers[i+(j)*d])
					else:
						strategyhelper = Activation(activation_dense)(layers[i+(j)*d])
				else:
					strategyhelper = Dense(1,
						 kernel_initializer=kernel_initializer,
						 bias_initializer=bias_initializer,
						 use_bias=True, 
						 name = "dense_" + str(i)+ "_" + str(j))(strategyhelper)
						 
					if activation_output is "leaky_relu":
						strategyhelper = LeakyReLU(name = "delta_" + str(j))(strategyhelper)
					else:
						strategyhelper = Activation(activation_output, name = "delta_" + str(j))(strategyhelper)
						 
			# strategy_-1 is set to 0
			# delta_strategy = strategy_{t+1} - strategy_t
			if j == 0:              
				delta_strategy = strategyhelper
			else:
				delta_strategy = Subtract(name = "delta_strategy_" + str(j))([strategyhelper, strategy])
				
			# Proportional transaction cost
			absolutechanges = Lambda(lambda x : K.abs(x), name = "absolutechanges_" + str(j))(delta_strategy)
			costs = Dot(axes=1)([absolutechanges,prc])
			costs = Lambda(lambda x : epsilon*x, name = "cost_" + str(j))(costs)
			
			if j == 0:
				wealth = Lambda(lambda x : initial_wealth - x, name = "costDot_" + str(j))(costs)
			else:
				wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
			
			# Wealth for the next period
			# w_{t+1} = w_t + (strategy_t-strategy_{t+1})*prc_t
			#         = w_t - delta_strategy*prc_t
			mult = Dot(axes=1)([delta_strategy, prc])
			wealth = Subtract(name = "wealth_" + str(j))([wealth, mult])

			# Accumulate interest rate for next period.
			FV_factor = np.exp(risk_free*dt)
			wealth = Lambda(lambda x: x*FV_factor)(wealth)
			
			prc = Input(shape=(1,),name = "prc_" + str(j+1))
			information_set = Input(shape=(1,), name = "information_set_" + str(j+1))
			
			strategy = strategyhelper    
			
			if j != N - 1:
				inputs += [prc, information_set]
			else:
				inputs += [prc]
		else:
			# The paper assumes no transaction costs for the final period 
			# when the position is liquidated.
			if final_period_cost:
				# Proportional transaction cost
				absolutechanges = Lambda(lambda x : K.abs(x), name = "absolutechanges_" + str(j))(strategy)
				costs = Dot(axes=1)([absolutechanges,prc])
				costs = Lambda(lambda x : epsilon*x, name = "cost_" + str(j))(costs)
				wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
			
			# Wealth for the final period
			# -delta_strategy = strategy_t
			mult = Dot(axes=1)([strategy, prc])
			wealth = Add()([wealth, mult])
				 
			# Add the terminal payoff of any derivatives.
			payoff = Input(shape=(1,), name = "payoff")
			inputs += [payoff]
			
			wealth = Add(name = "wealth_" + str(j))([wealth,payoff])
	if output_type == "gui:
		return Model(inputs=inputs, outputs=wealth)
	elif output_type == "colab":
		w = tf.Variable(0.0, name = "certainty_equiv")
		if loss_type == "Entropy":
			loss = Entropy(wealth,w,loss_param)
		elif loss_type == "CVaR":
			loss = CVaR(wealth,w,loss_param)
		model = Model(inputs=inputs, outputs=wealth)
		model.add_loss(loss)
		return model
