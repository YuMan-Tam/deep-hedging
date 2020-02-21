from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
				Lambda, Add, Dot, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
from tensorflow.keras.activations import tanh, relu, linear
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from Loss_Metrics import Loss_Metrics

intitalizer_dict = { 
	"he_normal": he_normal(),
	"zeros": Zeros(),
	"he_uniform": he_uniform(),
	"truncated_normal": TruncatedNormal()
}

class Deep_Hedging:
	def __init__ (self, N = None, d = None, m = None, \
			risk_free = None, maturity = None, \
			num_days_in_a_year = 365):
		self.N = N
		self.d = d
		self.m = m
		self.maturity = maturity
		self.risk_free = risk_free
		
		self.num_days_in_a_year = num_days_in_a_year
		
	def model(self, inititial_wealth = 0.0, strategy_type = None, \
							epsilon = 0.0, loss_type = None, \
							use_batch_norm = True, kernel_initializer = "he_uniform", \
							activation_dense = "relu", activation_output = "linear", 
							final_period_cost = False, **kwargs):
		# State variables.
		prc = Input(shape=(1,), name = "prc_0")
		information_set = Input(shape=(1,), name = "information_set_0")
		
		# The control variable is the hedging strategy and
		# certainty equivalent.
		w = Input(shape=(1,))
		
		inputs = [w, prc, information_set]
		layers = [None for _ in range((self.N+1)*self.d)]
		for j in range(self.N+1):            
			if j < self.N:
				# The information set is P_{t} and strategy_{t-1}
				if j == 0:
					helper1 = information_set
				else:
					if strategy_type is "recurrent":
						helper1 = Concatenate()([information_set,strategy])
					elif strategy_type is "simple":
						helper1 = information_set
					
				# d hidden layers (each with m neurons) for the hedging strategy.
				for i in range(self.d):
					if i == 0:
						layers[i+(j)*self.d] = Dense(self.m,
							   kernel_initializer=kernel_initializer,
							   use_bias=(not use_batch_norm), 
							   name = "dense_" + str(i)+ "_" + str(j))(helper1)
						
						if use_batch_norm:
							# Batch normalization.
							layers[i+(j)*self.d] = BatchNormalization(momentum = 0.99, trainable=True, \
																		name= "BatchNorm_" + str(i)+ "_" + str(j) \
														)(layers[i+(j)*self.d], training=True)
						
						if activation_dense is "leaky_relu":
							strategyhelper = LeakyReLU(layers[i+(j)*self.d])
						else:
							strategyhelper = Activation(activation_dense)(layers[i+(j)*self.d])
						
					elif i != self.d-1:
						layers[i+(j)*self.d] = Dense(self.m,
							   kernel_initializer=kernel_initializer,
							   use_bias=(not use_batch_norm),
							   name = "dense_" + str(i)+ "_" + str(j))(strategyhelper)
						
						if use_batch_norm:
							# Batch normalization                        
							layers[i+(j)*self.d] = BatchNormalization(momentum = 0.99, trainable=True, \
																		name= "BatchNorm_" + str(i)+ "_" + str(j)
														)(layers[i+(j)*self.d], training=True)
														
						if activation_dense is "leaky_relu":
							strategyhelper = LeakyReLU(layers[i+(j)*self.d])
						else:
							strategyhelper = Activation(activation_dense)(layers[i+(j)*self.d])
					else:
						strategyhelper = Dense(1,
						   kernel_initializer=kernel_initializer,
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
					wealth = Subtract(name = "costDot_" + str(j))([tf.constant(inititial_wealth,shape=(1,)), costs])
				else:
					wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
				
				# Wealth for the next period
				# w_{t+1} = w_t + (strategy_t-strategy_{t+1})*prc_t
				#         = w_t - delta_strategy*prc_t
				mult = Dot(axes=1)([delta_strategy, prc])
				wealth = Subtract(name = "wealth_" + str(j))([wealth, mult])
 
				# Accumulate interest rate for next period.
				FV_factor = np.exp(self.risk_free/self.num_days_in_a_year)
				wealth = Lambda(lambda x: x*FV_factor)(wealth)
				
				prc = Input(shape=(1,),name = "prc_" + str(j+1))
				information_set = Input(shape=(1,), name = "information_set_" + str(j+1))
				
				strategy = strategyhelper    
				
				if j != self.N - 1:
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

				# The bias with weights for the kernel should be zero.
				if loss_type is "CVaR":
					# The bias with weights for the kernel should be zero.
					# If you run model.summary(), it will say there are two parameters.
					# But it's really just one because of the MaxNorm constraint.
					w = Dense(1, activation='linear', trainable= True,
							kernel_constraint=MaxNorm(0.0),
							kernel_initializer=Zeros(),
							bias_initializer=he_normal(),
							name = "certainty_equiv")(w)
					loss = Loss_Metrics(wealth,w).CVaR(kwargs["loss_param"])
				elif loss_type is "Entropy":
					w = Dense(1, activation='linear', trainable= False,
							kernel_initializer=Zeros(),
							bias_initializer=Zeros(),
							name = "certainty_equiv")(w)
				loss = Loss_Metrics(wealth,w).Entropy(kwargs["loss_param"])
				
				model = Model(inputs, outputs=[wealth, w])                    
				model.add_loss(loss)
		return model
