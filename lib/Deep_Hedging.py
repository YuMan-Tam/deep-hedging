from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
							Lambda, Add, Dot, \
							BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import he_normal, Zeros
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from Loss_Metrics import Loss_Metrics

seed = None
class Deep_Hedging:
	def __init__ (self, num_asset = 1, N = None, d = None, m = None, \
					risk_free = None, maturity = None):
		self.num_asset = num_asset
		self.N = N
		self.d = d
		self.m = m
		self.maturity = maturity
		self.risk_free = risk_free
		
		self.day_count_in_a_year = N/maturity
		
	def model(self, strategy_type = None, \
					is_trainable=True, is_training = True, \
					epsilon = 0.0, initial_wealth = 0.0, \
					loss_type = None, **kwargs):
		# The state variables.
		prc = Input(shape=(self.num_asset,), name = "prc_0")
		information_set = Input(shape=(self.num_asset,), name = "information_set_0")
		
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
							   kernel_initializer=he_normal(),
							   use_bias=False, 
							   name = "dense_" + str(i)+ "_" + str(j))(helper1)
							   
						# Batch normalization.                       
						layers[i+(j)*self.d] = BatchNormalization(momentum = 0.99, \
																	name= "BatchNorm_" + str(i)+ "_" + str(j), \
																	trainable=is_trainable \
													)(layers[i+(j)*self.d], training=is_training)
						strategyhelper = LeakyReLU(name= "Activation_" + str(i)+ "_" + str(j),)(layers[i+(j)*self.d])               
					elif i != self.d-1:
						layers[i+(j)*self.d] = Dense(self.m,
							   kernel_initializer=he_normal(),
							   use_bias=False,
							   name = "dense_" + str(i)+ "_" + str(j))(strategyhelper)
						
						# Batch normalization                        
						layers[i+(j)*self.d] = BatchNormalization(momentum = 0.99, \
																	name= "BatchNorm_" + str(i)+ "_" + str(j), \
																	trainable=is_trainable \
													)(layers[i+(j)*self.d], training=is_training)
						strategyhelper = LeakyReLU(name= "Activation_" + str(i)+ "_" + str(j))(layers[i+(j)*self.d])
					else:
						strategyhelper = Dense(self.num_asset,
						   activation="linear",
						   kernel_initializer=he_normal(),
						   use_bias=True, 
						   name = "dense_" + str(i)+ "_" + str(j))(strategyhelper)
						   
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
					wealth = Subtract(name = "costDot_" + str(j))([tf.constant(initial_wealth,shape=(1,)), costs])
				else:
					wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
				
				# Wealth for the next period
				# w_{t+1} = w_t + (strategy_t-strategy_{t+1})*prc_t
				#         = w_t - delta_strategy*prc_t
				mult = Dot(axes=1)([delta_strategy, prc])
				wealth = Subtract(name = "wealth_" + str(j))([wealth, mult])
 
				# Accumulate interest rate for next period.
				FV_factor = np.exp(self.risk_free/self.day_count_in_a_year)
				wealth = Lambda(lambda x: x*FV_factor)(wealth)
				
				prc = Input(shape=(self.num_asset,),name = "prc_" + str(j+1))
				information_set = Input(shape=(self.num_asset,), name = "information_set_" + str(j+1))
				
				strategy = strategyhelper    
				
				if j != self.N - 1:
					inputs += [prc, information_set]
				else:
					inputs += [prc]
			else:
				# The paper assumes no transaction costs for the final period 
				# when the position is liquidated.
				final_period_cost = True
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
