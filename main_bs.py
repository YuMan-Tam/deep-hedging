"""
Reference:

https://people.math.ethz.ch/~jteichma/deep_portfolio_optimization_keras.html

Teichmann is one of the co-authors for the paper Deep Hedging (2019) by
Buehler et al. 

Creator: Yu-Man Tam
Last update: 1/8/2020

exec(open("main_bs.py").read())

May need:

import importlib
importlib.reload()

"""

# Ignore numpy warnings (version compatibility issue).
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.getcwd() + "/lib")

import time
import pathos.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql
import tensorflow as tf

from itertools import repeat
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
                                            ReduceLROnPlateau, TensorBoard
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.keras.models import Model

from Stochastic_Processes import HestonProcess, BlackScholesProcess
from Deep_Hedging import Deep_Hedging
from Utilities import train_test_split, reshape_1D
from Validation import Validation
import Plots
 
# Configure Tensorflow
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Parameters for the instrument to be priced.
N = 30 # Time step (in days)
maturity = N/365 # As fraction of a year.

# Parameters for the transaction cost.
epsilon = 0.001
                                                    
# Initial Conditions
S0 = 100.0
initial_wealth = 0.0

# Parameters for the simulated data.
num_asset = 1

sigma = 0.2
risk_free = 0.0
dividend = 0.0

stochastic_process = BlackScholesProcess(s0 = S0, \
                                   sigma = sigma, \
                                   risk_free = risk_free, \
								   dividend = dividend)
                                   
use_saved_simulated_data = True # Whether to use saved simulated data to save speed.
use_saved_model_delta = True # Whether to use saved model_delta.
use_saved_model_vega = True # Whether to use saved model_vega.

is_fitting_mode = True # Whether to fit the model or only load existing weights.

# Produce figures (as referenced in Buehler et al 2019).
is_Figure_2 = False # Figure 2 - Model Hedge vs Deep Hedge 50% CVaR
is_Figure_3 = True # Figure 3 - Model and NN Delta

# Strategy Type: recurrent, simple
strategy_type = "simple"

# Terminal payoff of derivatives.
pos = "short"

# ATM call (recall date format is MM/DD/YYYY in UK format).
pos_dict = {"short": -1, "long": 1}

strike=S0
payoff_func = lambda x: pos_dict[pos]*np.maximum(x - strike, 0.0)
calculation_date = ql.Date.todaysDate()
maturity_date = ql.Date.todaysDate() + N

# Information set (in string)
# Choose from: S, V, VarSwap, log_S, Scaled_V
trade_set_str = "(S)"
information_set = "(S)"
                          
# Parameters for the neural network.
m = 15 # Width
d = 4 # Depth (including input and output layer)
    
# Ktrain = 5*(10**5) # Size of training sample
Ktrain = int(1*(10**5)/0.8) # Size of training sample
Ktest_ratio = 0.2 # Percentage of Ktrain reserved for out-of-sample test.
                
# Loss function parameter
# loss_type = "CVaR" -> loss_param = alpha
# loss_type = "Entropy" -> loss_param = lambda
loss_type = "CVaR"
loss_param = 0.5
                
# Tuning parameter.
lr = 5e-3
optimizer = Adam(learning_rate=lr)

batch_size=256
epochs=100

# Callback parameters.
subfix = strategy_type + "_" + str(loss_type) + "_" + str(loss_param)+ "_" + str(epsilon) + "_" + pos + \
			"_" + str(risk_free) + "_" + str(strike) + "_" + str(N)

model_str = "best_nn" + "_" + subfix
print("I am running model: {}".format(model_str))

best_model_file = "./data/" + model_str + ".h5"
early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=1e-4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.9, patience=2, min_delta=1e-3, verbose=1)

model_checkpoint = ModelCheckpoint(best_model_file, monitor="loss", \
                                        save_best_only=True, save_weights_only=False, verbose=1)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

"""
*****************************************
Buehler (2019)'s deep hedging paper.
*****************************************
"""

if use_saved_simulated_data:
    # Use saved simulated data.
    S = np.load("./data/stock.npy")
else:
    # Simulate training data.
    S= stochastic_process.gen_path(maturity, N, Ktrain)
    np.save("./data/stock.npy", S) 

log_S = np.log(S)
    
# Payoff of the derivative to be hedged.
payoff_T = payoff_func(S[:,-1])

trade_set =  eval("np.stack(" + trade_set_str + ",axis=1)")
I =  eval("np.stack(" + information_set + ",axis=1)")

xtrain = [0.0*np.zeros(len(S))] # For the certainty equivalent w.
for i in range(N+1):
    xtrain += [trade_set[i,:]]
    if i != N:
        xtrain += [I[i,:]]
xtrain += [payoff_T]
    
# Note: Type of output is list rather than np.ndarray.
# Structure of xtest:
#   1) w (dim = 1)
#   2) Trade set: (S, Variance Swap)
#   2) Information set: [log_S, V] 
#   3) payoff (dim = 1)

[xtrain, xtest] = train_test_split(xtrain, test_size=Ktest_ratio)

# Splitting simulated paths for S and V (temp)
[greek_train, greek_test] = train_test_split([S], test_size=Ktest_ratio)

# Setup and compile the model.
model = Deep_Hedging(num_asset, N, d, m, risk_free, maturity).model(strategy_type=strategy_type,\
                    is_trainable = True, is_training = True, \
                    epsilon = epsilon, initial_wealth = initial_wealth, \
                    loss_type=loss_type, loss_param = loss_param)
model.compile(optimizer=optimizer)

if is_fitting_mode:
    start = time.time()
    
    # Try to load weights if possible.
    try:
        model.load_weights(best_model_file)
        print("Load model successfully! Training resume!")
    except:
        pass
        
    # Fit with training data and then predict with test data for rho(-Z).
    model.fit(x=xtrain, \
                batch_size=batch_size, epochs=epochs, \
                validation_data=(xtest, np.empty(0)), \
                callbacks=callbacks, \
                verbose=True)
    end = time.time()
    model.save(best_model_file)
    
    print("Running time is {} hours.".format(str((end-start)/3600.0)))
else:
    model = tf.keras.models.load_model(best_model_file)
    
# Evaluate the model out-of-sample.
[nn_wealth,w] = model.predict(xtest, batch_size=batch_size)
nn_loss = model.evaluate(xtest, batch_size=batch_size, verbose=0)

print("The mean of terminal wealth is: {0}.".format(np.mean(nn_wealth)))
print("Final loss is: {0}".format(nn_loss))

# Validation
report = Validation(model = model, data = xtest, \
                            process = stochastic_process)
                                
"""
Calculate model price for different instruments.

European_Call: strike, maturity_date, exercise_date, stochastic_process
"""
instrument = report.get_instrument(name = "European_Call", \
                                    strike = strike, \
                                    maturity_date = maturity_date)

model_price = report.get_model_PV(instrument)
risk_neutral_price = report.get_risk_neutral_PV()

# Calculate model delta for each date on each of the simulated paths.
if use_saved_model_delta:
    model_delta = np.load("./data/model_delta.npy")
else:
    length_test_sample = int(Ktrain*Ktest_ratio)
    model_delta = np.zeros((length_test_sample, N+1))
    func_delta = lambda i, j: report.get_model_delta(instrument, s0=greek_test[0][i,j], calculation_date=calculation_date + j)
    for j in range(N+1):
        print("Calculating the sensitivity (delta) of date {}.".format(str(j)))
        tmp_model_delta = list(map(func_delta, range(length_test_sample), repeat(j,length_test_sample)))
        model_delta[:,j] = tmp_model_delta
        np.save("./data/model_delta.npy", model_delta)
    
# Calculate model hedge
model_wealth = report.get_model_wealth(initial_wealth, model_delta)

if is_Figure_2 and \
	os.path.exists("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5"):
	model = tf.keras.models.load_model("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5")
	[nn_wealth,w] = model.predict(xtest, batch_size=batch_size, verbose=0)
	
	# Plot Figure 2.
	x1 = reshape_1D(model_wealth) - model_price
	x2 = reshape_1D(nn_wealth) + model_price
	
	fig_2 = Plots.compare_model_vs_nn_hedge(x1=x1, x2=x2, range = (-3.0,3.0), bin = 25, subfix = "Fig2")

if is_Figure_3 and \
	os.path.exists("./data/" + model_str + ".h5"):
	model = tf.keras.models.load_model("./data/" + model_str + ".h5")	
	
	figure_dir = "./figure/robustness/" + model_str
	try:
		os.system("mkdir " + figure_dir)
	except:
		pass

	for i in range(30):
		days_from_today = i
		
		# Plot two standard deviation from mean.
		S_range = np.sort(greek_test[0][:,i])
		S_range = S_range[np.append(np.arange(0, len(S_range), int(len(S_range)/100)), len(S_range)-1)]
		
		# Plot Figure 3.
		[nn_delta, model_delta, diff_delta] = Plots.compare_model_vs_nn_delta(model, \
							report, instrument, calculation_date, \
							S = S_range, days_from_today = i , subfix = "Fig3", \
							fig_dir = figure_dir +"/")
