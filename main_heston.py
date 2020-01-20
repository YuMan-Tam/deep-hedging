"""
Reference:

https://people.math.ethz.ch/~jteichma/deep_portfolio_optimization_keras.html

Teichmann is one of the co-authors for the paper Deep Hedging (2019) by
Buehler et al. 

Creator: Yu-Man Tam
Last update: 1/7/2020

exec(open("main_heston.py").read())

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

from Stochastic_Processes import HestonProcess
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
epsilon = 0.0
                                                    
# Initial Conditions
S0 = 100.0
initial_wealth = 0.0

# Parameters for the simulated data.
num_asset = 2

v0 = 0.04
kappa = 1.0
theta = 0.04
sigma = 2
rho = -0.7
risk_free = 0.0
dividend = 0.0

stochastic_process = HestonProcess(s0 = S0, \
                                   v0 = v0, kappa = kappa, theta = theta, \
                                   sigma = sigma, rho = rho, \
                                   risk_free = risk_free, dividend = dividend)
                                   
use_saved_simulated_data = True # Whether to use saved simulated data to save speed.
use_saved_model_delta = True # Whether to use saved model_delta.
use_saved_model_vega = True # Whether to use saved model_vega.

is_fitting_mode = False # Whether to fit the model or only load existing weights.

# Produce figures (as referenced in Buehler et al 2019).
is_Figure_2 = False # Figure 2 - Model Hedge vs Deep Hedge 50% CVaR
is_Figure_3 = False # Figure 3 - Model and NN Delta
is_Figure_4 = False # Figure 4 - Comparison of Recurrent and Simpler NN (No Transaction Cost)
# is_Figure_5 = True # Figure 5 - Comparison of Recurrent and Simpler NN (With Transaction Cost and 99% CVaR
# is_Figure_6 = True # Figure 6 - Comparison of 99% CVaR and 50% CVaR

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
trade_set_str = "(S,Scaled_V)"
information_set = "(log_S,V)"
                          
# Parameters for the neural network.
m = 15 # Width
d = 4 # Depth (including input and output layer)
    
Ktrain = 5*(10**5) # Size of training sample
Ktest_ratio = 0.2 # Percentage of Ktrain reserved for out-of-sample test.
                
# Loss function parameter
CVar_alpha = 0.5
                
# Tuning parameter.
lr = 5e-3
optimizer = Adam(learning_rate=lr)

batch_size=1000
epochs=100

# Callback parameters.
subfix = strategy_type + "_" + str(CVar_alpha) + "_" + str(epsilon) + "_" + pos

model_str = "best_nn" + "_" + subfix
print("I am running model: {}".format(model_str))

best_model_file = "./data/" + model_str + ".h5"
early_stopping = EarlyStopping(monitor="loss", patience=5, min_delta=1e-4, restore_best_weights=True)
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
    V = np.load("./data/realized_variance.npy")
else:
    # Simulate training data.
    [S, V] = stochastic_process.gen_path(maturity, N, Ktrain)
    np.save("./data/stock.npy", S)
    np.save("./data/realized_variance.npy", V)    

VarSwap = stochastic_process.get_VarSwap_path(V, maturity)
log_S = np.log(S)

# Scaled_V is martingale.
t = np.linspace(0.0,maturity,N+1)
Scaled_V = np.zeros((Ktrain, N+1))

for i in range(0,N+1):
    factor = np.exp(-t[i])
    Scaled_V[:,i] = (V[:,i] - 0.04*(1.0-factor))/factor
    
# Payoff of the derivative to be hedged.
payoff_T = payoff_func(S[:,-1])

trade_set =  eval("np.stack(" + trade_set_str + ",axis=1)")
I =  eval("np.stack(" + information_set + ",axis=1)")

xtrain = [0.0*np.zeros(len(S))] # For the certainty equivalent w.
for i in range(N+1):
    xtrain += [trade_set[:,:,i]]
    if i != N:
        xtrain += [I[:,:,i]]
xtrain += [payoff_T]
    
# Note: Type of output is list rather than np.ndarray.
# Structure of xtest:
#   1) w (dim = 1)
#   2) Trade set: (S, Variance Swap)
#   2) Information set: [log_S, V] 
#   3) payoff (dim = 1)

[xtrain, xtest] = train_test_split(xtrain, test_size=Ktest_ratio)

# Splitting simulated paths for S and V (temp)
[greek_train, greek_test] = train_test_split([S]+[V], test_size=Ktest_ratio)

# Setup and compile the model.
model = Deep_Hedging(num_asset, N, d, m).model(strategy_type=strategy_type,\
                    is_trainable = True, is_training = True, \
                    epsilon = epsilon, initial_wealth = initial_wealth, \
                    loss_type="CVaR", alpha = CVar_alpha)
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
    pool = mp.ProcessPool(nodes = max(mp.cpu_count()-1,1))

    length_test_sample = int(Ktrain*Ktest_ratio)
    model_delta = np.zeros((length_test_sample, N+1))
    func_delta = lambda i, j: report.get_model_delta(instrument, s0=greek_test[0][i,j], v0=greek_test[1][i,j], calculation_date=calculation_date + j)
    for j in range(N+1):
        print("Calculating the sensitivity (delta) of date {}.".format(str(j)))
        tmp_model_delta = list(pool.map(func_delta, range(length_test_sample), repeat(j,length_test_sample)))
        model_delta[:,j] = tmp_model_delta
        np.save("./data/model_delta.npy", model_delta)

# Calculate model vega for each date on each of the simulated paths.
if use_saved_model_vega:
    model_vega = np.load("./data/model_vega.npy")
else:
    pool = mp.ProcessPool(nodes = max(mp.cpu_count()-1,1))

    length_test_sample = int(Ktrain*Ktest_ratio)
    model_vega = np.zeros((length_test_sample, N+1))
    func_vega = lambda i, j: report.get_model_vega(instrument,  s0= greek_test[0][i,j], v0=greek_test[1][i,j], calculation_date=calculation_date + j)
    for j in range(N+1):
        print("Calculating the sensitivity (vega) of date {}.".format(str(j)))
        tmp_model_vega = list(pool.map(func_vega, range(length_test_sample), repeat(j,length_test_sample)))
        model_vega[:,j] = tmp_model_vega
        np.save("./data/model_vega.npy", model_vega)
        
# Calculate model hedge
model_wealth = report.get_model_wealth(initial_wealth, model_delta, model_vega)

if is_Figure_2 and \
	os.path.exists("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5"):
    model = tf.keras.models.load_model("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5")
    [nn_wealth,w] = model.predict(xtest, batch_size=batch_size, verbose=0)
    
    # Plot Figure 2.
    x1 = reshape_1D(model_wealth) - model_price
    x2 = reshape_1D(nn_wealth) + model_price
    fig_2 = Plots.compare_model_vs_nn_hedge(x1=x1, x2=x2, range = (-3.0,3.0), bin = 25, subfix = "Fig2")
	
if is_Figure_3 and \
	os.path.exists("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5"):
    model = tf.keras.models.load_model("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5")	
	
    # Plot Figure 3.
    [nn_delta, model_delta, diff_delta] = Plots.compare_model_vs_nn_delta(model, \
                        report, instrument, calculation_date, \
                        S_range = np.linspace(96.0, 102.0, 31.0), V_range = np.linspace(0.04, 0.14, 51.0), days_from_today = 15, subfix = "Fig3")
						
if is_Figure_4 and \
	os.path.exists("./data/" + "best_nn_recurrent_0.5_0.0_short" + ".h5") and \
	os.path.exists("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5"):

    # Figure 4 of Buehler (2019)
    model_recurrent = tf.keras.models.load_model("./data/" + "best_nn_recurrent_0.5_0.0_short" + ".h5")
    [nn_wealth_recurrent,w_recurrent] = model_recurrent.predict(xtest, batch_size=batch_size, verbose=0)
    nn_loss_recurrent = model_recurrent.evaluate(xtest, batch_size=batch_size, verbose=0)

    model_simple = tf.keras.models.load_model("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5", verbose=0)
    [nn_wealth_simple,w_simple] = model_simple.predict(xtest, batch_size=batch_size)
    nn_loss_simple = model_simple.evaluate(xtest, batch_size=batch_size)

    x1 = nn_wealth_recurrent + nn_loss_recurrent
    x2 = nn_wealth_simple + nn_loss_simple
    fig_4 = Plots.compare_model_vs_nn_hedge(x1,x2, range = (-3.0,3.0), bin = 25, 
                                                subfix = "Fig4", label = ["Recurrent", "Simple"])
												
if is_Figure_5 and \
	os.path.exists("./data/" + "best_nn_recurrent_0.99_0.01_short" + ".h5") and \
	os.path.exists("./data/" + "best_nn_simple_0.99_0.01_short" + ".h5"):

    # Figure 5 of Buehler (2019)
    model_recurrent = tf.keras.models.load_model("./data/" + "best_nn_recurrent_0.99_0.01_short" + ".h5")
    [nn_wealth_recurrent,w_recurrent] = model_recurrent.predict(xtest, batch_size=batch_size, verbose=0)
    nn_loss_recurrent = model_recurrent.evaluate(xtest, batch_size=batch_size, verbose=0)
    
    model_simple = tf.keras.models.load_model("./data/" + "best_nn_simple_0.99_0.01_short" + ".h5")
    [nn_wealth_simple,w_simple] = model_simple.predict(xtest, batch_size=batch_size, verbose=0)
    nn_loss_simple = model_simple.evaluate(xtest, batch_size=batch_size, verbose=0)

    x1 = nn_wealth_recurrent + nn_loss_recurrent
    x2 = nn_wealth_simple + nn_loss_simple    
    fig_5 = Plots.compare_model_vs_nn_hedge(x1, x2, range = (-1.0,6.0), bin = 25, 
                                                subfix = "Fig5", label = ["Recurrent", "Simple"])  

if is_Figure_6 and \
	os.path.exists("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5") and \
	os.path.exists("./data/" + "best_nn_simple_0.99_0.0_short" + ".h5"):
	
    # Figure 6 of Buehler (2019)
    model_CVaR_low = tf.keras.models.load_model("./data/" + "best_nn_simple_0.5_0.0_short" + ".h5")
    [nn_wealth_CVaR_low ,w_CVaR_low ] = model_CVaR_low .predict(xtest, batch_size=batch_size, verbose=0)
    nn_loss_CVaR_low  = model_CVaR_low.evaluate(xtest, batch_size=batch_size, verbose=0)
    
    model_CVaR_high = tf.keras.models.load_model("./data/" + "best_nn_simple_0.99_0.0_short" + ".h5")
    [nn_wealth_CVaR_high,w_CVaR_high] = model_CVaR_high.predict(xtest, batch_size=batch_size, verbose=0)
    nn_loss_CVaR_high = model_CVaR_high.evaluate(xtest, batch_size=batch_size, verbose=0)

    x1 = nn_wealth_CVaR_low + nn_loss_CVaR_low
    x2 = nn_wealth_CVaR_high + nn_loss_CVaR_high    
    fig_6 = Plots.compare_model_vs_nn_hedge(x1, x2, range = (-3.0,3.0), bin = 25, 
                                                subfix = "Fig6", label = ["CVar - 0.5", "CVar - 0.99"])

                                                


