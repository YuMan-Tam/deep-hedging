# importlib.reload(Plots)
import itertools
import numpy as np
import matplotlib as plt

from matplotlib import figure, cm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Model
from Utilities import reshape_1D

# Plot Figure 2 on P.24 of Buehler (2019).
def compare_model_vs_nn_hedge(x1 = None, x2 = None, \
								range = None, bin = None, \
								fig_name = "Buehler_2019", subfix = None, ext = "png", \
								label = None):
	fig_name = fig_name + "_" + subfix

	fig = figure.Figure()
	ax = fig.gca()
	
	# Plot histogram
	if label is None:
		label=["Model Hedge", "Network Hedge"]
		
	ax.hist((reshape_1D(x1), reshape_1D(x2)), bins=bin, range = range, label=label)
	ax.legend(loc="upper left")
	
	fig.savefig(fig_dir + fig_name + "." + ext)
	
	return fig

# Plot Figure 3 on P.24 of Buehler (2019).
def compare_model_vs_nn_delta(model = None, report = None, instrument = None, calculation_date = None, \
								S = None, V_range = None, days_from_today = None, \
								fig_name = "Buehler_2019", subfix = None, ext = "png", \
								fig_dir = None):
	fig_name = fig_name + "_" + subfix

	fig = figure.Figure()
	
	# Model delta
	model_delta = np.zeros(S.shape)
	for i in range(S.shape[0]):
		model_delta[i] = report.get_model_delta(instrument, s0= S[i], v0=None, \
			calculation_date=calculation_date + days_from_today)
	
	# Plot network delta.
	# Find out the number of hidden layers in the network. One can also just pass the parameter d into the function.
	d = 0
	try:
		while True:
			model.get_layer("dense_" + str(d) + "_" + str(days_from_today))
			d += 1
	except:
		pass

	nn_strategy_model = Model(model.get_layer("dense_0_" + str(days_from_today)).input, model.get_layer("dense_" + str(d-1) \
						+ "_" + str(days_from_today)).output)
	
	nn_delta = reshape_1D(nn_strategy_model.predict(S.flatten(), batch_size=1000,verbose=1))
	
	# Plot Overlap delta.
	ax_diff_delta = fig.gca(projection=None)
	
	surf = ax_diff_delta.plot(S, model_delta)
	surf = ax_diff_delta.scatter(S, nn_delta, c="red", s=2)
	
	fig.savefig(fig_dir + fig_name +"_overlap_delta" + "_" + str(days_from_today) + "." + ext)
	fig.clf()
	
	# Plot differences between network delta and model delta.
	ax_diff_delta = fig.gca(projection=None)
	
	diff_delta = nn_delta - model_delta
	surf = ax_diff_delta.plot(S,diff_delta)
	fig.savefig(fig_dir + fig_name +"_diff_delta" + "_" + str(days_from_today) + "." + ext)
	fig.clf()
	
	return nn_delta, model_delta, diff_delta
