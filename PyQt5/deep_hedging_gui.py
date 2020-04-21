# Commented out IPython magic to ensure Python compatibility.
#@title <font color='Blue'>**Overheads**</font>

# Author: Yu-Man Tam
# Last updated: 4/5/2020

# Reference: Deep Hedging (2019, Quantitative Finance) by Buehler et al.
# https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1571683

# Qt references: https://doc.qt.io/qt-5/qmainwindow.html

import sys, os
sys.path.insert(0, os.getcwd() + "/../lib")

import time

# Linear algebra, finance, and machine learning libraries
import numpy as np
import QuantLib as ql
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from scipy.stats import norm

# For PyQtgraph
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import ParameterTree, Parameter

# User-defined libraries
from stochastic_processes import BlackScholesProcess
from instruments import European_Call
from deep_hedging import Deep_Hedging_Model
from loss_metrics import Entropy, CVaR
from utilities import train_test_split
from default_params import Deep_Hedging_Params

# Tensorflow settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

# PyQtGraph Settings
pg.setConfigOptions(antialias=False)

# Default Parameters
# European call option (short).
calculation_date = ql.Date.todaysDate()

# Day convention.
day_count = ql.Actual365Fixed() # Actual/Actual (ISDA)         

# Information set (in string)
# Choose from: S, log_S, normalized_log_S (by S0)
information_set = "normalized_log_S"

# Loss function
# loss_type = "CVaR" (Expected Shortfall) -> loss_param = alpha 
# loss_type = "Entropy" -> loss_param = lambda
loss_type = "Entropy"

# Other NN parameters
use_batch_norm = False
kernel_initializer = "he_uniform"

activation_dense = "leaky_relu"
activation_output = "sigmoid"
final_period_cost = False

# Reducing learning rate
reduce_lr_param = {"patience" : 2, "min_delta" : 1e-3, "factor" : 0.5}

# Early stopping
early_stopping_param = {"patience": 10, "min_delta": 1e-4}

# Number of bins to plot for the PnL histograms.
num_bins = 30

# Need a separate threads for deep hedging algo and plot the graphs.
class DH_Worker(QtCore.QThread):
  DH_outputs = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.float32, float, float, bool)
  def __init__(self):
    QtCore.QThread.__init__(self)
      
  def __del__(self):
    self.wait()
      
  def run_deep_hedge_algo(self, training_dataset = None, epochs = None, Ktrain = None, batch_size = None, \
                              model = None, submodel = None, strategy_type = None, loss_param = None, learning_rate = None, xtest = None, xtrain= None, \
                              initial_price_BS = None, width = None, I_range = None, x_range = None):
    self.training_dataset = training_dataset
    self.Ktrain = Ktrain
    self.batch_size = batch_size
    self.model = model
    self.submodel = submodel
    self.loss_param = loss_param
    self.initial_price_BS = initial_price_BS
    self.width = width
    self.epochs = epochs
    self.xtest = xtest
    self.xtrain = xtrain
    self.I_range = I_range
    self.x_range = x_range
    self.strategy_type = strategy_type
    self.learning_rate = learning_rate

    self.start()
      
  def pause(self):
    self._pause = True
      
  def cont(self):
    self._pause = False
      
  def stop(self):
    self._exit = True
    self.exit()
      
  def is_running(self):
    if self._pause or self._exit:
        return False
    else:
        return True

  def Reduce_Learning_Rate(self, num_epoch, loss):
    # Extract in-sample loss from the previous epoch. Comparison starts in epoch 2 and the index for epoch 1 is 0 -> -2
    min_loss = self.loss_record[:,1].min()
    if min_loss - loss < reduce_lr_param["min_delta"]:
        self.reduce_lr_counter += 1

    if self.reduce_lr_counter > reduce_lr_param["patience"]:
        self.learning_rate = self.learning_rate * reduce_lr_param["factor"]
        self.optimizer.learning_rate = self.learning_rate
        print("The learning rate is reduced to {}.".format(self.learning_rate))
        self.reduce_lr_counter = 0

  def Early_Stopping(self, num_epoch, loss):
    min_loss = self.loss_record[:,1].min()
    if min_loss - loss < early_stopping_param["min_delta"]:
        self.early_stopping_counter +=1
      
  def run(self):
    # Initialize pause and stop buttons.
    self._exit = False
    self._pause = False

    # Variables to control skipped frames. If the DH algo output much faster than the graphic output, the plots can be jammed.
    self.Figure_IsUpdated = True

    self.reduce_lr_counter = 0
    self.early_stopping_counter = 0

    certainty_equiv = tf.Variable(0.0, name = "certainty_equiv")
    
    # Accelerator Function.
    model_func = tf.function(self.model)
    submodel_func = tf.function(self.submodel)
    
    self.optimizer = Adam(learning_rate=self.learning_rate)
    
    num_epoch = 0
    while num_epoch <= self.epochs:
      # Exit event loop if the exit flag is set to True.
      if self._exit:
        mini_batch_iter = None
        self._exit = False
        self._pause = False
        break

      if not self._pause:
        try:
          mini_batch = mini_batch_iter.next()
        except:
          # Reduce learning rates and Early Stopping are based on in-sample losses calculated once per epoch.
          in_sample_wealth = model_func(self.xtrain)
          in_sample_loss = Entropy(in_sample_wealth,certainty_equiv,self.loss_param)

          if num_epoch >= 1:
            print("The deep-hedging price is {:0.4f} after {} epoch.".format(oos_loss, num_epoch))

            # Programming hack. The deep-hedging algo computes faster than the computer can plot, so there could be
            # missing frames, i.e. there is no guarantee that every batch is plotted. Here, I force a signal to be
            # emitted at the end of an epoch.
            time.sleep(1)

            self.DH_outputs.emit(PnL_DH, DH_delta, DH_bins, oos_loss.numpy().squeeze(), \
                                                          num_epoch, num_batch, True)

            # This is needed to prevent the output signals from emitting faster than the system can plot a graph.
            # The performance is much better than emitting at fixed time intervals.
            self.Figure_IsUpdated = False

          if num_epoch == 1:
            self.loss_record = np.array([num_epoch, in_sample_loss],ndmin=2)
          elif num_epoch > 1:
            self.Reduce_Learning_Rate(num_epoch, in_sample_loss)
            self.loss_record = np.vstack([self.loss_record, np.array([num_epoch,in_sample_loss])])
            
            # self.Early_Stopping(num_epoch,in_sample_loss)
            if self.early_stopping_counter > early_stopping_param["patience"]:
                print(early_stopping_param["patience"])
                print("The difference in losses are less than {} in {} consecutive epochs. We achieved convergence.".format(early_stopping_param["min_delta"], self.early_stopping_counter))
                break

          mini_batch_iter = self.training_dataset.shuffle(self.Ktrain).batch(self.batch_size).__iter__()
          mini_batch = mini_batch_iter.next()

          num_batch = 0
          num_epoch += 1

        num_batch += 1
        
        # Record gradient
        with tf.GradientTape() as tape:
          wealth = model_func(mini_batch)
          loss = Entropy(wealth, certainty_equiv, self.loss_param)

        oos_wealth = model_func(self.xtest)
        PnL_DH = oos_wealth.numpy().squeeze() # Out-of-sample

        submodel_delta_range = np.expand_dims(self.I_range,axis=1)
        if self.strategy_type == "simple":
            submodel_inputs = submodel_delta_range
        elif self.strategy_type == "recurrent":
            # Assume previous delta is ATM.
            submodel_inputs = [submodel_delta_range, np.ones_like(submodel_delta_range)*0.5]
        
        DH_delta = submodel_func(submodel_inputs).numpy().squeeze()
        DH_bins, _ = np.histogram(PnL_DH+self.initial_price_BS, bins = num_bins, range = self.x_range)
        
        # Forward and backward passes
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Compute Out-of-Sample Loss
        oos_loss =  Entropy(oos_wealth, certainty_equiv, self.loss_param)

        if self.Figure_IsUpdated:
          self.DH_outputs.emit(PnL_DH, DH_delta, DH_bins, oos_loss.numpy().squeeze(), \
                                                          num_epoch, num_batch, False)
          
          # This is needed to prevent the output signals from emitting faster than the system can plot a graph.
          # The performance is much better than emitting at fixed time intervals.
          self.Figure_IsUpdated = False

        # Mandatory pause for the first iteration to explain demo.
        if num_epoch == 1 and num_batch == 1:
          self.pause()
      else:
          time.sleep(1)

class MainWindow(QtWidgets.QMainWindow):
  def __init__(self):
    # Inheritance from the QMainWindow class
    # Reference: https://doc.qt.io/qt-5/qmainwindow.html
    super().__init__()
    self.days_from_today = 15
    self.Thread_RunDH = DH_Worker()
    
    # The order of code is important here: Make sure the emitted signals are connected
    # before actually running the Worker.
    self.Thread_RunDH.DH_outputs["PyQt_PyObject", "PyQt_PyObject", "PyQt_PyObject", \
            "PyQt_PyObject", "double", "double", "bool"].connect(self.Update_Plots_Widget)
            
    # Define a top-level widget to hold everything
    self.w = QtGui.QWidget()
    
    # Create a grid layout to manage the widgets size and position
    self.layout = QtGui.QGridLayout()
    self.w.setLayout(self.layout)
    
    self.setCentralWidget(self.w)

    # Add the parameter menu.
    self.tree_height = 5 # Must be Odd number.
    
    self.tree = self.Deep_Hedging_Parameter_Widget()
    self.layout.addWidget(self.tree, 0, 0, self.tree_height, 2)   # upper-left

    self.tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    self.tree.setMinimumSize(350,650)
    
    # Add a run button
    self.run_btn = QtGui.QPushButton('Run')
    self.layout.addWidget(self.run_btn, self.tree_height+1, 0, 1, 1)   # button goes in upper-left
    
    # Add a pause button
    self.pause_btn = QtGui.QPushButton('Pause')
    self.layout.addWidget(self.pause_btn, self.tree_height+1, 1, 1, 1)   # button goes in upper-left
    
    # Run the deep hedging algo in a separate thread when the run button is clicked.
    self.run_btn.clicked.connect(self.RunButton)
    
    # Pause button.
    self.pause_btn.clicked.connect(self.Pause)

  def Deep_Hedging_Parameter_Widget(self):
    tree = ParameterTree()
    
    ## Create tree of Parameter objects
    self.params = Parameter.create(name='params', type='group', children=Deep_Hedging_Params())
    tree.setParameters(self.params, showTop=False)

    return tree
  
  # Define the event when the "Run" button is clicked.
  def RunButton(self):
    if self.run_btn.text() == "Stop":
      self.Thread_RunDH.stop()
      if self.pause_btn.text() == "Continue":
        self.pause_btn.setText("Pause")
      self.run_btn.setText("Run")
    elif self.run_btn.text() == "Run":
      self.run_btn.setText("Stop")
      
      # Set parameters
      self.Ktrain = self.params.param("Monte-Carlo Simulation", 'Sample Size', "Training").value()
      self.Ktest_ratio = self.params.param("Monte-Carlo Simulation", 'Sample Size', "Testing (as fraction of Training)").value()
      self.N = self.params.param("European Call Option", "Maturity (in days)").value()
      self.S0 = self.params.param("European Call Option", "S0").value()
      self.strike = self.params.param("European Call Option", "Strike").value()
      self.sigma = self.params.param("European Call Option", "Implied Volatility").value()
      self.risk_free = self.params.param("European Call Option", "Risk-Free Rate").value()
      self.dividend = self.params.param("European Call Option", "Dividend Yield").value()
      
      self.loss_param = self.params.param("Deep Hedging Strategy", 'Loss Function (Exponential)', "Risk Aversion").value()
      self.epsilon = self.params.param("European Call Option", "Proportional Transaction Cost", "Cost").value()
      self.d = self.params.param("Deep Hedging Strategy", "Network Structure", "Number of Hidden Layers").value()
      self.m = self.params.param("Deep Hedging Strategy", "Network Structure", "Number of Neurons").value()
      self.strategy_type = self.params.param("Deep Hedging Strategy", "Network Structure", "Network Type").value()
      self.lr = self.params.param("Deep Hedging Strategy", "Learning Parameters", "Learning Rate").value()
      self.batch_size = self.params.param("Deep Hedging Strategy", "Learning Parameters", "Mini-Batch Size").value()
      self.epochs = self.params.param("Deep Hedging Strategy", "Learning Parameters", "Number of Epochs").value()
      
      self.maturity_date = calculation_date + self.N
      self.payoff_func = lambda x: -np.maximum(x - self.strike, 0.0)
      
      # Simulate the stock price process.
      self.S = self.simulate_stock_prices()
      
      # Assemble the dataset for training and testing.
      # Structure of data:
      #   1) Trade set: [S]
      #   2) Information set: [S] 
      #   3) payoff (dim = 1)
      self.training_dataset = self.assemble_data()

      # Compute Black-Scholes prices for benchmarking.
      self.price_BS, self.delta_BS, self.PnL_BS = self.get_Black_Scholes_Prices()

      # Compute the loss value for Black-Scholes PnL
      self.loss_BS = Entropy(self.PnL_BS,tf.Variable(0.0),self.loss_param).numpy()

      # Define model and sub-models
      self.model = self.Define_DH_model()
      self.submodel = self.Define_DH_Delta_Strategy_Model()
      
      plot_height_split = (self.tree_height+1)/2
      # Add the PnL histogram (PlotWidget) - Black-Scholes vs Deep Hedging.
      self.fig_PnL = self.PnL_Hist_Widget()
      self.layout.addWidget(self.fig_PnL, 0, 3, plot_height_split, 1) 
      self.fig_PnL.setMinimumWidth(600)
      
      # Add the Delta line plot (PlotWidget) - Black-Scholes vs Deep Hedging.
      self.fig_delta = self.Delta_Plot_Widget()
      self.layout.addWidget(self.fig_delta, 0, 4, plot_height_split , 1)
      self.fig_delta.setMinimumWidth(600)
      
      # Add the loss plot (PlotWidget) - Black-Scholes vs Deep Hedging.
      self.fig_loss = self.Loss_Plot_Widget()
      self.layout.addWidget(self.fig_loss, plot_height_split, 3, plot_height_split+1, 2)
      self.fig_loss.setMinimumWidth(1200)
      
      # Run the deep hedging algo in a separate thread.
      self.Thread_RunDH.run_deep_hedge_algo(training_dataset = self.training_dataset, epochs = self.epochs, \
                              Ktrain = self.Ktrain, batch_size = self.batch_size, model = self.model, \
                              submodel = self.submodel, strategy_type = self.strategy_type, loss_param = self.loss_param, learning_rate = self.lr, \
                              xtest = self.xtest, xtrain = self.xtrain, \
                              initial_price_BS = self.price_BS[0][0], width = self.width, I_range = self.I_range, x_range = self.x_range)

  # Define action when the Pause button is clicked.
  def Pause(self):
    if self.pause_btn.text() == "Pause":
      self.Thread_RunDH.pause()
      self.pause_btn.setText("Continue")
    elif self.pause_btn.text() == "Continue":
      self.Thread_RunDH.cont()
      self.pause_btn.setText("Pause")
  
  # Define deep hedging model
  def Define_DH_model(self):
    # Setup and compile the model
    model = Deep_Hedging_Model(N=self.N, d=self.d+2, m=self.m, risk_free=self.risk_free, \
                                                dt = self.dt, strategy_type=self.strategy_type, epsilon = self.epsilon, \
                                                use_batch_norm = use_batch_norm, kernel_initializer = kernel_initializer, \
                                                activation_dense = activation_dense, activation_output = activation_output, \
                                                final_period_cost = final_period_cost, delta_constraint = None)
    
    return model
  
  def Define_DH_Delta_Strategy_Model(self):
    if self.strategy_type == "simple":
      # Set up the sub-model that outputs the delta.
      submodel = Model(self.model.get_layer("delta_" + \
                          str(self.days_from_today)).input, self.model.get_layer("delta_" + \
                          str(self.days_from_today)).output)
    elif self.strategy_type == "recurrent":
      # For "recurrent", the information set is price as well as the past delta.
      inputs = [Input(1,), Input(1,)]
      
      intermediate_inputs = Concatenate()(inputs)
     
      outputs = self.model.get_layer("delta_" + str(self.days_from_today))(intermediate_inputs)
              
      submodel = Model(inputs=inputs, outputs=outputs)

    return submodel
      
  # Draw PnL histogram (PlotWidget) - Black-Scholes vs Deep Hedging.
  def PnL_Hist_Widget(self):      
    # Initialize the PnL Histogram Widget.
    fig_PnL = pg.PlotWidget()

    x_min = np.minimum(self.PnL_BS.min()+self.price_BS[0,0], -3)
    x_max = np.maximum(self.PnL_BS.max()+self.price_BS[0,0], 3)

    self.x_range = (x_min,x_max)
    self.BS_bins, self.bin_edges = np.histogram(self.PnL_BS+self.price_BS[0,0], bins = num_bins, range = self.x_range)
    self.width = (self.bin_edges[1] - self.bin_edges[0])/2.0

    self.BS_hist = pg.BarGraphItem(x=self.bin_edges[:-2], height=self.BS_bins, width=self.width, brush='r', \
            name = "Red - Black-Scholes", antialias = False)
    
    fig_PnL.setTitle("<font size='5'>Profit and Loss (PnL) Histogram</font>")
    fig_PnL.setLabels(left="<font size='4'>Frequency</font>", bottom="<font size='4'>Profit and Loss (PnL) </font>")

    fig_PnL_text = \
        pg.TextItem(html="<div align='center'><span style='color: rgb(255,0,0);'>Black-Scholes PnL (Benchmark)</span><br><span style='color: rgb(0,0,255); ;'>Deep-Hedging PnL </span></div>", \
        anchor=(0,0), angle=0, border='w', fill=(225, 225, 200))
    fig_PnL_text.setPos(self.bin_edges.min()*1.35,self.BS_bins.max()*1.05)

    # Fix the problem that Y-axes keep moving when transactioni cost is greater than zero.
    fig_PnL.setYRange(0,self.BS_bins.max()*1.1)
    
    fig_PnL.addItem(self.BS_hist)
    fig_PnL.addItem(fig_PnL_text)

    return fig_PnL
          
  # Draw Delta plot (PlotWidget) - Black-Scholes vs Deep Hedging.
  # Assume the PnL_Hist_Widget ran first, so we don't need to run the model again.
  def Delta_Plot_Widget(self):
    self.tau = (self.N-self.days_from_today)*self.dt
    
    self.min_S = self.S_test[0][:,self.days_from_today].min()
    self.max_S = self.S_test[0][:,self.days_from_today].max()
    self.S_range = np.linspace(self.min_S,self.max_S,51)

    # Attention: Need to transform it to be consistent with the information set.
    if information_set == "S":
      self.I_range =  self.S_range # Information set
    elif information_set == "log_S":
      self.I_range =  np.log(self.S_range)
    elif information_set == "normalized_log_S":
      self.I_range =  np.log(self.S_range/self.S0)        
        
    # Compute Black-Scholes delta for S_range.
    # Reference: https://en.wikipedia.org/wiki/Greeks_(finance)
    self.d1 = (np.log(self.S_range) - np.log(self.strike) + (self.risk_free - self.dividend + (self.sigma**2)/2)*self.tau) \
                / (self.sigma*np.sqrt(self.tau))  
                
    self.model_delta = norm.cdf(self.d1)*np.exp(-self.dividend*self.tau)
    
    fig_delta = pg.PlotWidget()
    
    self.BS_delta_plot = pg.PlotCurveItem(pen = pg.mkPen(color="r", width=2.5), name = "Black-Scholes")
    self.BS_delta_plot.setData(self.S_range, self.model_delta)
    
    fig_delta.setTitle("<font size='5'> Hedging Strategy: Delta (at t = 15 days)</font>")
    fig_delta.setLabels(left="<font size='4'>Delta</font>", bottom="<font size='4'>Stock Price</font>")

    fig_delta_text = \
        pg.TextItem(html="<div align='center'><span style='color: rgb(255,0,0);'>Black-Scholes Delta (Benchmark)</span><br><span style='color: rgb(0,0,255); ;'>Deep-Hedging Delta </span></div>", \
        anchor=(0,0), angle=0, border='w', fill=(255, 255, 200))
    fig_delta_text.setPos(self.S_range.min(),self.model_delta.max())

    fig_delta.addItem(self.BS_delta_plot)
    fig_delta.addItem(fig_delta_text)
                            
    return fig_delta
  
      
  # Draw loss plot (PlotWidget) - Black-Scholes vs Deep Hedging.
  def Loss_Plot_Widget(self):
    fig_loss = pg.PlotWidget()
        
    self.DH_loss_plot = pg.PlotDataItem(pen = pg.mkPen(color="b", width=6), \
                                            symbolBrush=(0,0,255), symbolPen='y', symbol='+', symbolSize=8)
    fig_loss.addItem(self.DH_loss_plot)

    # Add a line for the Black-Scholes price.
    fig_loss.addLine(y=self.loss_BS, pen=pg.mkPen(color="r", width=1.5))

    self.BS_loss_html = "<div align='center'><span style='color: rgb(255,0,0);'>Black-Scholes Loss (Benchmark)</span><br><span style='color: rgb(0,0,0); font-size: 16pt;'>{:0.3f}</span></div>".format(self.loss_BS)
    self.BS_loss_textItem = pg.TextItem(html=self.BS_loss_html, anchor=(1,1), angle=0, border='w', fill=(255,255,200))
    
    # Label the graph.
    fig_loss.setTitle("<font size='5'> Loss Function (Option Price) </font>")
    fig_loss.setLabels(left="<font size='4'>Loss Value</font>", bottom="<font size='4'>Loss Function (Option Price) - Number of Epochs</font>")

    # Set appropriate xRange and yRange.
    fig_loss.setRange(xRange = (0, self.epochs))

    return fig_loss
  
  # Update Plots - Black-Scholes vs Deep Hedging.
  def Update_Plots_Widget(self, PnL_DH = None, DH_delta = None, DH_bins = None, \
                                                          loss = None, num_epoch = None, num_batch = None, flag_last_batch_in_epoch = None):

    self.Update_PnL_Histogram(PnL_DH, DH_delta, DH_bins, loss, num_epoch, num_batch, flag_last_batch_in_epoch)
    self.Update_Delta_Plot(PnL_DH, DH_delta, DH_bins, loss, num_epoch, num_batch, flag_last_batch_in_epoch)
    self.Update_Loss_Plot(PnL_DH, DH_delta, DH_bins, loss, num_epoch, num_batch, flag_last_batch_in_epoch)
      
    self.Thread_RunDH.Figure_IsUpdated = True

  def Update_Loss_Plot(self, PnL_DH = None, DH_delta = None, DH_bins = None, \
                                                          loss = None, num_epoch = None, num_batch = None, flag_last_batch_in_epoch = None):
    # Get the latest viewRange
    yMin_View, yMax_View = self.fig_loss.viewRange()[1]

    # Update text position for Black-Scholes
    self.BS_loss_textItem.setPos(self.epochs*0.9,self.loss_BS + (yMax_View - self.loss_BS)*0.005)

    # Update text for Deep-Hedging.
    DH_loss_text_title = "<div align='center'><span style='color: rgb(0,0,255);'>Deep-Hedging Loss</span><br>"
    DH_loss_text_step = "<span style='color: rgb(34,139,34);'> Epoch: {} Batch: {}</span><br>"
    DH_loss_text_loss = "<span style='color: rgb(0,0,0); font-size: 16pt;'>{:0.3f}</span></div>"

    DH_loss_text_str = (DH_loss_text_title + DH_loss_text_step + DH_loss_text_loss).format(int(num_epoch), int(num_batch), loss)

    if num_epoch ==1 and num_batch == 1:
      self.fig_loss.addItem(self.BS_loss_textItem)

      # Setup the textbox for the deep-hedging loss.
      self.DH_loss_textItem = pg.TextItem(html= DH_loss_text_str, anchor=(0,0), angle=0, border='w', fill=(255,255,200))
      self.DH_loss_textItem.setPos(num_epoch-1,loss)
      self.fig_loss.addItem(self.DH_loss_textItem)

      self.fig_loss.enableAutoRange()

      # Mandatory pause to explain the demo. Remember to modify the algo thread as well if one wants to remove the feature.
      # This part takes care the pause button.
      self.Pause()
    else:
      self.DH_loss_textItem.setHtml(DH_loss_text_str)
      if flag_last_batch_in_epoch:
        self.DH_loss_textItem.setPos(num_epoch,loss)
        if num_epoch == 1:
          # Establish the data for the out-of-sample loss at the end of the first epoch.
          self.oos_loss_record = np.array([num_epoch, loss],ndmin=2)
        else:
          # Keep adding data at the end of each epoch.
          self.oos_loss_record = np.vstack([self.oos_loss_record, np.array([num_epoch,loss])])
        
        self.DH_loss_plot.setData(self.oos_loss_record)

    # Move the Black-Scholes textbox to the left to avoid collision of the deep-hedging textbox.
    if num_epoch > self.epochs*0.5:
        if self.epsilon == 0:
            anchor = (0,0)
        elif self.epsilon > 0 :
            anchor = (0,1)

        self.fig_loss.removeItem(self.BS_loss_textItem)
        self.BS_loss_textItem = pg.TextItem(html=self.BS_loss_html, anchor=anchor, angle=0, border='w', fill=(255,255,200))
        self.BS_loss_textItem.setPos(0,self.loss_BS + (yMax_View - self.loss_BS)*0.005)
        self.fig_loss.addItem(self.BS_loss_textItem)


  def Update_PnL_Histogram(self, PnL_DH = None, DH_delta = None, DH_bins = None, \
                                                          loss = None, num_epoch = None, num_batch = None, flag_last_batch_in_epoch = None):
    if num_epoch == 1 and num_batch == 1:
      # Update PnL Histogram
      self.DH_hist = pg.BarGraphItem(x=self.bin_edges[:-2]+self.width, height=DH_bins, width=self.width, brush='b', \
              name = "Blue - Deep Hedging", antialias=False)
      self.fig_PnL.addItem(self.DH_hist)
    else:
      # Update PnL Histograms
      self.DH_hist.setOpts(height=DH_bins)

  def Update_Delta_Plot(self, PnL_DH = None, DH_delta = None, DH_bins = None, \
                                                          loss = None, num_epoch = None, num_batch = None, flag_last_batch_in_epoch = None):
    if num_epoch == 1 and num_batch == 1:
      # Update the Delta plot
      self.DH_delta_plot = pg.PlotDataItem(symbolBrush=(0,0,255), symbolPen='b', symbol='+', symbolSize=10, name = "Deep Hedging")
      self.DH_delta_plot.setData(self.S_range, DH_delta)
      self.fig_delta.addItem(self.DH_delta_plot)
    else:
      # Update the Delta plot
      self.DH_delta_plot.setData(self.S_range,DH_delta)
          
  def simulate_stock_prices(self):
    self.nobs = int(self.Ktrain*(1+self.Ktest_ratio)) # Total obs = Training + Testing
    
    # Length of one time-step (as fraction of a year).
    self.dt = day_count.yearFraction(calculation_date,calculation_date + 1) 
    self.maturity = self.N*self.dt # Maturities (in the unit of a year)

    self.stochastic_process = BlackScholesProcess(s0 = self.S0, sigma = self.sigma, \
                                                risk_free = self.risk_free, dividend = self.dividend, day_count=day_count)
                                                
    print("\nRun Monte-Carlo Simulations for the Stock Price Process.\n")
    return self.stochastic_process.gen_path(self.maturity, self.N, self.nobs)
    print("\n")
      
  def assemble_data(self):
    self.payoff_T = self.payoff_func(self.S[:,-1]) # Payoff of the call option

    self.trade_set =  np.stack((self.S),axis=1) # Trading set

    if information_set == "S":
      self.I =  np.stack((self.S),axis=1) # Information set
    elif information_set == "log_S":
      self.I =  np.stack((np.log(self.S)),axis=1)
    elif information_set == "normalized_log_S":
      self.I =  np.stack((np.log(self.S/self.S0)),axis=1)
        
    # Structure of xtrain:
    #   1) Trade set: [S]
    #   2) Information set: [S] 
    #   3) payoff (dim = 1)
    self.x_all = []
    for i in range(self.N+1):
      self.x_all += [self.trade_set[i,:,None]]
      if i != self.N:
        self.x_all += [self.I[i,:,None]]
    self.x_all += [self.payoff_T[:,None]]

    # Split the entire sample into a training sample and a testing sample.
    self.test_size = int(self.Ktrain*self.Ktest_ratio)
    [self.xtrain, self.xtest] = train_test_split(self.x_all, test_size=self.test_size)
    [self.S_train, self.S_test] = train_test_split([self.S], test_size=self.test_size)
    [self.option_payoff_train, self.option_payoff_test] = \
            train_test_split([self.x_all[-1]], test_size=self.test_size)

    # Convert the training sample into tf.Data format (same as xtrain).
    training_dataset = tf.data.Dataset.from_tensor_slices(tuple(self.xtrain))
    return training_dataset.cache()
      
  def get_Black_Scholes_Prices(self):
    # Obtain Black-Scholes price, delta, and PnL
    call = European_Call()
    price_BS = call.get_BS_price(S = self.S_test[0], sigma = self.sigma, risk_free = self.risk_free, \
                                        dividend = self.dividend, K = self.strike, exercise_date = self.maturity_date, \
                                        calculation_date = calculation_date, day_count = day_count, dt = self.dt)
    delta_BS = call.get_BS_delta(S = self.S_test[0], sigma = self.sigma, risk_free = self.risk_free, \
                                        dividend = self.dividend, K = self.strike, exercise_date = self.maturity_date, \
                                        calculation_date = calculation_date, day_count = day_count, dt = self.dt)
    PnL_BS =  call.get_BS_PnL(S= self.S_test[0], payoff=self.payoff_func(self.S_test[0][:,-1]), \
                                        delta=delta_BS, dt=self.dt, risk_free = self.risk_free, \
                                        final_period_cost=final_period_cost, epsilon=self.epsilon)
    return price_BS, delta_BS, PnL_BS
        
if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  main = MainWindow()
  main.show()
  app.exec_()
