import os
import sys

# Add the parent directory to the search paths to import the libraries.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "/".join([dir_path, ".."]))

import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from pyqtgraph.Qt import QtCore

from loss_metrics import Entropy


# Reducing learning rate
reduce_lr_param = {"patience": 2, "min_delta": 1e-3, "factor": 0.5}

# Number of bins to plot for the PnL histograms.
num_bins = 30


# Put the deep-hedging algo in a separate thread than the plotting thread to
# improve performance.
class DHworker(QtCore.QThread):
    DH_outputs = QtCore.pyqtSignal(np.ndarray,
                                   np.ndarray,
                                   np.ndarray,
                                   np.float32,
                                   float,
                                   float,
                                   bool)

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run_deep_hedge_algo(self,
                            training_dataset=None,
                            epochs=None,
                            Ktrain=None,
                            batch_size=None,
                            model=None,
                            submodel=None,
                            strategy_type=None,
                            loss_param=None,
                            learning_rate=None,
                            xtest=None,
                            xtrain=None,
                            initial_price_BS=None,
                            width=None,
                            I_range=None,
                            x_range=None):
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
        # Extract in-sample loss from the previous epoch. Comparison starts in
        # epoch 2 and the index for epoch 1 is 0 -> -2
        min_loss = self.loss_record[:, 1].min()
        if min_loss - loss < reduce_lr_param["min_delta"]:
            self.reduce_lr_counter += 1

        if self.reduce_lr_counter > reduce_lr_param["patience"]:
            self.learning_rate = self.learning_rate * reduce_lr_param["factor"]
            self.optimizer.learning_rate = self.learning_rate
            print(
                "The learning rate is reduced to {}.".format(
                    self.learning_rate))
            self.reduce_lr_counter = 0

    def run(self):
        # Initialize pause and stop buttons.
        self._exit = False
        self._pause = False

        # Variables to control skipped frames. If the DH algo output much
        # faster than the graphic output, the plots can be jammed.
        self.Figure_IsUpdated = True

        self.reduce_lr_counter = 0
        self.early_stopping_counter = 0

        certainty_equiv = tf.Variable(0.0, name="certainty_equiv")

        # Accelerator Function.
        model_func = tf.function(self.model)
        submodel_func = tf.function(self.submodel)

        self.optimizer = Adam(learning_rate=self.learning_rate)

        oos_loss = None
        PnL_DH = None
        DH_delta = None
        DH_bins = None
        num_batch = None

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
                except BaseException:
                    # Reduce learning rates and Early Stopping are based on
                    # in-sample losses calculated once per epoch.
                    in_sample_wealth = model_func(self.xtrain)
                    in_sample_loss = Entropy(
                        in_sample_wealth, certainty_equiv, self.loss_param)

                    if num_epoch >= 1:
                        print(("The deep-hedging price is {:0.4f} after " +
                               "{} epoch.").format(oos_loss, num_epoch))

                        # Programming hack. The deep-hedging algo computes 
                        # faster than the computer can plot, so there could
                        # be missing frames, i.e. there is no guarantee 
                        # that every batch is plotted. Here, I force a 
                        # signal to be emitted at the end of an epoch.
                        time.sleep(1)

                        self.DH_outputs.emit(
                            PnL_DH,
                            DH_delta,
                            DH_bins,
                            oos_loss.numpy().squeeze(),
                            num_epoch,
                            num_batch,
                            True)

                        # This is needed to prevent the output signals from
                        # emitting faster than the system can plot a graph.
                        #
                        # The performance is much better than emitting at fixed
                        # time intervals.
                        self.Figure_IsUpdated = False

                    if num_epoch == 1:
                        self.loss_record = np.array(
                            [num_epoch, in_sample_loss], ndmin=2)
                    elif num_epoch > 1:
                        self.Reduce_Learning_Rate(num_epoch, in_sample_loss)
                        self.loss_record = np.vstack(
                            [self.loss_record,
                             np.array([num_epoch, in_sample_loss])])

                    mini_batch_iter = self.training_dataset.shuffle(
                        self.Ktrain).batch(self.batch_size).__iter__()
                    mini_batch = mini_batch_iter.next()

                    num_batch = 0
                    num_epoch += 1

                num_batch += 1

                # Record gradient
                with tf.GradientTape() as tape:
                    wealth = model_func(mini_batch)
                    loss = Entropy(wealth, certainty_equiv, self.loss_param)

                oos_wealth = model_func(self.xtest)
                PnL_DH = oos_wealth.numpy().squeeze()  # Out-of-sample

                submodel_delta_range = np.expand_dims(self.I_range, axis=1)
                if self.strategy_type == "simple":
                    submodel_inputs = submodel_delta_range
                elif self.strategy_type == "recurrent":
                    # Assume previous delta is ATM.
                    submodel_inputs = [
                        submodel_delta_range,
                        np.ones_like(submodel_delta_range) * 0.5]
                DH_delta = submodel_func(submodel_inputs).numpy().squeeze()
                DH_bins, _ = np.histogram(
                    PnL_DH + self.initial_price_BS,
                    bins=num_bins,
                    range=self.x_range)

                # Forward and backward passes
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_weights))

                # Compute Out-of-Sample Loss
                oos_loss = Entropy(
                    oos_wealth, certainty_equiv, self.loss_param)

                if self.Figure_IsUpdated:
                    self.DH_outputs.emit(
                        PnL_DH,
                        DH_delta,
                        DH_bins,
                        oos_loss.numpy().squeeze(),
                        num_epoch,
                        num_batch,
                        False)

                # This is needed to prevent the output signals from emitting
                # faster than the system can plot a graph.
                #
                # The performance is much better than emitting at fixed time
                # intervals.
                self.Figure_IsUpdated = False

                # Mandatory pause for the first iteration to explain demo.
                if num_epoch == 1 and num_batch == 1:
                    self.pause()
            else:
                time.sleep(1)
