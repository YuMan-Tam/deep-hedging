# Author: Yu-Man Tam
# Email: yuman.tam@gmail.com
#
# Last updated: 5/22/2020
#
# Reference: Deep Hedging (2019, Quantitative Finance) by Buehler et al.
# https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1571683

import sys
import os

import tensorflow as tf

# Add the parent directory to the search paths to import the libraries.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "/".join([dir_path, ".."]))

from pyqtgraph.Qt import QtWidgets
from main_window import MainWindow

# Tensorflow settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(0)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    app.exec_()
