import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import seaborn as sns
import pandas as pd

sys.path.append("/users/biocomp/frose/Documents/morphoProfile/pySpacell/pySpacell/")
from pySpacell import Spacell


###########################################################################
### INPUT 
###########################################################################
os.chdir(os.path.dirname(os.path.realpath(__file__)))

image_label_file = "../data/hepatocytes_label.tif"
feature_file = "../data/hepatocytes.csv"

column_x_y = ['Location_Center_X', 'Location_Center_Y']

spa = Spacell(feature_file, 
         image_label_file, 
         column_x_y=column_x_y)
spa._debug = True

niterations = 1

spa.compute_neighborhood_matrix('radius', 0, 100, iterations=niterations, save_neighbors=True)

spa.plot_neighborhood('radius', 0, 100, iterations=niterations)
plt.show()