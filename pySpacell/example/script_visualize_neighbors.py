import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import seaborn as sns
import pandas as pd

from pySpacell import Spacell


###########################################################################
### INPUT 
###########################################################################
os.chdir(os.path.dirname(os.path.realpath(__file__)))

image_label_file = "../data/FUCCI_label_crop.tif"
feature_file = "../data/FUCCI_crop.csv"

column_x_y = ['X', 'Y']
column_object = 'label'

spa = Spacell(feature_file, 
         image_label_file, 
         column_x_y=column_x_y,
         column_objectnumber=column_object)
spa._debug = True

niterations = 1

spa.compute_neighborhood_matrix('radius', 0, 80, iterations=niterations, save_neighbors=True)

spa.plot_neighborhood('radius', 0, 80, iterations=niterations)
plt.show()