
import pysal

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import pandas as pd

from pySpacell import Spacell


###########################################################################
###
### SCRIPT FIGURE HEPATOCYTES (FIG2)
###
###########################################################################


###########################################################################
### INPUT FILES
###########################################################################
os.chdir(os.path.dirname(os.path.realpath(__file__)))

image_label_file = "../data/hepatocytes_label.tif"
feature_file = "../data/hepatocytes.csv"

column_objectnumber=  'ObjectNumber'
column_x_y = ['Location_Center_X', 'Location_Center_Y']


feature = 'Intensity_MeanIntensity_OrigRed_highDR'
sa_method = 'moran'
sa_local_method = 'getisord'
sa_local_star = True

###########################################################################
### COMPUTE CELL CROWDNESS FEATURES
###########################################################################

spa = Spacell(feature_file, 
         image_label_file, 
         column_x_y=column_x_y,
         column_objectnumber=column_objectnumber)
spa._debug = False

neighborhood_matrix_type = 'network'
neighborhood_p0 = 0
neighborhood_p1 = 6
neighborhood_step = 1
iterations = 1
permutations = 999
quantiles = [2.5, 97.5]


### CORRELOGRAM NETWORK
seq_points_x = spa.correlogram(feature,
                                sa_method,
                                neighborhood_matrix_type,
                                neighborhood_p0,
                                neighborhood_p1,
                                neighborhood_step=neighborhood_step,
                                permutations=permutations,
                                quantiles=quantiles,
                                plot_bool=True)

## CORRELOGRAM RADIUSES, MATCHING FOR NUMBER OF PARIS IN EACH TEST
radiuses = spa.correlogram(feature,
                            sa_method,
                            'radius',
                            0,
                            5000,
                            like_network=True,
                            permutations=permutations,
                            quantiles=quantiles,
                            plot_bool=True)

print(radiuses)
plt.show()

### HOT AND COLD SPOTS IMAGE
spa.compute_per_object_analysis(feature, 
                            sa_local_method, 
                            'radius', 
                            radiuses[0], 
                            radiuses[1], 
                            iterations=iterations)

spa.get_hot_spots_image(feature,
                        sa_local_method,
                        'radius',
                        radiuses[0], 
                        radiuses[1], 
                        iterations=iterations,
                        image=None,
                        hot=True, cold=True)

plt.show()

