#! /usr/bin/python3

import pysal

import matplotlib as mpl
mpl.use('TkAgg')
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
from skimage import measure
import logging
import imageio

from pySpacell import Spacell



###########################################################################
### INPUT FILES
###########################################################################
os.chdir(os.path.dirname(os.path.realpath(__file__)))

image_label_file = "../data/FUCCI_label.tif"
feature_file = "../data/FUCCI.csv"


###########################################################################
### INPUT PARAMETERS
###########################################################################

column_objectnumber=  'label'
column_x_y = ['X', 'Y']

feature = 'mean_ratio_GFP/Cy3'
sa_method = 'moran'
sa_local_method = 'moran'
sa_local_star = True

neighborhood_matrix_type = 'k'
neighborhood_p0 = 0
neighborhood_p1 = 30
neighborhood_step = 5
iterations = 1
permutations = 999

###########################################################################
### INITIALIZE
###########################################################################

spa = Spacell(feature_file, 
         image_label_file, 
         column_x_y=column_x_y,
         column_objectnumber=column_objectnumber)

spa.feature_table.loc[:, 'log_{}'.format(feature)] = spa.feature_table[feature].apply(np.log)
log_feature = 'log_{}'.format(feature)


############################################################################
### COMPUTE AND PLOT CORRELOGRAM FOR LOG RATIO
###########################################################################

seq_points_x = spa.correlogram(log_feature,
                                sa_method,
                                neighborhood_matrix_type,
                                neighborhood_p0,
                                neighborhood_p1,
                                neighborhood_step=neighborhood_step,
                                permutations=permutations,
                                quantiles=[2.5, 97.5], 
                                plot_bool=True)

plt.show()


###########################################################################
### PER-CELL ANALYSIS ON A CROPPED SECTION OF THE IMAGE
###########################################################################


image_label_file = "../data/FUCCI_label_crop.tif"
feature_file = "../data/FUCCI_crop.csv"
im_RGB = imageio.imread("../data/FUCCI_RGB_crop.tif")

spa_crop = Spacell(feature_file, 
                 image_label_file, 
                 column_x_y=column_x_y,
                 column_objectnumber=column_objectnumber)

spa_crop.feature_table.loc[:, 'log_{}'.format(feature)] = spa_crop.feature_table[feature].apply(np.log)
log_feature = 'log_{}'.format(feature)

s0, s1 = 0, 5

spa_crop.compute_per_object_analysis(log_feature, sa_local_method, 
                                neighborhood_matrix_type, s0, s1,
                                star=sa_local_star, permutations=permutations)

spa_crop.get_hot_spots_image(log_feature, sa_local_method,
                   neighborhood_matrix_type, s0, s1, 
                   image=im_RGB, hot=True, cold=False, contours=True)
plt.show()

spa_crop.get_hot_spots_image(log_feature, sa_local_method,
                   neighborhood_matrix_type, s0, s1, 
                   image=im_RGB, hot=True, cold=False, contours=False)
plt.show()