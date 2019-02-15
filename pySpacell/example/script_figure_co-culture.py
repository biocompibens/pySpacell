#! /users/biocomp/frose/anaconda3/bin/python3

import matplotlib.pyplot as plt
import scipy
import os
import sys
import numpy as np
import cv2
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.spatial.distance import cdist
from pySpacell import Spacell

from sklearn.metrics import jaccard_similarity_score
from scipy.stats import gaussian_kde


###########################################################################
### INPUT FILES
###########################################################################
os.chdir(os.path.dirname(os.path.realpath(__file__)))

image_label_file = "../data/co-culture_label.tif"
feature_file = "../data/co-culture.csv"


###########################################################################
### INPUT PARAMETERS
###########################################################################

column_x_y = ['X', 'Y']
column_objectnumber = 'index'


###########################################################################
### COMPUTE CELL CROWDNESS FEATURES
###########################################################################

spa_co = Spacell(feature_file, 
                    image_label_file, 
                    column_x_y=column_x_y,
                    column_objectnumber = column_objectnumber)
spa_co._debug = False

max_radius = 700
nb_test_points = 7
n_permutations = 999


###########################################################################
### ASSORTATIVITY NORMAL RANDOMIZATION
###########################################################################
spa_co.correlogram('cell_type',
                   "assortativity",
                   'k',
                   0,
                   30,
                   nb_test_points=8,
                   permutations=n_permutations,
                   plot_bool=True)

plt.show()

###########################################################################
### RIPLEY NORMAL RANDOMIZATION
###########################################################################
spa_co.correlogram('cell_type',
                   "ripley",
                   'radius',
                   0,
                   max_radius,
                   nb_test_points=nb_test_points,
                   permutations=n_permutations,
                   plot_bool=True)

plt.show()


###########################################################################
### HISTOGRAM AND DENSITY ESTIMATION NEAREST NEIGHBORS
###########################################################################
NN1 = spa_co.NN_distances[:,1]
cell_types = spa_co.feature_table['cell_type'].values
gkde_type = [
                gaussian_kde((NN1[spa_co.feature_table.loc[:, 'cell_type'].values==i]).T)
                for i in np.unique(cell_types)
            ]

colors = ['g', 'r']
labels = ['fibroblast', 'Hela']
x_values = np.arange(0,250,4)
for index_i, i in enumerate(np.unique(cell_types)):
    plt.hist(NN1[cell_types == i], color=colors[index_i], alpha=0.5, bins=x_values, density=True, label=labels[index_i])
    plt.plot(x_values, gkde_type[index_i](x_values), colors[index_i])
plt.xlabel('Distance to nearest neighbor')
plt.legend()
# plt.show()


###########################################################################
### RIPLEY CONSTRAINED RANDOMIZATION
###########################################################################

n_controlled_dimensions = 1
seq_points_x = spa_co.correlogram('cell_type',
                   "ripley",
                   'radius',
                   0,
                   max_radius,
                   nb_test_points=nb_test_points,
                   permutations=n_permutations,
                   n_controlled_distances=n_controlled_dimensions, 
                   plot_bool=True)

plt.show()


###########################################################################
### APPENDIX: EXAMPLES OF SINGLE RIPLEY PLOTS
###########################################################################
spa_co.plot_ripley_cross("cell_type",
                        0, max_radius,
                        1, 1,
                        permutations=n_permutations,
                        nb_test_points=nb_test_points)

spa_co.plot_ripley_cross("cell_type",
                        0, max_radius,
                        1, 2,
                        nb_test_points=nb_test_points,
                        permutations=n_permutations)

spa_co.plot_ripley_diff("cell_type",
                        0, max_radius,
                        1, 1, 2,2,
                        nb_test_points=nb_test_points,
                        permutations=n_permutations)

plt.show()