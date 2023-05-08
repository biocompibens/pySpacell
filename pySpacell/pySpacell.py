#!/usr/bin/env python3
import PIL
import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndimage
import scipy.ndimage.measurements as measure
from scipy.spatial.distance import cdist
from itertools import *
from PIL import Image, ImageDraw
from skimage.io import imread

try:
    from ._neighborhood_matrix import NeighborhoodMatrixComputation
    from ._spatial_autocorrelation import SpatialAutocorrelation
    from ._assortativity import Assortativity
    from ._visualization import Visualization
    from ._ripley import Ripley
    from ._randomizations import Randomizations
except SystemError:
    from _neighborhood_matrix import NeighborhoodMatrixComputation
    from _spatial_autocorrelation import SpatialAutocorrelation
    from _assortativity import Assortativity
    from _visualization import Visualization
    from _ripley import Ripley
    from _randomizations import Randomizations


class Spacell(NeighborhoodMatrixComputation,
              SpatialAutocorrelation,
              Assortativity,
              Visualization,
              Ripley,
              Randomizations,
              object):
    matrix_types = ['k', 'radius', 'network']
    precision_float_radius = 3
    _debug = False

    def __init__(self,
                 feature_file,
                 image_label_file,
                 column_objectnumber='ObjectNumber',
                 column_x_y=['x', 'y']):
        '''
            :feature_file: str 
                           path to csv file where each row contains information of one object (cell).
                           File must contain a column (column_objectnumber) with corresponding object id from the label image.
                           File must contain 2 columns (column_x_y) with corresponding coordinates of the object (cell) center.


            :image_label_file: str
                               path to label image. Gray scale image where pixel values are corresponding to object ids.


            :column_x_y: list of str
                         2 column names for x and y coordinates in feature_file


            :column_objectnumber: str 
                                  column name for object number (corresponding to label image) in feature file

        '''

        if type(feature_file) == str:
            if not os.path.exists(feature_file):
                raise OSError("feature_file {} not found".format(feature_file))
            self.feature_table = pd.read_csv(feature_file)
        elif type(feature_file) == pd.DataFrame:
            self.feature_table = feature_file.copy()
        else:
            raise ValueError("[Spacell] feature_file should be a path to a csv file or a pandas Dataframe")

        if not column_objectnumber in self.feature_table.columns:
            raise ValueError("{} not in feature_table".format(column_objectnumber))
        self._column_objectnumber = column_objectnumber

        if not ((column_x_y[0] in self.feature_table.columns) or
                (column_x_y[1] in self.feature_table.columns)):
            raise ValueError("{} not in feature_table".format(column_x_y))
        self._column_x_y = column_x_y

        self.feature_table.dropna(axis=0,
                                  subset=self._column_x_y,
                                  inplace=True)

        self.n = self.feature_table.shape[0]

        if not os.path.exists(image_label_file):
            raise OSError("image_label_file {} not found".format(image_label_file))

        if image_label_file[-4:] == '.npy':
            self.image_label = np.load(image_label_file)
        else:
            try:
                self.image_label = np.array(Image.open(image_label_file, 'r'))
            except PIL.UnidentifiedImageError:
                self.image_label = imread(image_label_file)

        if len(self.image_label.shape) > 2:
            raise ValueError("label image should be gray scale")

        self.label_image_with_border = None
        self.voronoi_im_with_border = None

        self.NN = None
        self.NN_distances = None
        self.adjacency_matrix = {}  ## keys=number of iterations, values list of full numpy adjacency matrix and its power

        self.neighborhood_matrix_df = pd.DataFrame(columns=['type', 'p0', 'p1', 'nb_iterations', \
                                                            'neighborhood_matrix', \
                                                            'nb_pairs'])

        names_for_index = ['feature', \
                           'neighborhood_matrix_type', 'neighborhood_p0', 'neighborhood_p1',
                           'neighborhood_nb_iterations', \
                           'nb_permutations', 'low_quantile', 'high_quantile']
        my_index = pd.MultiIndex(levels=[[] for _ in range(len(names_for_index))],
                                 codes=[[] for _ in range(len(names_for_index))],
                                 names=names_for_index)
        ### to store results per image, not per self
        self.perimage_results_table = pd.DataFrame(index=my_index)

    def get_neighborhood_matrix(self,
                                neighborhood_matrix_type,
                                neighborhood_p0,
                                neighborhood_p1,
                                iterations='None',
                                pysal_object=False,
                                **kwargs):

        ''' 
        Return the neighborhood matrix for specified parameters if already computed, thraw a ValueError otherwise.
        3 modes are available: 


        1) 'k' for k-nearest neighbors;


        2) 'radius' for neighbors at an euclidean distance;


        3) 'network' for neighbors from the object graph (touching objects are neighbors) 


        For each mode, an interval is requested to know which neighbors to include.


        Examples: 

            'k', 2, 4               -> 2nd, 3rd, and 4th-nearest neighbors
                  

            'radius', 50.75, 100.85 -> neighbors at an euclidean distance falling between 50.75 pixels and 100.85 pixels
        
            'network', 0, 1         -> neighbors having boundaries touching 


        :neighborhood_matrix_type: str
                                   should be 'k', 'radius', or 'network'


        :neighborhood_p0: int or float
                          minimum bound for the neighborhood.
                          should be int for 'k' or 'network'. Can be int or float for 'radius'


        :neighborhood_p1: int or float
                          maximum bound for the neighborhood.
                          should be int for 'k' or 'network'. Can be int or float for 'radius'


        :pysal_object: bool 
                       if True, return a pysal.lib.weights object
                       if False, return a tuple (W, L) with W full numpy neighborhood matrix and L list of vertices' ids
        
        '''

        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(
            neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations)

        try:
            if self._debug:
                print("in get_neighborhood_matrix", neighborhood_p0, neighborhood_p1, iterations,
                      self.neighborhood_matrix_df[(self.neighborhood_matrix_df['type'] == neighborhood_matrix_type) &
                                                  (self.neighborhood_matrix_df['p0'] == neighborhood_p0) &
                                                  (self.neighborhood_matrix_df['p1'] == neighborhood_p1) &
                                                  (self.neighborhood_matrix_df['nb_iterations'] == iterations)][
                          'neighborhood_matrix'])

            w = self.neighborhood_matrix_df[(self.neighborhood_matrix_df['type'] == neighborhood_matrix_type) &
                                            (self.neighborhood_matrix_df['p0'] == neighborhood_p0) &
                                            (self.neighborhood_matrix_df['p1'] == neighborhood_p1) &
                                            (self.neighborhood_matrix_df['nb_iterations'] == iterations)][
                'neighborhood_matrix'].values[0]
        except:
            raise ValueError("WARNING - neighborhood matrix not already computed")

        if pysal_object:
            return w
        else:
            return w.full()

    def _check_neighborhood_matrix_parameters(self,
                                              neighborhood_matrix_type,
                                              neighborhood_p0,
                                              neighborhood_p1,
                                              iterations='None',
                                              **kwargs):

        if neighborhood_matrix_type == 'radius':

            if not isinstance(neighborhood_p0, (int, np.integer, float, np.floating)):
                raise TypeError("neighborhood_p0's type should be an int or a float for 'radius' neighborhood type")
            elif int(neighborhood_p0) == round(neighborhood_p0, self.precision_float_radius):
                neighborhood_p0 = int(neighborhood_p0)
            else:
                neighborhood_p0 = round(neighborhood_p0, self.precision_float_radius)

            if not isinstance(neighborhood_p1, (int, np.integer, float, np.floating)):
                raise TypeError("neighborhood_p1's type should be an int or a float for 'radius' neighborhood type")
            elif int(neighborhood_p1) == round(neighborhood_p1, self.precision_float_radius):
                neighborhood_p1 = int(neighborhood_p1)
            else:
                neighborhood_p1 = round(neighborhood_p1, self.precision_float_radius)

            return neighborhood_p0, neighborhood_p1, 'None'

        elif neighborhood_matrix_type == 'k':
            return int(neighborhood_p0), int(neighborhood_p1), 'None'

        elif neighborhood_matrix_type == 'network':
            if isinstance(iterations, str):
                iterations = 1
            return int(neighborhood_p0), int(neighborhood_p1), int(iterations)

        else:
            raise ValueError("neighborhood_matrix_type not understood, should be {}".format(self.matrix_types))

    def get_suffix(self,
                   neighborhood_matrix_type,
                   neighborhood_p0,
                   neighborhood_p1,
                   **kwargs):
        ''' Returns the suffix for the additionally computed features in the feature_table: local spatial-autocorrelation statistical tests' outputs, and neighborhood computation's number of neighbors for each object ('radius' or 'network' neighborhood matrix type) or distance of last neighbor ('k' neighborhood matrix type).

            :neighborhood_matrix_type: str
                                       should be 'k', 'radius', or 'network'


            :neighborhood_p0: int or float
                              minimum bound for the neighborhood.
                              should be int for 'k' or 'network'. Can be int or float for 'radius'


            :neighborhood_p1: int or float
                              maximum bound for the neighborhood.
                              should be int for 'k' or 'network'. Can be int or float for 'radius'

        '''

        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(
            neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        if neighborhood_matrix_type == 'radius':
            return 'r-{}-{}'.format(neighborhood_p0, neighborhood_p1)

        elif neighborhood_matrix_type == 'k':
            return "k-{:02d}-{:02d}".format(neighborhood_p0, neighborhood_p1)

        elif neighborhood_matrix_type == 'network':
            return "network-{:02d}-{:02d}-iterations-{}".format(neighborhood_p0, neighborhood_p1, iterations)

        else:
            raise ValueError(
                "in self.get_suffix, neighborhood_matrix_type {} not understood".format(neighborhood_matrix_type))

    def correlogram(self,
                    feature_columns,
                    method,
                    neighborhood_matrix_type,
                    neighborhood_min_p0,
                    neighborhood_max_p1,
                    **kwargs):

        ''' Computes a serie of spatial analysis tests for the provided features. Gives one value for the image.
            The starting and ending neighborhood parameters, neighborhood_p0 and neighborhood_p1, are to be set. 3 modes are available to define the intermediary neighborhood parameters.

            :feature_columns: list of str
                              features' names from feature_table.
                              All features will be tested on the same neighborhood matrices and with the same analysis method
                              
            :method: str
                     can be 
                     - 'assortativity' or 'ripley' (for categorical features), 
                     - 'moran', 'geary', 'getisord' (global spatial autocorrelation for continuous features)

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'
                                        Same 'neighborhood_matrix_type' for all the points in the correlogram.

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            

            :neighborhood_step: int or float
                                a step in terms of parameters ([p0, p0+step, p0+2step, ...p1])
                                should be int for 'k' or 'network'. Can be int or float for 'radius' 
                                one of neighborhood_step, nb_pairs_step or nb_test_points should be defined.

            :nb_pairs_step: int
                            a step in terms of number of pairs for each test.
                            Overlooked if neighborhood_step is provided.

            :nb_test_points: int
                             a number of test points.
                             Overlooked if neighborhood_step or nb_pairs_step are provided.

        '''

        neighborhood_min_p0, neighborhood_max_p1, iterations = self._check_neighborhood_matrix_parameters(
            neighborhood_matrix_type, neighborhood_min_p0, neighborhood_max_p1, **kwargs)

        seq_points_x = self._check_correlogram_input_arguments(neighborhood_matrix_type,
                                                               neighborhood_min_p0,
                                                               neighborhood_max_p1,
                                                               **kwargs)

        for index in range(len(seq_points_x) - 1):
            self.compute_per_image_analysis(feature_columns, method, neighborhood_matrix_type, seq_points_x[index],
                                            seq_points_x[index + 1], **kwargs)

        if 'plot_bool' in kwargs and kwargs.get('plot_bool'):
            if isinstance(feature_columns, list):
                for f in feature_columns:
                    if f in self.feature_table.columns:
                        self._plot_correlogram_from_seq_points_x(seq_points_x,
                                                                 f,
                                                                 method,
                                                                 neighborhood_matrix_type,
                                                                 **kwargs)
                    else:
                        print('WARNING - {} not in feature_table'.format(f))
            elif isinstance(feature_columns, str):
                if feature_columns in self.feature_table.columns:
                    self._plot_correlogram_from_seq_points_x(seq_points_x,
                                                             feature_columns,
                                                             method,
                                                             neighborhood_matrix_type,
                                                             **kwargs)
                else:
                    print('WARNING - {} not in feature_table'.format(feature_columns))

        return seq_points_x

    def _check_correlogram_input_arguments(self,
                                           neighborhood_matrix_type,
                                           neighborhood_min_p0,
                                           neighborhood_max_p1,
                                           **kwargs):

        if 'like_network' in kwargs and kwargs.get('like_network'):
            if neighborhood_matrix_type != 'radius':
                raise ValueError('nb_pairs_step is only available for neighborhood_matrix_type == radius')
            coords = self.feature_table[self._column_x_y].values
            dist_mat = cdist(coords, coords)
            total_number_pairs = self.n * (self.n - 1) / 2.

            all_distances_1d = np.triu(dist_mat)
            all_distances_1d = all_distances_1d[all_distances_1d > 0]

            if 'iterations' not in kwargs:
                iterations = 1

            nb_pairs_steps = [np.sum(mat) / 2. for mat in self.adjacency_matrix[iterations]]

            increment_percentile = np.array(nb_pairs_steps) * 1. / total_number_pairs * 100

            distance_percentiles = np.percentile(all_distances_1d, increment_percentile)

            seq_points_x = [0]
            seq_points_x.extend(distance_percentiles[np.argmax(distance_percentiles >= neighborhood_min_p0):])

        elif 'neighborhood_step' in kwargs:
            neighborhood_step = kwargs.get('neighborhood_step')
            if neighborhood_step > neighborhood_max_p1 - neighborhood_min_p0 + 1:
                raise ValueError('neighborhood_step > neighborhood_max_p1-neighborhood_min_p0+1')

            seq_points_x = np.arange(neighborhood_min_p0, neighborhood_max_p1, neighborhood_step, dtype=float)

        elif 'nb_pairs_step' in kwargs:
            nb_pairs_step = kwargs.get('nb_pairs_step')
            if neighborhood_matrix_type != 'radius':
                raise ValueError('nb_pairs_step is only available for neighborhood_matrix_type == radius')

            coords = self.feature_table[self._column_x_y].values
            dist_mat = cdist(coords, coords)
            total_number_pairs = self.n * (self.n - 1) / 2.

            all_distances_1d = np.triu(dist_mat)
            all_distances_1d = all_distances_1d[all_distances_1d > 0]

            increment_percentile = nb_pairs_step * 1. / total_number_pairs * 100

            distance_percentiles = np.percentile(all_distances_1d,
                                                 np.arange(0, 100, increment_percentile))

            seq_points_x = distance_percentiles[np.argmax(distance_percentiles >= neighborhood_min_p0):np.argmax(
                distance_percentiles > neighborhood_max_p1) + 1]

        elif 'nb_test_points' in kwargs:
            nb_test_points = kwargs.get('nb_test_points')
            seq_points_x = np.linspace(neighborhood_min_p0, neighborhood_max_p1, num=nb_test_points + 1)

        else:
            raise ValueError("one of neighborhood_step, nb_pairs_step, nb_test_points should be provided")

        if neighborhood_matrix_type in ['k', 'network']:
            seq_points_x = np.unique([int(w) for w in seq_points_x])

        return seq_points_x

    def compute_per_image_analysis(self,
                                   feature_columns,
                                   method,
                                   neighborhood_matrix_type,
                                   neighborhood_p0,
                                   neighborhood_p1,
                                   **kwargs):

        ''' Computes per image spatial analysis tests for the provided features for one neighborhood matrix.

            :feature_columns: list of str
                              features' names from feature_table.
                              All features will be tested on the same neighborhood matrix and with the same analysis method.
                              
            :method: str
                     can be 
                     - 'assortativity' or 'ripley' (for categorical features), 
                     - 'moran', 'geary', 'getisord' (global spatial autocorrelation for continuous features)

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            

        '''

        if self._debug:
            print("\ncompute_per_image_analysis", feature_columns, method, neighborhood_matrix_type, neighborhood_p0,
                  neighborhood_p1, kwargs)

        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(
            neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        if method.lower() == 'assortativity':
            if isinstance(feature_columns, list):
                for f in feature_columns:
                    if f in self.feature_table.columns:
                        self._compute_assortativity(f,
                                                    neighborhood_matrix_type,
                                                    neighborhood_p0,
                                                    neighborhood_p1,
                                                    **kwargs)
                    else:
                        print("WARNING feature {} not in feature_table".format(f))
            elif isinstance(feature_columns, str):
                if feature_columns in self.feature_table.columns:
                    self._compute_assortativity(feature_columns,
                                                neighborhood_matrix_type,
                                                neighborhood_p0,
                                                neighborhood_p1,
                                                **kwargs)
                else:
                    print("WARNING feature {} not in feature_table".format(feature_columns))

        elif method.lower() == 'ripley':
            print("neighborhood_p0 is set to 0")
            if isinstance(feature_columns, list):
                for f in feature_columns:
                    if f in self.feature_table.columns:
                        self._compute_ripley(f,
                                             neighborhood_p1,
                                             **kwargs)
                    else:
                        print("WARNING feature {} not in feature_table".format(f))
            elif isinstance(feature_columns, str):
                if feature_columns in self.feature_table.columns:
                    self._compute_ripley(feature_columns,
                                         neighborhood_p1,
                                         **kwargs)
                else:
                    print("WARNING feature {} not in feature_table".format(feature_columns))

        elif method.lower() in self.global_sa_methods:
            if isinstance(feature_columns, list):
                for f in feature_columns:
                    if f in self.feature_table.columns:
                        self._test_SA(f,
                                      method,
                                      neighborhood_matrix_type,
                                      neighborhood_p0,
                                      neighborhood_p1,
                                      **kwargs)

            elif isinstance(feature_columns, str):
                if feature_columns in self.feature_table.columns:
                    self._test_SA(feature_columns,
                                  method,
                                  neighborhood_matrix_type,
                                  neighborhood_p0,
                                  neighborhood_p1,
                                  **kwargs)
        else:
            raise ValueError("method should be assortativity, ripley, moran, geary, or getisord.")

    def compute_per_object_analysis(self,
                                    feature_columns,
                                    method,
                                    neighborhood_matrix_type,
                                    neighborhood_p0,
                                    neighborhood_p1,
                                    **kwargs):

        ''' Computes per object spatial analysis tests for the provided features for one neighborhood matrix.

            :feature_columns: list of str
                              features' names from feature_table.
                              All features will be tested on the same neighborhood matrices and with the same analysis method
                              
            :method: str
                     can be 'moran' or'getisord' (local spatial autocorrelation for continuous features)

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            
        '''

        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(
            neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        if self._debug:
            print("\ncompute_per_object_analysis", feature_columns, method, neighborhood_matrix_type, neighborhood_p0,
                  neighborhood_p1, iterations, kwargs)

        if method.lower() in self.local_sa_methods:
            if isinstance(feature_columns, list):
                for f in feature_columns:
                    if f in self.feature_table.columns:
                        self._test_local_SA(f,
                                            method,
                                            neighborhood_matrix_type,
                                            neighborhood_p0,
                                            neighborhood_p1,
                                            **kwargs)
                    else:
                        print("WARNING feature {} not in feature_table".format(f))
            elif isinstance(feature_columns, str):
                if feature_columns in self.feature_table.columns:
                    self._test_local_SA(feature_columns,
                                        method,
                                        neighborhood_matrix_type,
                                        neighborhood_p0,
                                        neighborhood_p1,
                                        **kwargs)
                else:
                    print("WARNING feature {} not in feature_table".format(feature_columns))
        else:
            raise ValueError("per object analysis is only available for moran or getisord methods.")
