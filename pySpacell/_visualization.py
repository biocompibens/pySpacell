#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw



class Visualization(object):

    def get_feature_filled_image(self, 
                                 column_to_display, 
                                 with_border=True, 
                                 voronoi_bool=False,
                                 **kwargs):
        ''' Fills the image with values from column_to_display.
            :column_to_display: str
                                column name from feature table with values you want to display
            :with_border: bool 
                          if True, will add a border of background-colored pixels between each object
                          if False, objects will touch each other like in provided label image_label
            :voronoi_bool: bool
                           if True, computes the voronoi tessalation from the object centers
                           if False, will color only the delimited object area provided by label image 
        '''

        if isinstance(column_to_display, str):
            if not column_to_display in self.feature_table.columns:
                raise ValueError("{} is not in feature_table".format(column_to_display))            
            to_plot = self.feature_table[column_to_display].values

        elif isinstance(column_to_display, np.ndarray):
            if len(column_to_display) != self.n:
                raise ValueError("column_to_display does not have the good length of {}".format(self.n))
            to_plot = column_to_display

        objNum = self.feature_table[self._column_objectnumber].values

        if voronoi_bool:
            if self.voronoi_im_with_border is None:
                self._compute_voronoi_image(with_border)
            im_to_fill = np.array(self.voronoi_im_with_border, dtype=int)
        elif with_border:
            if self.label_image_with_border is None:
                self._get_label_image_with_border()
            im_to_fill = np.array(self.label_image_with_border, dtype=int)
        else:
            im_to_fill = np.array(self.image_label, dtype=int)

        output_image = np.ones(im_to_fill.shape)*np.nan

        for i in range(objNum.shape[0]):
            output_image[im_to_fill == objNum[i]] = to_plot[i]
        return output_image


    def plot_2im(self, input1, input2, 
             cmap1='plasma', cmap2='plasma',
             **kwargs):

        ''' Plots 2 filled images side by side. 
            :column_name1: str
                       column name from feature table which contains values for the left plot
            :column_name2: str
                       column name from feature table which contains values for the right plot
            :title1: str, optional
                    title for the left plot
            :title2: str, optional
                    title for the right plot
            :cmap1: str
                    name of a matplotlib cmap, for the left plot
            :cmap2: str
                    name of a matplotlib cmap, for the right plot
            :min_value1: float
                         minimum value of the colormap for the left plot
            :min_value2: float
                         minimum value of the colormap for the right plot
            :max_value1: float
                         maximum value of the colormap for the left plot
            :max_value2: float
                         maximum value of the colormap for the right plot

        '''

        if isinstance(input1, str):
            if not input1 in self.feature_table.columns:
                raise ValueError("{} is not in feature_table".format(input1))
            im1 = self.get_feature_filled_image(input1, **kwargs)
        elif isinstance(input1, np.ndarray):
            im1 = input1
        else:
            raise ValueError("input1 should be a string or a numpy array to display")

        if isinstance(input2, str):
            if not input2 in self.feature_table.columns:
                raise ValueError("{} is not in feature_table".format(input2))
            im2 = self.get_feature_filled_image(input2, **kwargs)
        elif isinstance(input2, np.ndarray):
            im2 = input2
        else:
            raise ValueError("input2 should be a string or a numpy array to display")
            

        ax1 = plt.subplot(1,2,1)

        if 'min_value1' in kwargs:
            vmin1 = kwargs.get('min_value1')
        else:
            vmin1 = np.nanmin(im1)

        if 'max_value1' in kwargs:
            vmax1 = kwargs.get('max_value1')
        else:
            vmax1 = np.nanmax(im1)

        plt.imshow(im1, 
                interpolation='none', 
                cmap=plt.get_cmap(cmap1),
                vmin=vmin1,
                vmax=vmax1)

        if 'title1' in kwargs:
            plt.title(kwargs.get('title1'))
        plt.colorbar()
        plt.axis("off")

        if 'min_value2' in kwargs:
            vmin2 = kwargs.get('min_value2')
        else:
            vmin2 = np.nanmin(im2)

        if 'max_value2' in kwargs:
            vmax2 = kwargs.get('max_value2')
        else:
            vmax2 = np.nanmax(im2)

        if im1[:2].shape == im2[:2].shape:
            plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
        else:
            plt.subplot(1,2,2)

        plt.imshow(im2, 
            interpolation='none', 
            cmap=plt.get_cmap(cmap2),
            vmin=vmin2,
            vmax=vmax2)

        plt.colorbar()
        if 'title2' in kwargs:
            plt.title(kwargs.get('title2'))
        plt.axis('off')




    def _get_label_image_with_border(self):

        self.label_image_with_border = np.array(self.image_label, dtype=np.uint16)
        kernel = np.array([[0 ,-1, 0], \
                           [-1, 4,-1], \
                           [0 ,-1, 0]])
        im_border = filters.convolve(self.image_label, kernel)
        self.label_image_with_border[im_border > 0] = 0


    def _compute_voronoi_image(self, 
                               with_border=False):
        img = Image.new('F', self.image_label.shape[::-1], 0)
        vor = Voronoi(self.feature_table[self._column_x_y].values)
        objNum = self.feature_table[self._column_objectnumber].values
        for index_pr, pr in enumerate(vor.point_region):
            if not -1 in vor.regions[pr]:
                pol = [tuple(vor.vertices[r]) for r in vor.regions[pr]]
                if with_border:
                    ImageDraw.Draw(img).polygon(pol, outline=0, fill=objNum[index_pr])
                else:
                    ImageDraw.Draw(img).polygon(pol, outline=objNum[index_pr], fill=objNum[index_pr])
        self.voronoi_im_with_border = np.array(img) 


    def plot_neighborhood(self, 
                            neighborhood_matrix_type, 
                            neighborhood_p0, 
                            neighborhood_p1,
                            image=None,
                            with_border=True, 
                            voronoi_bool=False,
                            **kwargs):
        """ Visualize neighbors on the label image (default) or on a provided image.

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :image: None or numpy array. Optional
                    if provided, neighbors will be displayed on this image
                    Default value: None
            :with_border: bool. Optional
                          Default value: True
                          if image not provided and if True, then display image will be label image with two pixels of background between each object
                          if image not provided and if False, then display image will be label image
            :voronoi_bool: bool. Optional.
                           Default value: False
                           if image not provided and if True, then display image will be the voronoi tessalation computed from label image
                           if image not provided and if False, then display image will be the label image
        """

        if image is None:
            if voronoi_bool:
                if self.voronoi_im_with_border is None:
                    self._compute_voronoi_image(with_border)
                image = np.array(self.voronoi_im_with_border, dtype=int)
            elif with_border:
                if self.label_image_with_border is None:
                    self._get_label_image_with_border()
                image = np.array(self.label_image_with_border, dtype=int)
            else:
                image = np.array(self.image_label, dtype=int)

        plt.imshow(image)
        plt.axis('off')


        suffix = self.get_suffix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)
        if neighborhood_matrix_type != 'k':
            cond = self.feature_table['NumberNeighbors_{}'.format(suffix)]>0
        else:
            cond = np.array([True for _ in range(self.n)])
        coords = self.feature_table[self._column_x_y].values

        ## plot all cells
        plt.plot(coords[:,0], coords[:,1], 'wo')

        ### only keep the cells with neighbors for the next drawing step
        coords = coords[cond]

        for index_cell, neighbors in enumerate(self.feature_table[cond]['neighbors_{}'.format(suffix)].values):
            for n in neighbors:
                neighbor_coords = self.feature_table[self.feature_table[self._column_objectnumber] == n][self._column_x_y].values[0]

                plt.plot([coords[index_cell,0], neighbor_coords[0]],\
                         [coords[index_cell,1], neighbor_coords[1]],\
                         'w-')


    def plot_correlogram(self, 
                        feature_column,
                        method,
                        neighborhood_matrix_type,
                        neighborhood_min_p0,
                        neighborhood_max_p1,
                        quantiles=[2.5, 97.5],
                        **kwargs):


        ''' Plots one feature at a time. 
            Checks if the computing has been done.
            Retrieves info from perimage_results_table.

            :feature_column: str
                              features' name from feature_table.
                              
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

        seq_points_x = self._check_correlogram_input_arguments(neighborhood_matrix_type, 
                                           neighborhood_min_p0, 
                                           neighborhood_max_p1, 
                                           **kwargs)

        self._plot_correlogram_from_seq_points_x(seq_points_x,
                                            feature_column,
                                            method,
                                            neighborhood_matrix_type, 
                                            **kwargs)


    def _plot_correlogram_from_seq_points_x(self,
                                            seq_points_x,
                                            feature_column,
                                            method,
                                            neighborhood_matrix_type, 
                                            permutations=999,
                                            quantiles=[2.5, 97.5],
                                            **kwargs):
        if method == 'ripley':
            try:
                Ks = [self.perimage_results_table.loc[(feature_column, neighborhood_matrix_type, 0, n1, 'None', permutations, quantiles[0], quantiles[1]), "ripley_results"].K for n1 in seq_points_x[1:]]
            except Exception:
                print("Ripley has not been computed on {} with {}".format(feature_column, seq_points_x))
                return 

            figK = plt.figure("Ripley_diff_cross_function_{}_{}_matrix-{}-{}-{}-_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], permutations, quantiles[0], quantiles[1]))
            plt.plot(seq_points_x[1:], Ks, 'o-', label='K')
            plt.plot(seq_points_x[1:], np.pi*seq_points_x[1:]**2, 'r-', label="pi * t^2")
            plt.legend()

            types = np.unique(self.feature_table[feature_column])
            n_types = len(types)

            if n_types > 3:
                warnings.warn("WARNING!\nThe number of possible plots for Ripley's cross functions is too large for a number of categories > 3. Please use the dedicated function BLABLABLA")

            obj_ripley = [self.perimage_results_table.loc[(feature_column, neighborhood_matrix_type, 0, n1, 'None', permutations, quantiles[0], quantiles[1]), "ripley_results"] for n1 in seq_points_x[1:]]

            fig_crossK = plt.figure("Ripley_cross_functions_{}_{}_matrix-{}-{}_{}_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], permutations, quantiles[0], quantiles[1]))

            for c, name_c in enumerate(types):
                cross = [obj.get_cross_function(c, c, **kwargs) for obj in obj_ripley]
                cross_quantiles = np.array([obj.get_quantiles(c, c, **kwargs) for obj in obj_ripley])

                if n_types == 2:
                    plt.subplot(2,2,1+c)
                else:
                    plt.subplot(2,3,1+c)

                lineplot = plt.plot(seq_points_x[1:], cross, 'o-', label='({}, {})'.format(name_c, name_c))
                plt.plot(seq_points_x[1:], cross_quantiles[:,0],'+--', c=lineplot[0].get_color())
                plt.plot(seq_points_x[1:], cross_quantiles[:,1], '+--', c=lineplot[0].get_color())
                plt.fill_between(seq_points_x[1:], y1=cross_quantiles[:,0], y2=cross_quantiles[:,1], alpha=0.2)
                plt.xlabel(neighborhood_matrix_type)
                plt.legend()

            fig_diffcrossK = plt.figure("Ripley_diff_cross_functions_{}_{}_matrix-{}-{}-{}_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], permutations, quantiles[0], quantiles[1])) 

            n_current_subplot = 1
            for c, name_c in enumerate(types):
                for c2, name_c2 in enumerate(types):
                    if c2 <= c:
                        continue

                    diff = [obj.get_diff_cross_function(c, c, c2, c2, **kwargs) for obj in obj_ripley]
                    diff_quantiles = np.array([obj.get_diff_quantiles(c, c, c2, c2, **kwargs) for obj in obj_ripley])

                    if n_types==2:
                        plt.subplot(2,2,n_current_subplot)
                        n_current_subplot += 1
                    else:
                        plt.subplot(3,4, n_current_subplot)
                        n_current_subplot += 1

                    lineplot = plt.plot(seq_points_x[1:], diff, 'o-', label='({}, {}) - ({}, {})'.format(name_c, name_c, name_c2, name_c2))
                    plt.plot(seq_points_x[1:], diff_quantiles[:,0],'--', c=lineplot[0].get_color())
                    plt.plot(seq_points_x[1:], diff_quantiles[:,1], '--', c=lineplot[0].get_color())
                    plt.fill_between(seq_points_x[1:], y1=diff_quantiles[:,0], y2=diff_quantiles[:,1], alpha=0.5)
                    plt.xlabel(neighborhood_matrix_type)
                    plt.legend()
                    
                for c2, name_c2 in enumerate(types):
                    for c3, name_c3 in enumerate(types):
                        if c3 <= c2:
                            continue
                        diff = [obj.get_diff_cross_function(c, c2, c, c3, **kwargs) for obj in obj_ripley]
                        diff_quantiles = np.array([obj.get_diff_quantiles(c, c2, c, c3, **kwargs) for obj in obj_ripley])

                        if n_types==2:
                            plt.subplot(2,2,n_current_subplot)
                            n_current_subplot += 1
                        else:
                            plt.subplot(3,4,n_current_subplot)
                            n_current_subplot += 1

                        lineplot = plt.plot(seq_points_x[1:], diff, 'o-', label='({}, {}) - ({}, {})'.format(name_c, name_c2, name_c, name_c3))
                        plt.plot(seq_points_x[1:], diff_quantiles[:,0],'--', c=lineplot[0].get_color())
                        plt.plot(seq_points_x[1:], diff_quantiles[:,1], '--', c=lineplot[0].get_color())
                        plt.fill_between(seq_points_x[1:], y1=diff_quantiles[:,0], y2=diff_quantiles[:,1], alpha=0.5)
                        plt.xlabel(neighborhood_matrix_type)
                        plt.legend()

        else:
            stats = []
            low_q = []
            high_q = []

            for index in range(len(seq_points_x)-1):
                neighborhood_p0 = seq_points_x[index]
                neighborhood_p1 = seq_points_x[index+1]

                neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

                multiIndex_f = (feature_column, \
                                neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                                permutations, quantiles[0], quantiles[1])

                if self._debug:
                    print("in _plot_correlogram_from_seq_points_x", multiIndex_f, method)

                stats.append(self.perimage_results_table.loc[multiIndex_f, "{}_stats".format(method)])

                low_q.append(self.perimage_results_table.loc[multiIndex_f, "{}_low_quantile".format(method)])
                high_q.append(self.perimage_results_table.loc[multiIndex_f, "{}_high_quantile".format(method)])

            plt.figure("{}_{}_matrix-{}-{}-{}-iterations-{}_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], iterations, permutations, quantiles[0], quantiles[1]))

            lineplot = plt.plot(seq_points_x[1:], stats, '+-', label='data')
            plt.plot(seq_points_x[1:], low_q, '--', label='{:.1f} quantile'.format(quantiles[0]), c=lineplot[0].get_color())
            plt.plot(seq_points_x[1:], high_q, ':', label='{:.1f} quantile'.format(quantiles[1]), c=lineplot[0].get_color())
            plt.fill_between(seq_points_x[1:], low_q, high_q, where=high_q>=low_q, alpha=0.5, color=lineplot[0].get_color())
            plt.legend()
            plt.title(feature_column)
            if neighborhood_matrix_type == 'radius':
                plt.xticks(ticks=seq_points_x[1:], 
                           labels=["[{:.0f},{:.0f}]".format(seq_points_x[i], seq_points_x[i+1]) for i in range(len(seq_points_x)-1)])
            else:
                plt.xticks(ticks=seq_points_x[1:], 
                           labels=["[{},{}]".format(max(1, seq_points_x[i]), seq_points_x[i+1]) for i in range(len(seq_points_x)-1)])

            plt.xlabel(neighborhood_matrix_type)
            plt.ylabel('{} statistics'.format(method))


    def plot_ripley_cross(self, 
                        feature_column,
                        neighborhood_min_p0,
                        neighborhood_max_p1,
                        class0, class1,
                        quantiles=[2.5, 97.5],
                        **kwargs):


        ''' Plots one feature at a time. 
            Checks if the computing has been done.
            Retrieves info from perimage_results_table.

            :feature_column: str
                              features' name from feature_table.
                              
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
        neighborhood_matrix_type = 'radius'
        seq_points_x = self._check_correlogram_input_arguments(neighborhood_matrix_type, 
                                           neighborhood_min_p0, 
                                           neighborhood_max_p1, 
                                           **kwargs)

        self._plot_ripley_cross_from_seq_points_x(seq_points_x,
                                            feature_column,
                                            class0, class1,
                                            **kwargs)


    def _plot_ripley_cross_from_seq_points_x(self,
                                            seq_points_x,
                                            feature_column,
                                            class0, class1,
                                            permutations=999,
                                            quantiles=[2.5, 97.5],
                                            **kwargs):
        method = 'ripley'
        neighborhood_matrix_type = 'radius'
        try:
            obj_ripley = [self.perimage_results_table.loc[(feature_column, neighborhood_matrix_type, 0, n1, 'None', permutations, quantiles[0], quantiles[1]), "ripley_results"] for n1 in seq_points_x[1:]]
        except Exception:
            print("Ripley has not been computed on {} with {}".format(feature_column, seq_points_x))
            return 


        types = np.unique(self.feature_table[feature_column])
        n_types = len(types)
        if not class0 in types:
            raise ValueError("class0 {} is not in types {}".format(class0, types))
        if not class1 in types:
            raise ValueError("class1 {} is not in types {}".format(class1, types))

        fig_crossK = plt.figure("Ripley_cross_functions_{}_{}_matrix-{}-{}_{}_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], permutations, quantiles[0], quantiles[1]))

        c0 = np.where(types == class0)[0]
        c1 = np.where(types == class1)[0]

        cross = [obj.get_cross_function(c0, c1, **kwargs) for obj in obj_ripley]
        cross_quantiles = np.array([obj.get_quantiles(c0, c1, **kwargs) for obj in obj_ripley])

        lineplot = plt.plot(seq_points_x[1:], cross, 'o-', label='({}, {})'.format(class0, class1))
        plt.plot(seq_points_x[1:], cross_quantiles[:,0],'+--', c=lineplot[0].get_color())
        plt.plot(seq_points_x[1:], cross_quantiles[:,1], '+--', c=lineplot[0].get_color())
        plt.fill_between(seq_points_x[1:], y1=cross_quantiles[:,0], y2=cross_quantiles[:,1], alpha=0.2)
        plt.xlabel(neighborhood_matrix_type)
        plt.legend()


    def plot_ripley_diff(self, 
                        feature_column,
                        neighborhood_min_p0,
                        neighborhood_max_p1,
                        class0, class1, class2, class3,
                        **kwargs):


        ''' Plots one feature at a time. 
            Checks if the computing has been done.
            Retrieves info from perimage_results_table.

            :feature_column: str
                              features' name from feature_table.
                              
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
        neighborhood_matrix_type = 'radius'
        seq_points_x = self._check_correlogram_input_arguments(neighborhood_matrix_type, 
                                           neighborhood_min_p0, 
                                           neighborhood_max_p1, 
                                           **kwargs)

        self._plot_ripley_diff_from_seq_points_x(seq_points_x,
                                            feature_column,
                                            class0, class1, class2, class3,
                                            **kwargs)



    def _plot_ripley_diff_from_seq_points_x(self,
                                            seq_points_x,
                                            feature_column,
                                            class0, class1, class2, class3,
                                            quantiles=[2.5, 97.5],
                                            permutations=999,
                                            **kwargs):
        method = 'ripley'
        neighborhood_matrix_type = 'radius'

        try:
            obj_ripley = [self.perimage_results_table.loc[(feature_column, neighborhood_matrix_type, 0, n1, 'None', permutations, quantiles[0], quantiles[1]), "ripley_results"] for n1 in seq_points_x[1:]]
        except Exception:
            print("Ripley has not been computed on {} with {}".format(feature_column, seq_points_x))
            return

        types = np.unique(self.feature_table[feature_column])
        n_types = len(types)

        if not class0 in types:
            raise ValueError("class0 {} is not in types {}".format(class0, types))
        if not class1 in types:
            raise ValueError("class1 {} is not in types {}".format(class1, types))
        if not class2 in types:
            raise ValueError("class2 {} is not in types {}".format(class2, types))
        if not class3 in types:
            raise ValueError("class3 {} is not in types {}".format(class3, types))

        fig_diffcrossK = plt.figure("Ripley_diff_cross_functions_{}_{}_matrix-{}-{}-{}_permutations-{}_quantiles-{}-{}".format(feature_column, method, neighborhood_matrix_type, seq_points_x[0], seq_points_x[-1], permutations, quantiles[0], quantiles[1])) 

        c0 = np.where(types == class0)[0]
        c1 = np.where(types == class1)[0]
        c2 = np.where(types == class2)[0]
        c3 = np.where(types == class3)[0]

        diff = [obj.get_diff_cross_function(c0, c1, c2, c3, **kwargs) for obj in obj_ripley]
        diff_quantiles = np.array([obj.get_diff_quantiles(c0, c1, c2, c3, **kwargs) for obj in obj_ripley])


        lineplot = plt.plot(seq_points_x[1:], diff, 'o-', label='({}, {}) - ({}, {})'.format(class0, class1, class2, class3))
        plt.plot(seq_points_x[1:], diff_quantiles[:,0],'--', c=lineplot[0].get_color())
        plt.plot(seq_points_x[1:], diff_quantiles[:,1], '--', c=lineplot[0].get_color())
        plt.fill_between(seq_points_x[1:], y1=diff_quantiles[:,0], y2=diff_quantiles[:,1], alpha=0.5)
        plt.xlabel(neighborhood_matrix_type)
        plt.legend()



    def get_hot_spots_image(self, feature_column, method,
                   neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, 
                   plot_bool=False, **kwargs):
        """ Returns a numpy array with the hot and cold spots image if plot_bool=False.
            Plots the hot and cold spots image and returns the numpy array and the matplotlib figure

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

            :plot_bool: bool
                        if True, returns the numpy array and the figure with the plot
                        if False, returns onyl the numpy array

        """

        if method not in self.local_sa_methods:
            raise ValueError('method should be {}'.format(self.local_sa_methods))

        suffix = self.get_suffix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        low_quantile = self.feature_table.loc[:, "local_{}_{}_{}_low_quantile".format(method, feature_column, suffix)].values
        high_quantile = self.feature_table.loc[:, "local_{}_{}_{}_high_quantile".format(method, feature_column, suffix)].values
        values = self.feature_table.loc[:, "local_{}_{}_{}_stats".format(method, feature_column, suffix)].values
        low_q_values = np.array(values)
        low_q_values[values>=low_quantile] = np.nan
        high_q_values = np.array(values)
        high_q_values[values<=high_quantile] = np.nan 

        low_q_im = self.get_feature_filled_image(low_q_values, **kwargs)
        high_q_im = self.get_feature_filled_image(high_q_values, **kwargs)

        if plot_bool:

            fig = plt.figure()
            if method.lower() == 'getisord':
                cmap1 = plt.get_cmap('Reds_r')
                cmap2 = plt.get_cmap('Blues')
            elif method.lower() =='moran':
                cmap1 = plt.get_cmap('Blues_r')
                cmap2 = plt.get_cmap('Reds')

            plt.subplot(1,2,1)
            plt.title("Low quantiles")
            plt.imshow(low_q_im, cmap=cmap1, vmax=np.nanmax(low_quantile))
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.title("High quantiles")
            plt.imshow(high_q_im, cmap=cmap2, vmin=np.nanmin(high_quantile))
            plt.colorbar()

            return low_q_im, high_q_im, fig

        else:
            return low_q_im, high_q_im


