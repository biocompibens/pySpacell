#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pysal
import libpysal


class SpatialAutocorrelation(object):
    
    global_sa_methods = ['moran', 'geary', 'getisord']
    local_sa_methods = ['moran', 'getisord']

    def _test_SA(self,  
                feature_column, 
                method,
                neighborhood_matrix_type,
                neighborhood_p0,
                neighborhood_p1,
                permutations=999,
                quantiles=[2.5, 97.5],
                **kwargs):
        ''' Test spatial-autocorrelation with the selected method on the selected feature from feature_table.
            Computes one test (per feature per image). 
            Stores the result in the dataframe perimage_results_table.
            Is computed on one neighborhood matrix.

            :method: str
                     test method. Should be moran, geary or getisord

            :feature_column: str
                              features' name from feature_table.

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :permutations: int
                           number of random permutations to compute the quantiles and p-value
            :quantiles: list of 2 float
                        quantiles, 2 numbers between 0 and 100
        '''

        method_fct, get_stats  = self._autocorrelation_method(method)

        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(neighborhood_matrix_type,
                                                                                                neighborhood_p0,
                                                                                                neighborhood_p1,
                                                                                                **kwargs)
        suffix = self.get_suffix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        try:
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, pysal_object=True, **kwargs)
        except ValueError:
            self.compute_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

            if self._debug:
                print("function _test_SA", self.neighborhood_matrix_df)
                print(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, "\n")
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, pysal_object=True, **kwargs)


        if neighborhood_matrix_type == 'k':
            cond = np.array([True for _ in range(self.feature_table.shape[0])])
        else:
            cond = self.feature_table['NumberNeighbors_{}'.format(suffix)].values>0


        method_comput, cond_X = self._apply_method_to_vector(w, feature_column, cond, method_fct, permutations)
        
        multiIndex_f = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1])

        self.perimage_results_table.loc[multiIndex_f, "{}_stats".format(method)] = get_stats(method_comput)
        self.perimage_results_table.loc[multiIndex_f, "{}_z_val".format(method)] = method_comput.z_sim
        self.perimage_results_table.loc[multiIndex_f, "{}_p_val".format(method)] = method_comput.p_sim
        self.perimage_results_table.loc[multiIndex_f, "{}_low_quantile".format(method)] = np.percentile(method_comput.sim, quantiles[0])
        self.perimage_results_table.loc[multiIndex_f, "{}_high_quantile".format(method)] = np.percentile(method_comput.sim, quantiles[1])



    def _test_local_SA(self, 
                      feature_column, 
                      method, 
                      neighborhood_matrix_type,
                      neighborhood_p0,
                      neighborhood_p1,
                      permutations=999,
                      star=False,
                      quantiles=[2.5, 97.5],
                      **kwargs):
        ''' 
        Tests feature for LOCAL/SINGLE-CELL spatial autocorrelation with the specified method.
        Computes one test per object.
        Stores the test results in the dataframe feature_table.
        Is computed for a given neighborhood matrix.

        :method: str
                 test method. Should be moran or getisord

        :feature_column: str
                          features' name from feature_table.

        :neighborhood_matrix_type: str
                                    should be 'k', 'radius', or 'network'

        :neighborhood_min_p0: int or float
                              minimum bound for the neighborhood.
                              should be int for 'k' or 'network'. Can be int or float for 'radius' 
        :neighborhood_min_p1: int or float
                              maximum bound for the neighborhood.
                              should be int for 'k' or 'network'. Can be int or float for 'radius' 
        :permutations: int
                       number of random permutations to compute the quantiles and p-value
        :quantiles: list of 2 float
                    quantiles, 2 numbers between 0 and 100

        :star: bool 
               Used only if method is 'getisord'
               if star=True, includes the central object in the statistics. Hot spot analysis
               if star=False, does not include the central object in the statistics. Outlier analysis.
        ''' 

        method_fct, get_stats = self._local_autocorrelation_method(method, star)

        suffix = self.get_suffix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        try:
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, pysal_object=True, **kwargs)
        except ValueError:
            self.compute_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, pysal_object=True, **kwargs)
        
        if neighborhood_matrix_type == 'k':
            cond = np.array([True for _ in range(self.feature_table.shape[0])])
        else:
            cond = self.feature_table['NumberNeighbors_{}'.format(suffix)].values>0

        method_comput, cond_X = self._apply_method_to_vector(w, feature_column, cond, method_fct, permutations)

        self.feature_table.loc[cond&cond_X, "local_{}_{}_{}_p_val".format(method.lower(), feature_column, suffix)] = method_comput.p_sim
        self.feature_table.loc[cond&cond_X, "local_{}_{}_{}_z_val".format(method.lower(), feature_column, suffix)] = method_comput.z_sim
        self.feature_table.loc[cond&cond_X, "local_{}_{}_{}_stats".format(method.lower(), feature_column, suffix)] = get_stats(method_comput) 

        self.feature_table.loc[cond&cond_X, "local_{}_{}_{}_low_quantile".format(method.lower(), feature_column, suffix)] = np.percentile(method_comput.sim, quantiles[0], axis=0)
        self.feature_table.loc[cond&cond_X, "local_{}_{}_{}_high_quantile".format(method.lower(), feature_column, suffix)] = np.percentile(method_comput.sim, quantiles[1], axis=0)




    def _autocorrelation_method(self, method):
        if method.lower() == 'moran':
            method_fct = lambda *args, **kwargs: pysal.explore.esda.moran.Moran(*args, **kwargs,two_tailed=True)
            def get_stats(_object):
                return _object.I

        elif method.lower() == 'geary':
            method_fct = lambda *args, **kwargs: pysal.explore.esda.geary.Geary(*args, **kwargs, two_tailed=True)
            def get_stats(_object):
                return _object.C

        elif method.lower() == 'getisord':
            method_fct = lambda *args, **kwargs: pysal.explore.esda.getisord.G(*args, **kwargs, two_tailed=True)
            def get_stats(_object):
                return _object.G

        else:
            raise ValueError("no corresponding autocorrelation function to {}".format(method))

        return method_fct, get_stats





    def _apply_method_to_vector(self, w, f, cond, method_fct, permutations):

        X = self.feature_table.loc[cond,f].values
        cond_X = (~np.isinf(X))&(~np.isnan(X))

        big_X = self.feature_table.loc[:,f].values
        big_cond_X = (~np.isinf(big_X))&(~np.isnan(big_X))

        if self._debug:
            print("in _apply_method_to_vector", w.full()[0].shape, X.shape, cond_X.shape, cond.shape)

        w_full = (w.full()[0])[cond_X,:][:,cond_X]
        w = libpysal.weights.full2W(w_full)
        X = X[cond_X]

        if np.sum(cond_X) <= 2:
            raise ValueError("feature {} not suitable - all nan".format(f))

        method_comput = method_fct(X, w, permutations=permutations)

        return method_comput, big_cond_X




    def _local_autocorrelation_method(self, method, star):
        if method.lower() == 'moran':
            method_fct = lambda *args, **kwargs: pysal.explore.esda.moran.Moran_Local(*args, **kwargs)
            def get_stats(_object):
                return _object.Is

        elif method.lower() == 'getisord':
            method_fct = lambda *args, **kwargs: pysal.explore.esda.getisord.G_Local(*args, **kwargs, star=True)
            def get_stats(_object):
                return _object.Gs

        else:
            raise ValueError("no corresponding local autocorrelation function to {}".format(method))

        return method_fct, get_stats


