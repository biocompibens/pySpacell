#!/usr/bin/env python3

import numpy as np
import pandas as pd
import collections

class Assortativity(object):

    def _compute_assortativity(self,
                              feature_column, 
                              neighborhood_matrix_type,
                              neighborhood_p0,
                              neighborhood_p1,
                              *args,
                              iterations='None',
                              permutations=0,
                              quantiles = [2.5, 97.5],
                              **kwargs):
        ''' 
            Computes Newman's assortativity function.
            one-sided test: only look at larger assortativity than random.
            Stores the results in the dataframe perimage_results_table.

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

        if len(quantiles) != 2:
            raise ValueError("quantiles need to be a tuple or a list of 2 elements: the low and high quantiles values from 0 to 100%")

        ### get the neighborhood matrix
        try:
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations=iterations)
        except ValueError:
            self.compute_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations=iterations)
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations=iterations)

        W_mat = np.array(w[0])
        ## make sure its a binary matrix
        W_mat[W_mat > 0] = 1

        ### get the feature values
        cond = self.feature_table[self._column_objectnumber].isin(w[1])
        indices_X, values_X = pd.factorize(self.feature_table.loc[cond, feature_column], sort=True)
        self.feature_table.loc[cond, "class_indices_assortativity_{}".format(feature_column)] = indices_X

        ### compute asortativity
        norm_mat = self._compute_matrix_for_assortativity(indices_X, W_mat)
        M = np.sum(W_mat)
        true_assortativity = self._newman_assortativity_coef(norm_mat, M)
        
        ### compute the randomizations and corresponding assortativity
        if permutations != 0:
            random_categories = self._compute_randomizations(cond,
                              "class_indices_assortativity_{}".format(feature_column),
                              permutations, **kwargs)
            randomizations = [self._compute_matrix_for_assortativity(random_cat, W_mat) for random_cat in random_categories]
            
            sim = [self._newman_assortativity_coef(rand, M) for rand in randomizations]

            r_sim = [s.r for s in sim]
            larger = np.sum(r_sim > true_assortativity.r)
            # if (permutations - larger) < larger:
            #     larger = permutations - larger
            ### ONE-SIDED TEST: ONLY LOOK AT LARGER ASSORTATIVITY THAN RANDOM
            p_sim = (larger + 1.) / (permutations + 1.)
            low_q, high_q = np.percentile(r_sim, quantiles)

            ## r_i_sim matrix with as many lines as permutations, and as many columns as categories
            r_i_sim = np.vstack([s.r_i for s in sim])
            r_i = np.array(true_assortativity.r_i)
            larger = np.sum(r_i_sim > r_i, axis=0)
            p_i_sim = (larger + 1.) / (permutations + 1.)
            low_q_i, high_q_i = np.percentile(r_i_sim, quantiles, axis=0)

        else:

            p_sim, p_i_sim, low_q, high_q = 'None', 'None', 'None', 'None'

        subset_columns = ['feature', \
                           'neighborhood_matrix_type', 'neighborhood_p0', 'neighborhood_p1',
                           'neighborhood_nb_iterations', \
                           'nb_permutations', 'low_quantile', 'high_quantile', 'type_result', 'result']

        values = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1], "assortativity_stats", true_assortativity.r)
        self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

        values = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1], "assortativity_error", true_assortativity.err)
        self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

        values = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1], "assortativity_p_val", p_sim)
        self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

        values = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1], "assortativity_low_quantile", low_q)
        self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

        values = (feature_column, \
                        neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations,\
                        permutations, quantiles[0], quantiles[1], "assortativity_high_quantile", high_q)
        self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values


        for index_category, category in enumerate(values_X):

            values = ("{}_{}".format(feature_column, category), \
                      neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, \
                      permutations, quantiles[0], quantiles[1], "assortativity_high_quantile", high_q_i[index_category])
            self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

            values = ("{}_{}".format(feature_column, category), \
                      neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, \
                      permutations, quantiles[0], quantiles[1], "assortativity_low_quantile", low_q_i[index_category])
            self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

            values = ("{}_{}".format(feature_column, category), \
                      neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, \
                      permutations, quantiles[0], quantiles[1], "assortativity_stats", true_assortativity.r_i[index_category])
            self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values

            values = ("{}_{}".format(feature_column, category), \
                      neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, \
                      permutations, quantiles[0], quantiles[1], "assortativity_p_val", p_i_sim[index_category])
            self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values


    def _compute_matrix_for_assortativity(self, X, W_mat):
        X1 = np.tile(X, X.shape[0]).reshape((X.shape[0], X.shape[0]))
        X2 = X1.T
        end1 = X1[W_mat.astype(bool)]
        end2 = X2[W_mat.astype(bool)]

        nb_classes = len(np.unique(X))
        mat = np.zeros((nb_classes, nb_classes))

        for in_ in range(len(end1)):
            i = end1[in_]
            j = end2[in_]
            mat[i,j] += 1

        norm_mat = mat/np.sum(mat)
        return norm_mat


    def _newman_assortativity_coef(self, norm_mat, M):
        ## norm_mat: normalized matrix for 
        ## M: total number of links
        trace = np.sum(np.diag(norm_mat))
        a_i = np.sum(norm_mat, axis=0)
        norm = np.sum(a_i**2)

        r_i = np.array([(norm_mat[i,i] - a_i[i]**2)/(a_i[i] - a_i[i]**2) for i in range(norm_mat.shape[0])])

        r = (trace - norm)/(1 - norm)
        power3 = np.sum(a_i*a_i*a_i)                                                                                      
        err = 1./M * (norm + norm**2 - 2*power3) / (1 - norm) 

        assortativty_results = collections.namedtuple("assortativity_results", ["r_i", "r", "err"])
        return assortativty_results(r_i, r, err)
