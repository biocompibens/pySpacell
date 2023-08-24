#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial import distance
import collections

class Pairs(object):

    def _compute_pairs_test(self,
                        feature_column,
                        permutations=999,
                        quantiles=[2.5, 97.5],
                        **kwargs):

        '''
        Computes Ripley's K cross-function. Only works with 'radius' neighborhood matrix type.
        Stores results in the dataframe perimage_results_table.
        Only works with network, 0, 1 neighborhood matrix
        cf "Multiplexed imaging mass cytometry reveals distinct tumor-immune microenvironments linked to Immunotherapy
        Responses in Melanoma" paper of DOI 10.1038/s43856-022-00197-2

        :feature_column: str
                          features' name from feature_table.

        :permutations: int
                       number of random permutations to compute the quantiles and p-value
        :quantiles: list of 2 float
                    quantiles, 2 numbers between 0 and 100
        '''

        try:
            w = self.get_neighborhood_matrix('network', 0, 1)
        except ValueError:
            self.compute_neighborhood_matrix('network', 0, 1, save_neighbors=False)
            w = self.get_neighborhood_matrix('network', 0, 1)

        suffix = self.get_suffix('network', 0, 1)
        cond = self.feature_table['NumberNeighbors_{}'.format(suffix)] > 0
        class_object, classes = pd.factorize(self.feature_table.loc[cond, feature_column], sort=True)
        count_classes = pd.value_counts(self.feature_table.loc[cond, feature_column])[classes].values
        #
        # if (count_classes <= 1).all():
        #     raise TypeError("{} is not categorical".format(feature_column))

        # if self._debug:
        #     print("in _compute_pairs_test", classes, count_classes)

        # area = self.image_label.size
        # lambda_i = count_classes/area
        # sum_lambda = lambda_i + lambda_i[:, np.newaxis]

        # edge_corr = self._get_edge_correction_area(radius)
        product = w[0][cond][:, cond] #* edge_corr

        ### monte carlo simulations: "fix the combined set of locations and the number of each type of event,
        ### then randomly assigns labels to locations." Dixon 2002
        shuffled_labels = self._compute_randomizations(cond,
                                  feature_column,
                                  permutations,
                                  **kwargs)

        return w, shuffled_labels
        # pairs_cross = np.zeros((len(classes), len(classes)))
        # pairs_cross_permutations = np.zeros((len(classes), len(classes), permutations))
        #
        # for i, count_i in enumerate(count_classes):
        #     for j, count_j in enumerate(count_classes):
        #         big_sum = np.nansum(product[class_object == i, :]
        #                             [:, class_object == j])
        #         pairs_cross[i, j] = big_sum
        #
        #         pairs_cross_perm = np.array([np.nansum(product[class_shuffled == classes[i], :]
        #                                                 [:, class_shuffled == classes[j]])
        #                                       for class_shuffled in shuffled_labels])
        #         pairs_cross_permutations[i, j, :] = pairs_cross_perm
        #
        # proba_interaction = ((np.sum(pairs_cross[..., np.newaxis] >= pairs_cross_permutations, axis=-1) + 1)
        #                      / (permutations + 1))
        #
        # for index_category1, category1 in enumerate(classes):
        #     for index_category2, category2 in enumerate(classes):
        #         subset_columns = ['feature', \
        #                    'neighborhood_matrix_type', 'neighborhood_p0', 'neighborhood_p1',
        #                    'nb_permutations', 'type_result', 'result']
        #         values = ("{}_{}_{}".format(feature_column, category1, category2), \
        #                     'network', 0, 1, permutations, 'pair_proba_interaction', proba_interaction[index_category1, index_category2])
        #
        #         self.perimage_results_table.loc[self.perimage_results_table.shape[0], subset_columns] = values
        #
        # return classes, proba_interaction