#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

class Randomizations(object):

    def _compute_randomizations(self,
                              cond,
                              column_cell_categories,
                              permutations,
                              n_controlled_distances=0,
                              **kwargs):

        if n_controlled_distances == 0:

            return [np.asarray(np.random.permutation(self.feature_table.loc[cond,column_cell_categories]), dtype=int) for _ in range(permutations)]

        else:
            nearest_distances = self.NN_distances[cond,1:n_controlled_distances+1]

            types = np.unique(self.feature_table.loc[cond, column_cell_categories])
            n_types = self.feature_table.loc[cond,:].groupby(column_cell_categories).agg('count').values[:,0]

            gkde_types = [
                    gaussian_kde((nearest_distances[self.feature_table.loc[cond,column_cell_categories].values==i]).T)
                    for i in types
                           ]

            randomizations = np.zeros((permutations, np.sum(cond)), dtype=int)

            for index_n, n in enumerate(nearest_distances):

                p_vect = np.cumsum([gkde_i.evaluate(n)*n_type_i for gkde_i, n_type_i in zip(gkde_types, n_types)])
                p_vect /= p_vect[-1]

                r = np.random.rand(permutations)
                randomizations[:, index_n] = [types[np.argmax(ri<=p_vect)] for ri in r]

            # print(np.sum(randomizations == 1, axis=1), np.sum(randomizations == 2, axis=1))

            return list(randomizations) 