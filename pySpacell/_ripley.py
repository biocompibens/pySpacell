#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial import distance
import collections

class Ripley(object):

    def _compute_ripley(self,
                        feature_column,
                        radius,
                        permutations=999,
                        quantiles=[2.5, 97.5],
                        **kwargs):

        '''
        Computes Ripley's K cross-function. Only works with 'radius' neighborhood matrix type.
        Stores results in the dataframe perimage_results_table.

        :feature_column: str
                          features' name from feature_table.

        :radius: int or float
                 maximum distance at which objects are considered save_neighbors

        :permutations: int
                       number of random permutations to compute the quantiles and p-value
        :quantiles: list of 2 float
                    quantiles, 2 numbers between 0 and 100
        '''

        area = self.image_label.size

        _, radius, _ = self._check_neighborhood_matrix_parameters('radius', 0, radius, 'None')

        try:
            w = self.get_neighborhood_matrix('radius', 0, radius)
        except ValueError:
            self.compute_neighborhood_matrix('radius', 0, radius, save_neighbors=False)
            w = self.get_neighborhood_matrix('radius', 0, radius)

        suffix = self.get_suffix('radius', 0, radius)
        cond = self.feature_table['NumberNeighbors_{}'.format(suffix)]>0
        class_object, classes = pd.factorize(self.feature_table.loc[cond, feature_column], sort=True)
        count_classes = pd.value_counts(self.feature_table.loc[cond, feature_column])[classes].values

        if self._debug:
            print("in _compute_ripley", classes, count_classes)

        lambda_i = count_classes/area
        sum_lambda = lambda_i + lambda_i[:,np.newaxis]

        if (count_classes<=1).all():
            raise TypeError("{} is not categorical".format(feature_column))

        edge_corr = self._get_edge_correction_area(radius)
        product = w[0]*edge_corr

        ### monte carlo simulations: "fix the combined set of locations and the number of each type of event, then randomly assigns labels to locations." Dixon 2002
        shuffled_labels = self._compute_randomizations(cond,
                                  feature_column,
                                  permutations,
                                  **kwargs)
        print(shuffled_labels[0].shape, product.shape)

        ripley_cross = np.zeros((len(classes), len(classes)))
        ripley_cross_permutations = np.zeros((len(classes), len(classes), permutations))

        for i, count_i in enumerate(count_classes):
            for j, count_j in enumerate(count_classes):

                big_sum = np.nansum(product[class_object == i,:]
                                           [:,class_object == j])
                factor = area/count_i/count_j
                ripley_cross[i,j] = factor*big_sum

                ripley_cross_perm = np.array([factor*
                                np.nansum(product[class_shuffled == classes[i],:]
                                                 [:,class_shuffled == classes[j]]) 
                                for class_shuffled in shuffled_labels])
                ripley_cross_permutations[i,j,:] = ripley_cross_perm


        if self._debug:
            print("in _compute_ripley", ripley_cross, classes, lambda_i)
            
        ripley_cross_star = (ripley_cross*lambda_i + (ripley_cross*lambda_i).T)/sum_lambda
        K = area*np.nansum(product)/self.n/self.n

        ripley_results = RipleyObject(classes, count_classes, K, ripley_cross, ripley_cross_star, ripley_cross_permutations, lambda_i)

        multiIndex_f = (feature_column, \
                        'radius', 0, radius, 'None',\
                        permutations, quantiles[0], quantiles[1])

        self.perimage_results_table.loc[multiIndex_f, "ripley_results"] = ripley_results
        self.perimage_results_table.sortlevel(inplace=True)

        
        

    def _get_edge_correction_area(self, radius):
        '''Corrects the estimation for the cases where the disc is not completely inside the image.
           cf Protazio
        '''

        length, width = self.image_label.shape

        cond = self.feature_table['NumberNeighbors_{}'.format(self.get_suffix('radius', 0, radius))] > 0

        if self._debug:
            print("_get_edge_correction_area", 'NumberNeighbors_{}'.format(self.get_suffix('radius', 0, radius)), np.unique(cond, return_counts=True))

        coordinates = self.feature_table.loc[cond, self._column_x_y].values
        size = coordinates.shape[0]

        d_x = np.min([coordinates[:,0], width - coordinates[:,0]], axis=0)
        d_y = np.min([coordinates[:,1], length - coordinates[:,1]], axis=0)
        d = np.minimum(d_x, d_y)


        cond1 = radius < d_x ## no intersection for x axis
        cond2 = radius < d_y ## no intersection for y axis
        cond3 = radius**2 < d_x**2 + d_y**2 ## intersection for x axis, y axis or both

        alpha = np.arccos(d/radius)
        e = np.sqrt(radius**2 - d**2)
        case2 = (np.pi * radius**2)/(e*d + (np.pi-alpha)*radius**2) 
        del d, alpha, e

        ## CASE1 = no intersection, value is 1
        edge_corr = np.ones(cond1.shape)

        ## CASE2 = cond1 or cond2
        edge_corr[cond1 ^ cond2] = case2[cond1 ^ cond2]
        del case2

        ## else
        alpha_x = np.arccos(d_x/radius)
        alpha_y = np.arccos(d_y/radius)

        e_x = np.sqrt(radius**2 - d_x**2)
        e_y = np.sqrt(radius**2 - d_y**2)

        ## CASE3: 2 intersections with r^2 > d_x^2 + d_y^2
        case3 = (np.pi * radius**2)/(d_x*d_y + 0.5*(e_x*d_x+e_y*d_y) + (0.75*np.pi-0.5*alpha_x-0.5*alpha_y)*radius**2)
        edge_corr[(~cond1)&(~cond2)&(cond3)] = case3[(~cond1)&(~cond2)&(cond3)]
        del case3

        case4 = (np.pi * radius**2)/(e_x*d_x + e_y*d_y + (np.pi - alpha_x - alpha_y)*radius**2)
        del alpha_x, alpha_y, e_x, e_y, d_x, d_y, radius

        edge_corr[(~cond1)&(~cond2)&(~cond3)] = case4[(~cond1)&(~cond2)&(~cond3)]
        del cond1, cond2, cond3, case4

        return edge_corr





        
class RipleyObject(object):

    def __init__(self, 
                classes,
                count_classes,
                K,
                K_cross,
                K_cross_star,
                permutations,
                lambda_i):
        self.K = K
        self.K_cross = K_cross
        self.K_cross_star = K_cross_star
        self.permutations = permutations
        self.classes = classes
        self.count_classes = count_classes
        self.lambda_i = lambda_i

    def get_cross_function(self, class1, class2, star=True, variance_stabilization=True, **kwargs):
        if star:
            K = self.K_cross_star[class1, class2]
        else:
            K = self.K_cross[class1, class2]
        if variance_stabilization:
            return np.sqrt(K/np.pi)
        return K

    def get_pval(self, class1, class2, star=True, **kwargs):
        n_permutations = self.permutations.shape[2]
        if star:
            sum_lambda = self.lambda_i + self.lambda_i[:,np.newaxis]
            shuffled_K_cross_star = np.array([(self.permutations[class1,class2,p]*self.lambda_i[class1] + (self.permutations[class1,class2,p]*self.lambda_i[class2]).T)/sum_lambda for p in range(n_permutations)])
            p_val = (np.sum(shuffled_K_cross_star > self.K_cross_star[class1,class2]) + 1.)/(n_permutations + 1.)
        else:
            p_val = (np.sum(permutations[class1, class2,:] > self.K_cross[class1,class2]) + 1.)/(n_permutations + 1.)

        return p_val

    def get_quantiles(self, class1, class2, star=True, quantiles=[2.5, 97.5], variance_stabilization=True, **kwargs):
        n_permutations = self.permutations.shape[2]
        if star:
            sum_lambda = self.lambda_i + self.lambda_i[:,np.newaxis]
            shuffled_K_cross_star = np.array([(self.permutations[class1,class2,p]*self.lambda_i[class1] + (self.permutations[class1,class2,p]*self.lambda_i[class2]).T)/sum_lambda for p in range(n_permutations)])
            quantiles = np.nanpercentile(shuffled_K_cross_star, quantiles)
        else:
            quantiles = np.nanpercentile(self.permutations[class1, class2], quantiles)
        if variance_stabilization:
            return np.sqrt(quantiles/np.pi)
        return quantiles

    def get_diff_cross_function(self, class11, class12, class21, class22, **kwargs):
        return self.get_cross_function(class11, class12, **kwargs) - self.get_cross_function(class21, class22, **kwargs)


    def get_diff_pval(self, class11, class12, class21, class22, star=True, **kwargs):
        n_permutations = self.permutations.shape[2]
        if star:
            sum_lambda = self.lambda_i + self.lambda_i[:,np.newaxis]
            shuffled_K_cross_star1 = np.array([(self.permutations[class11,class12,p]*self.lambda_i[class11] + (self.permutations[class11,class12,p]*self.lambda_i[class12]).T)/sum_lambda for p in range(n_permutations)])
            shuffled_K_cross_star2 = np.array([(self.permutations[class21,class22,p]*self.lambda_i[class21] + (self.permutations[class21,class22,p]*self.lambda_i[class22]).T)/sum_lambda for p in range(n_permutations)])
            shuffled_K_cross_star = shuffled_K_cross_star1 -  shuffled_K_cross_star2
            diff = self.K_cross_star[class11,class12] - self.K_cross_star[class21,class22]
            p_val = (np.sum(shuffled_K_cross_star > diff) + 1.)/(n_permutations + 1.)
        else:
            shuffled_diff = self.permutations[class11, class12,:] - self.permutations[class21, class22,:]
            diff = self.K_cross[class11,class12] - self.K_cross[class21,class22]
            p_val = (np.sum(shuffled_diff > diff) + 1.)/(n_permutations + 1.)

        return p_val


    def get_diff_quantiles(self, class11, class12, class21, class22, star=True, quantiles=[2.5, 97.5], variance_stabilization=True, **kwargs):
        n_permutations = self.permutations.shape[2]

        if star:
            sum_lambda = self.lambda_i + self.lambda_i[:,np.newaxis]

            shuffled_K_cross_star1 = np.array([(self.permutations[class11,class12,p]*self.lambda_i[class11] + (self.permutations[class12,class11,p]*self.lambda_i[class12]))/sum_lambda[class11, class12] for p in range(n_permutations)])
            shuffled_K_cross_star2 = np.array([(self.permutations[class21,class22,p]*self.lambda_i[class21] + (self.permutations[class22,class21,p]*self.lambda_i[class22]))/sum_lambda[class21, class22] for p in range(n_permutations)])

            if variance_stabilization:
                shuffled_K_cross_star1 = np.sqrt(shuffled_K_cross_star1/np.pi)
                shuffled_K_cross_star2 = np.sqrt(shuffled_K_cross_star2/np.pi)

            shuffled_K_cross_star = shuffled_K_cross_star1 -  shuffled_K_cross_star2

            p0, p1 = np.nanpercentile(shuffled_K_cross_star, quantiles)

        else:

            if variance_stabilization:
                shuffled_K_cross_star1 = np.sqrt(self.permutations[class11, class12, :]/np.pi)
                shuffled_K_cross_star2 = np.sqrt(self.permutations[class21, class22, :]/np.pi)

            shuffled_diff = shuffled_K_cross_star1 -  shuffled_K_cross_star2
            p0, p1 = np.nanpercentile(shuffled_diff, quantiles)

        return p0, p1