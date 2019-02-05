#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pysal
import cv2
import scipy
import scipy.spatial as spatial


class NeighborhoodMatrixComputation(object):
    
    def compute_neighborhood_matrix(self, 
                              neighborhood_matrix_type,
                              neighborhood_p0,
                              neighborhood_p1,
                              **kwargs):
        ''' Computes the neighborhood matrix from the label image as a pysal object. 
            Stores it in the dataframe neighborhood_matrix_df.

            :neighborhood_matrix_type: str
                                        should be 'k', 'radius', or 'network'

            :neighborhood_min_p0: int or float
                                  minimum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :neighborhood_min_p1: int or float
                                  maximum bound for the neighborhood.
                                  should be int for 'k' or 'network'. Can be int or float for 'radius' 
            :iterations: int. Optional
                         Only used for 'network' neighborhood computation mode.
                         Number of iterations of dilation to select the object neighbors from the label image.
                         Default value: 1

            :kd_tree_approx: boolean. Optional 
                             Used for 'radius' and 'k'.
                             If set to True, then use a kd-tree to find kNN. 
                             Else compute all pair distances.

            :save_neighbors: bool 
                             if True, will add a column to feature_table with the ids of neighbors
                             if False, do not keep the ids of neighbors
        '''

        if self._debug:
            print("\ncompute_neighborhood_matrix", neighborhood_matrix_type,
                                  neighborhood_p0,
                                  neighborhood_p1,
                                  kwargs)
        
        neighborhood_p0, neighborhood_p1, iterations = self._check_neighborhood_matrix_parameters(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

        try:
            w = self.get_neighborhood_matrix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)
            ## already computed
            return
        except ValueError:

            suffix = self.get_suffix(neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, **kwargs)

            if neighborhood_matrix_type == 'k':
                neighbor_dict = self._compute_matrix_k(suffix, 
                                                       neighborhood_p0, 
                                                       neighborhood_p1,
                                                       **kwargs)

            elif neighborhood_matrix_type == 'radius':
                neighbor_dict = self._compute_matrix_radius(suffix, 
                                                            neighborhood_p0, 
                                                            neighborhood_p1,
                                                            **kwargs)

            elif neighborhood_matrix_type == 'network':
                neighbor_dict = self._compute_matrix_network(suffix, 
                                                             neighborhood_p0, 
                                                             neighborhood_p1,
                                                             **kwargs)

            neighborhood_matrix = pysal.lib.weights.weights.W(neighbor_dict)
            nb_pairs = np.sum(neighborhood_matrix.full()[0])/2

            self.neighborhood_matrix_df.loc[self.neighborhood_matrix_df.shape[0],:] = [neighborhood_matrix_type, neighborhood_p0, neighborhood_p1, iterations, \
                neighborhood_matrix, \
                nb_pairs]



    def _compute_matrix_k(self, 
                          suffix, 
                          neighborhood_p0, 
                          neighborhood_p1,
                          save_neighbors=True,
                          kd_tree_approx=False,
                          **kwargs):
        if self.NN is None:
            self._compute_NN(kd_tree_approx)

        NN = self.NN[:,neighborhood_p0:neighborhood_p1+1]
        
        NN = [[self.feature_table.iloc[l][self._column_objectnumber] for l in line] for line in NN]

        self.feature_table.loc[:, 'DistanceLastNeighbor_{}'.format(suffix)] = self.NN_distances[:,neighborhood_p1]

        # if save_neighbors:
        #     self.feature_table['neighbors_{}'.format(suffix)] = np.empty((self.n, 0)).tolist()
            
            
        if neighborhood_p0 == 0:
            neighbor_dict = {line[0]: line[1:] for line in NN if len(line) > 1}
            
            NN_for_df = [line[1:] for line in NN]
            if save_neighbors:
                self.feature_table['neighbors_{}'.format(suffix)] = np.empty((self.n, 0)).tolist()
                self.feature_table.loc[:, 'neighbors_{}'.format(suffix)] = pd.Series(NN_for_df)
        
        else:
            if save_neighbors:
                self.feature_table.loc[:, 'neighbors_{}'.format(suffix)] = pd.Series(NN)
        
            ObjNum = self.feature_table[self._column_objectnumber].values
            neighbor_dict = {obj_num: neighbors for (obj_num, neighbors) in zip(ObjNum, NN) if len(neighbors)>0}
        
        return neighbor_dict


    def _compute_matrix_radius(self, 
                               suffix,
                               neighborhood_p0,
                               neighborhood_p1,
                               save_neighbors=True,
                               kd_tree_approx=False,
                               **kwargs):
        ## keep neighborhood_p0 <= r <= neighborhood_p1
        if self.NN is None:
            self._compute_NN(kd_tree_approx)

        mask = (self.NN_distances > neighborhood_p1)+(self.NN_distances < neighborhood_p0)
        mask[:,0] = True
        
        NN = np.array(self.NN)
        NN[mask] = -1
        NN = [[l for l in line if l != -1] for line in NN]
        
        NN = [[self.feature_table.iloc[l][self._column_objectnumber] for l in line if l != -1] for line in NN]

        NN_distances = [[l for l in line if (l < neighborhood_p1) and (l > neighborhood_p0)] for line in self.NN_distances]
        self.feature_table.loc[:, 'NumberNeighbors_{}'.format(suffix)] = np.sum(~mask, axis=1)

        if save_neighbors:
            self.feature_table['neighbors_{}'.format(suffix)] = np.empty((self.n, 0)).tolist()
            self.feature_table.loc[:, 'neighbors_{}'.format(suffix)] = pd.Series(NN)

        ObjNum = self.feature_table[self._column_objectnumber].values
        neighbor_dict = {obj_num: neighbors for (obj_num, neighbors) in zip(ObjNum, NN) if len(neighbors)>0}
        # print(neighbor_dict)
        return neighbor_dict


    def _compute_matrix_network(self, 
                                suffix,
                                neighborhood_p0,
                                neighborhood_p1,
                                save_neighbors=True,
                                iterations=1,
                                **kwargs):

        def custom_pow(firstM, lastM, n):
            if n > 1:
                mylist = custom_pow(firstM, lastM, n-1)
                mylist.append(np.dot(mylist[-1], firstM))
                return mylist
            else:
                return [np.dot(firstM,lastM)]

        def matrix_treatment(M):
            np.fill_diagonal(M,0)
            M[M>1] = 1
            return M

        ## actual neighbors
        ## labels start at 1
        if iterations not in self.adjacency_matrix:
            self._get_neighbors_from_label_image(self.get_suffix('network', 0, 1, iterations=iterations), 
                                                 iterations=iterations, save_neighbors=save_neighbors,
                                                 **kwargs)

        ## compute the needed power matrices
        if len(self.adjacency_matrix[iterations]) < neighborhood_p1:
            last_adj_mat = self.adjacency_matrix[iterations][-1]
            first_adj_mat = self.adjacency_matrix[iterations][0]

            new_power_matrices = custom_pow(first_adj_mat, \
                    last_adj_mat, 
                    neighborhood_p1 - len(self.adjacency_matrix[iterations]))

            new_power_matrices[:] = map(matrix_treatment, new_power_matrices)
            self.adjacency_matrix[iterations].extend(new_power_matrices)

        obj_nums = self.feature_table[self._column_objectnumber].values
        ### real network
        if neighborhood_p1 == 1: 
            list_where = np.where(self.adjacency_matrix[iterations][0])

        ### extended network
        else:
            cumsum_mat = self.adjacency_matrix[iterations][:neighborhood_p0]
            cumsum_mat = np.sum(cumsum_mat, axis=0)

            cumsum_mat2 = self.adjacency_matrix[iterations][neighborhood_p0:neighborhood_p1]
            cumsum_mat2 = np.sum(cumsum_mat2, axis=0)
            w = cumsum_mat2 - cumsum_mat
            w[w<0] = 0

            list_where = np.where(w)

        neighbor_dict = {}
        # print(list_where)
        for key, value in zip(list_where[0], list_where[1]):
            neighbor_dict.setdefault(obj_nums[key], []).append(obj_nums[value])

        if neighborhood_p1 != 1:
            ## neighborhood_p1 == 1 done by _get_neighbors_from_label_image
            # index_for_series = self.feature_table.index[self.feature_table[self._column_objectnumber].isin(neighbor_dict.keys())]
            # correct_keys = self.feature_table[self._column_objectnumber].isin(neighbor_dict.keys())
            # print(len(neighbor_dict.keys()), np.sum(correct_keys), w.shape)
            self.feature_table.loc[:, 'NumberNeighbors_{}'.format(suffix)] = np.sum(w, axis=0)

            if save_neighbors:
                self.feature_table.loc[:, 'neighbors_{}'.format(suffix)] = pd.Series([obj_nums[list_where[1][list_where[0] == index]] for index in range(self.n)])

        return neighbor_dict




    def _compute_NN(self, kd_tree_approx=False):
        coordinates = self.feature_table.loc[:,self._column_x_y].values

        if kd_tree_approx:
            tree = spatial.KDTree(coordinates)
            NN_distances, NN = tree.query(coordinates, self.n)

        else: 
            ## no approximation, compute all distances
            distance_bw_points = spatial.distance.cdist(coordinates, coordinates)
            NN = np.argsort(distance_bw_points, axis=1)
            xs = np.tile(np.arange(self.n), self.n).reshape((self.n, self.n)).T
            NN_distances = distance_bw_points[xs,NN]

        self.NN = NN
        self.NN_distances = NN_distances


    def _get_neighbors_from_label_image(self, 
                                        suffix, 
                                        iterations=1, 
                                        save_neighbors=True,
                                        **kwargs):
        ''' From the image_label, find all_neighbors
            Assumes that the labels on the image are the same as in the feature table.

            on the label image, 0 is bg
            objectNumbers start at 1
        '''
        
        labels = np.unique(self.feature_table[self._column_objectnumber].values)
        #np.sort(list(set(np.unique(self.image_label[self.image_label > 0].flatten())).intersection(np.unique(self.feature_table[self._column_objectnumber]))))

        if self._debug:
            print("_get_neighbors_from_label_image\n#labels = {}; #objects = {}, starting label id = {}, iterations={}".format(len(labels), self.n, min(labels), iterations))

        # self.feature_table['neighbors_{}'.format(suffix)] = np.empty((self.n, 0)).tolist()
        # self.feature_table['NumberNeighbors_{}'.format(suffix)] = 0

        sum_neighbors = []
        if save_neighbors:
            list_neighbors = []

        adj_mat = np.zeros((self.n, self.n))
        if self._debug:
            print("_get_neighbors_from_label_image", adj_mat.shape)
        for index_l, l in enumerate(labels):
            list_neighbor, sum_neighbor = self._get_label_neighbors(l, iterations)
            sum_neighbors.append(sum_neighbor)
            if save_neighbors:
                list_neighbors.append(list_neighbor)
            if len(list_neighbor) > 0:
                adj_mat[index_l, np.isin(labels, list_neighbor)] = 1

        self.adjacency_matrix[iterations] = [adj_mat]

        if self._debug:
            print("_get_neighbors_from_label_image", suffix)
            
        self.feature_table.loc[self.feature_table[self._column_objectnumber].isin(labels), 'NumberNeighbors_{}'.format(suffix)] = sum_neighbors

        if save_neighbors:
            index_for_series = self.feature_table.index[self.feature_table[self._column_objectnumber].isin(labels)]
            self.feature_table.loc[self.feature_table[self._column_objectnumber].isin(labels), 'neighbors_{}'.format(suffix)] = pd.Series(list_neighbors, index=index_for_series)


    def _get_label_neighbors(self, 
                                label,
                                iterations):
        
        binary = np.zeros(self.image_label.shape)
        binary[self.image_label == label] = 1

        dilated = cv2.dilate(binary, np.ones((3,3)), iterations=iterations)
        neighbor = np.unique(self.image_label[dilated == 1])

        ## remove the picked cells itself
        index = np.argwhere(neighbor == label)
        neighbor = np.delete(neighbor, index)

        ## remove background as neighbor
        index = np.argwhere(neighbor <= 0)
        neighbor = np.delete(neighbor, index)

        ## remove neighbors that are not in self.feature_table
        neighbor_mask = np.isin(neighbor, self.feature_table[self._column_objectnumber].values)
        list_neighbor = list(neighbor[neighbor_mask])
        sum_neighbor = np.sum(neighbor_mask)
        return list_neighbor, sum_neighbor