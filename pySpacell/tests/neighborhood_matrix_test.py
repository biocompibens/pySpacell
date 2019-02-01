#! /usr/bin/env pythn3

import unittest
from glob import glob
import numpy as np
import os
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pySpacell import Spacell

class NeighborhoodMatrixTest(unittest.TestCase):


	@classmethod
	def setUpClass(cls):
		pass

	@classmethod
	def tearDownClass(cls):
		pass

	def setUp(self):
		THIS_DIR = os.path.dirname(os.path.abspath(__file__))
		csv_file = glob(os.path.join(THIS_DIR, "../toy_dataset", "*.csv"))[0]
		label_file = glob(os.path.join(THIS_DIR, "../toy_dataset", "*_labels_*.tiff"))[0]
		self.spa = Spacell(csv_file, label_file,
                 column_objectnumber='ObjectNumber',
                 column_x_y=['Location_X', 'Location_Y'])
		self.spa._debug = True


	def tearDown(self):
		del self.spa


	def test_compute_neighborhood_matrix_radius_starting_0(self):
		radii = [100, 135.5, 300, 267.0907864735632546]
		for r in radii:
			### test compute function does not crash
			self.spa.compute_neighborhood_matrix('radius', 0, r, save_neighbors=True)
			suffix = self.spa.get_suffix('radius', 0, r)

			self.assertTrue('NumberNeighbors_{}'.format(suffix) in self.spa.feature_table.columns)
			self.assertTrue('neighbors_{}'.format(suffix) in self.spa.feature_table.columns)			

			### test if get_neighborhood_matrix is working
			w = self.spa.get_neighborhood_matrix('radius', 0, r, pysal_object=True)

			### test if pysal object is valid
			w_full = w.full()

			### test if correct neighbors
			pairs = np.random.choice(np.arange(self.spa.n), 100).reshape((50,2))
			for p in pairs:
				if p[0] == p[1]:
					continue
				coord1 = self.spa.feature_table.iloc[p[0]][self.spa._column_x_y].values
				coord2 = self.spa.feature_table.iloc[p[1]][self.spa._column_x_y].values
				d = cdist([coord1], [coord2])[0][0]

				obj1 = self.spa.feature_table.iloc[p[0]][self.spa._column_objectnumber]
				obj2 = self.spa.feature_table.iloc[p[1]][self.spa._column_objectnumber]
				if d > r:
					continue
				# print(d)
				# print(coord2, obj2, self.spa.feature_table.iloc[p[0]]['neighbors_{}'.format(suffix)])
				# print(coord1, obj1, self.spa.feature_table.iloc[p[1]]['neighbors_{}'.format(suffix)])
				self.assertTrue(obj2 in self.spa.feature_table.iloc[p[0]]['neighbors_{}'.format(suffix)])
				self.assertTrue(obj1 in self.spa.feature_table.iloc[p[1]]['neighbors_{}'.format(suffix)])


	def test_compute_neighborhood_matrix_radius(self):
		radii = [[100, 200], 
		         [135.5, 300], 
		         [267.0907864735632546, 289.950689465384]]
		for r in radii:
			### test compute function does not crash
			self.spa.compute_neighborhood_matrix('radius', r[0], r[1], save_neighbors=True)
			suffix = self.spa.get_suffix('radius', r[0], r[1])

			self.assertTrue('NumberNeighbors_{}'.format(suffix) in self.spa.feature_table.columns)
			self.assertTrue('neighbors_{}'.format(suffix) in self.spa.feature_table.columns)			

			### test if get_neighborhood_matrix is working
			w = self.spa.get_neighborhood_matrix('radius', r[0], r[1], pysal_object=True)

			### test if pysal object is valid
			w_full = w.full()

			### test if correct neighbors
			pairs = np.random.choice(np.arange(self.spa.n), 100).reshape((50,2))
			for p in pairs:
				if p[0] == p[1]:
					continue
				coord1 = self.spa.feature_table.iloc[p[0]][self.spa._column_x_y].values
				coord2 = self.spa.feature_table.iloc[p[1]][self.spa._column_x_y].values
				d = cdist([coord1], [coord2])[0][0]

				obj1 = self.spa.feature_table.iloc[p[0]][self.spa._column_objectnumber]
				obj2 = self.spa.feature_table.iloc[p[1]][self.spa._column_objectnumber]
				if d <= r[1] and d >= r[0]:
					self.assertTrue(obj2 in self.spa.feature_table.iloc[p[0]]['neighbors_{}'.format(suffix)])
					self.assertTrue(obj1 in self.spa.feature_table.iloc[p[1]]['neighbors_{}'.format(suffix)])


	def test_compute_neighborhood_matrix_k_starting_0(self):
		ks = [1,2,5,10]
		for k in ks:
			### test compute function does not crash
			self.spa.compute_neighborhood_matrix('k', 0, k, save_neighbors=True)
			suffix = self.spa.get_suffix('k', 0, k)

			self.assertTrue('DistanceLastNeighbor_{}'.format(suffix) in self.spa.feature_table.columns)
			self.assertTrue('neighbors_{}'.format(suffix) in self.spa.feature_table.columns)			

			### test if get_neighborhood_matrix is working
			w = self.spa.get_neighborhood_matrix('k', 0, k, pysal_object=True)

			### test if pysal object is valid
			w_full = w.full()

			### test if correct neighbors
			objs = np.random.choice(np.arange(self.spa.n), 100)
			for obj in objs:

				coord1 = self.spa.feature_table.iloc[obj][self.spa._column_x_y].values
				coords = self.spa.feature_table.loc[:,self.spa._column_x_y].values
				d = cdist([coord1], coords)[0]
				neighbors = np.argsort(d)[1:k+1]
				neighbors_obj = self.spa.feature_table.iloc[neighbors][self.spa._column_objectnumber].values

				self.assertEqual(len(self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]), k)
				self.assertTrue((neighbors_obj==self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]).all())


	def test_compute_neighborhood_matrix_k(self):
		ks = [1,2,5,10]
		for index_k1 in range(len(ks)-1):
			for index_k2 in range(index_k1+1,len(ks)):
				### test compute function does not crash
				self.spa.compute_neighborhood_matrix('k', ks[index_k1], ks[index_k2], save_neighbors=True)
				suffix = self.spa.get_suffix('k', ks[index_k1], ks[index_k2])

				self.assertTrue('DistanceLastNeighbor_{}'.format(suffix) in self.spa.feature_table.columns)
				self.assertTrue('neighbors_{}'.format(suffix) in self.spa.feature_table.columns)			
				### test if get_neighborhood_matrix is working
				w = self.spa.get_neighborhood_matrix('k', ks[index_k1], ks[index_k2], pysal_object=True)

				### test if pysal object is valid
				w_full = w.full()

				### test if correct neighbors
				objs = np.random.choice(np.arange(self.spa.n), 100)
				for obj in objs:

					coord1 = self.spa.feature_table.iloc[obj][self.spa._column_x_y].values
					coords = self.spa.feature_table.loc[:,self.spa._column_x_y].values
					d = cdist([coord1], coords)[0]
					neighbors = np.argsort(d)[ks[index_k1]:ks[index_k2]+1]
					neighbors_obj = self.spa.feature_table.iloc[neighbors][self.spa._column_objectnumber].values
					# print(ks[index_k1], ks[index_k2], neighbors_obj, self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)])
					self.assertEqual(len(self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]), ks[index_k2]-ks[index_k1]+1)
					self.assertTrue((neighbors_obj==self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]).all())

	def test_compute_neighborhood_matrix_network_starting_0(self):
		rounds = [1,2, 4, 6]
		for r in rounds:
			for iterations in range(1, 10, 2):
				### test compute function does not crash
				self.spa.compute_neighborhood_matrix('network', 0, r, save_neighbors=True, iterations=iterations)
				suffix = self.spa.get_suffix('network', 0, r, iterations=iterations)

				self.assertTrue('NumberNeighbors_{}'.format(suffix) in self.spa.feature_table.columns)
				self.assertTrue('neighbors_{}'.format(suffix) in self.spa.feature_table.columns)			

				### test if get_neighborhood_matrix is working
				print(iterations, self.spa.adjacency_matrix)
				w = self.spa.get_neighborhood_matrix('network', 0, r, pysal_object=True, iterations=iterations)

				### test if pysal object is valid
				w_full = w.full()

				### test if correct neighbors
				objs = np.random.choice(np.arange(self.spa.n), 100)
				for obj in objs:

					obj_id = self.spa.feature_table.iloc[obj][self.spa._column_objectnumber]

					if r == 1:
						im_obj = np.array(self.spa.image_label==obj_id, dtype=np.uint8)
						_, contours, hierarchy = cv2.findContours(im_obj, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
						# print(hierarchy, len(contours))
						contour_obj = contours[np.argmax([len(c) for c in contours])][:,0,:]

						for neighbor_id in self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]:
							im_neighbor = np.array(self.spa.image_label==neighbor_id, dtype=np.uint8)
							_, contours, hierarchy = cv2.findContours(im_neighbor, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
							# print(hierarchy, [len(c) for c in contours])
							contour_neighbor = contours[np.argmax([len(c) for c in contours])][:,0,:]

							d = np.nanmin(cdist(contour_obj, contour_neighbor, metric='cityblock'))
							self.assertTrue(d <= 2*iterations)
							# im_to_show = np.zeros(self.spa.image_label.shape)
							# im_to_show[self.spa.image_label==obj_id] = 1
							# im_to_show[self.spa.image_label==neighbor_id] = 2
							# ax1=plt.subplot(1,2,1)
							# plt.imshow(im_to_show)
							# plt.title(d)
							# plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
							# for c in contour_obj:
							# 	plt.plot(c[0], c[1], 'r+')
							# for c in contour_neighbor:
							# 	plt.plot(c[0], c[1], 'b+')
							# # plt.ylim(2000, 0)
							# plt.show()

					else:

						suffix_first_neighbors = self.spa.get_suffix('network', 0, 1, iterations=iterations)
						already_neighbors = set(self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix_first_neighbors)])
						previous_neighbors = set(already_neighbors)

						n_link = 1
						while n_link < r:
							new_neighbors = []
							for n in previous_neighbors:
								new_neighbors.extend(self.spa.feature_table.loc[self.spa.feature_table[self.spa._column_objectnumber] == n, 'neighbors_{}'.format(suffix_first_neighbors)].values[0])

							# print(new_neighbors, already_neighbors, previous_neighbors)
							new_neighbors = set(new_neighbors).difference(already_neighbors)
							previous_neighbors = set(new_neighbors)
							already_neighbors = already_neighbors.union(set(new_neighbors))
							n_link += 1

						already_neighbors = already_neighbors.difference(set([obj_id]))
						# print(already_neighbors, set(self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]))
						self.assertTrue(already_neighbors == set(self.spa.feature_table.iloc[obj]['neighbors_{}'.format(suffix)]))





if __name__ == '__main__':
    unittest.main()
