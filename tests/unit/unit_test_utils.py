import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from utils import *


class TestGetNumClusters(unittest.TestCase):

    ### Test get_num_clusters ###
    def test__get_num_clusters_small(self):
        # 10k vectors
        num_clusters = get_num_clusters(num_vectors=10000)
        self.assertEqual(num_clusters, 200)
    
    def test__get_num_clusters_medium(self):
        # 1M vectors
        num_clusters = get_num_clusters(num_vectors=1000000)
        self.assertEqual(num_clusters, 6324)
    
    def test__get_num_clusters_large(self):
        # 100M vectors
        num_clusters = get_num_clusters(num_vectors=100000000)
        self.assertEqual(num_clusters, 200000)
    

class TestGetNProbe(unittest.TestCase):

    ### Test get_n_probe ###
    def test__get_n_probe_small(self):
        # 200 clusters, which corresponds to 10k vectors
        n_probe = get_n_probe(num_clusters=200)
        self.assertEqual(n_probe, 100)
    
    def test__get_n_probe_medium(self):
        # 6324 clusters, which corresponds to 1M vectors
        n_probe = get_n_probe(num_clusters=6324)
        self.assertEqual(n_probe, 445)
    
    def test__get_n_probe_large(self):
        # 200k clusters, which corresponds to 100M vectors
        n_probe = get_n_probe(num_clusters=200000)
        self.assertEqual(n_probe, 6000)
    

class TestGetTrainingMemoryUsage(unittest.TestCase):

    ### Test get_training_memory_usage ###
    def test__get_training_memory_usage(self):
        memory_usage = get_training_memory_usage(vector_dimension = 768, num_vectors = 100000)
        self.assertEqual(memory_usage, 921600000)


class TestGetNumBatches(unittest.TestCase):

    ### Test get_num_batches ###
    def test__get_num_batches(self):
        num_batches = get_num_batches(num_vectors = 1000000, vector_dimension = 768, max_memory_usage = 4*1024*1024*1024)
        self.assertEqual(num_batches, 3)


class TestDetermineOptimalTrainingMethod(unittest.TestCase):

    ### Test determine_optimal_training_method ###
    def test__determine_optimal_training_method_clustering(self):
        # 5M vectors
        method = determine_optimal_training_method(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 5000000)
        self.assertEqual(method, 'clustering')
    
    def test__determine_optimal_training_method_subsample(self):
        # 1M vectors
        method = determine_optimal_training_method(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 1000000)
        self.assertEqual(method, 'subsample')

