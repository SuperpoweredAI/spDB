import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from utils import *


class TestGetNumClusters(unittest.TestCase):

    test_cases = [
        (10000, 200),
        (1000000, 6324),
        (100000000, 200000),
    ]
    def test__get_num_clusters(self):
        for num_vectors, expected_num_clusters in self.test_cases:
            with self.subTest(num_vectors=num_vectors, expected_num_clusters=expected_num_clusters):
                num_clusters = get_num_clusters(num_vectors)
                self.assertEqual(num_clusters, expected_num_clusters)


class TestGetNProbe(unittest.TestCase):

    test_cases = [
        (200, 100),
        (1000, 250),
        (6350, 444),
        (200000, 6000),
    ]
    def test__get_n_probe(self):
        for num_clusters, expected_n_probe in self.test_cases:
            with self.subTest(num_clusters=num_clusters, expected_n_probe=expected_n_probe):
                n_probe = get_n_probe(num_clusters)
                self.assertEqual(n_probe, expected_n_probe)
    

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
    def test__determine_optimal_training_method__clustering(self):
        # 5M vectors
        method = determine_optimal_training_method(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 5000000)
        self.assertEqual(method, 'two_level_clustering')
    
    def test__determine_optimal_training_method__subsampling(self):
        # 1M vectors
        method = determine_optimal_training_method(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 1000000)
        self.assertEqual(method, 'subsampling')

