import unittest
import faiss

from spdb import utils

class TestGetNumClusters(unittest.TestCase):

    test_cases = [
        (10000, 200),
        (1000000, 6324),
        (100000000, 200000),
    ]
    def test__get_num_clusters(self):
        for num_vectors, expected_num_clusters in self.test_cases:
            with self.subTest(num_vectors=num_vectors, expected_num_clusters=expected_num_clusters):
                num_clusters = utils.get_num_clusters(num_vectors)
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
                n_probe = utils.get_n_probe(num_clusters)
                self.assertEqual(n_probe, expected_n_probe)


class TestGetTrainingMemoryUsage(unittest.TestCase):

    ### Test get_training_memory_usage ###
    def test__get_training_memory_usage(self):
        memory_usage = utils.get_training_memory_usage(vector_dimension = 768, num_vectors = 100000)
        self.assertEqual(memory_usage, 921600000)


class TestGetNumBatches(unittest.TestCase):

    ### Test get_num_batches ###
    def test__get_num_batches(self):
        num_batches = utils.get_num_batches(num_vectors = 1000000, vector_dimension = 768, max_memory_usage = 4*1024*1024*1024)
        self.assertEqual(num_batches, 3)


class TestDetermineOptimalTrainingMethod(unittest.TestCase):

    ### Test determine_optimal_training_method ###
    def test__is_two_level_clustering_optimal__clustering(self):
        # 5M vectors
        use_two_level_clustering = utils.is_two_level_clustering_optimal(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 5000000)
        self.assertEqual(use_two_level_clustering, True)
    
    def test__is_two_level_clustering_optimal__subsampling(self):
        # 1M vectors
        use_two_level_clustering = utils.is_two_level_clustering_optimal(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 1000000)
        self.assertEqual(use_two_level_clustering, False)


class TestCalculateTrainedIndexCoverageRatio(unittest.TestCase):

    # Create a list of 1000 vectors from 0 to 999
    lmdb_ids = range(1000)
    # Create a list of 100 vectors from 0 to 99
    saved_index_ids = range(100)

    ### Partial coverage ###
    def test__calculate_trained_index_coverage_ratio__with_saved_index(self):
        coverage_ratio = utils.calculate_trained_index_coverage_ratio(self.lmdb_ids, self.saved_index_ids)
        self.assertEqual(coverage_ratio, 0.1)
    
    ### Full coverage ###
    def test__calculate_trained_index_coverage_ratio__with_saved_index(self):
        coverage_ratio = utils.calculate_trained_index_coverage_ratio(self.saved_index_ids, self.lmdb_ids)
        self.assertEqual(coverage_ratio, 1)

    ### No saved ids (case where an index hasn't been trained yet) ###
    def test__calculate_trained_index_coverage_ratio__no_saved_index(self):
        coverage_ratio = utils.calculate_trained_index_coverage_ratio(self.lmdb_ids, [])
        self.assertEqual(coverage_ratio, 0)
    
    ### No lmdb ids (case where someone removed all vectors, but previously trained an index) ###
    def test__calculate_trained_index_coverage_ratio__no_lmdb_ids(self):
        coverage_ratio = utils.calculate_trained_index_coverage_ratio([], self.saved_index_ids)
        self.assertEqual(coverage_ratio, 0)
        use_two_level_clustering = utils.is_two_level_clustering_optimal(max_memory_usage = 4*1024*1024*1024, vector_dimension = 768, num_vectors = 1000000)
        self.assertEqual(use_two_level_clustering, False)


class TestCheckIsFlatIndex(unittest.TestCase):
    
    def test__check_is_flat_index__True(self):
        faiss_index = faiss.IndexFlat(768)
        faiss_index = faiss.IndexIDMap(faiss_index)
        print (type(faiss_index))
        is_index_flat = utils.check_is_flat_index(faiss_index)
        self.assertTrue(is_index_flat)
    
    def test__check_is_flat_index__False(self):
        faiss_index = faiss.index_factory(768, "PCA256,IVF4096,PQ32")
        is_index_flat = utils.check_is_flat_index(faiss_index)
        self.assertFalse(is_index_flat)
