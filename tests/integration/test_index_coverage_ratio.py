import os
import sys
import unittest

import helpers

from spdb.spdb import spDB
from spdb import lmdb_utils


class TestIndexCoverageRatio(unittest.TestCase):

    def setUp(self):
        self.db_name = "index_coverage_ratio_eval_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
        self.db = spDB(self.db_name)
    
    def tear_down(self):
        self.db.delete()
    
    def test__index_coverage_ratio(self):

        # Test that the index coverage ratio is 0 before adding any vectors
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        # Test that the index coverage ratio is still 0 after adding vectors
        self.db.add(self.vectors, self.text)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        # Test that the index coverage ratio is 1 after training the index
        self.db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 1)

        # Test that the index coverage ratio is 0.5 after adding another set of vectors
        self.db.add(self.vectors, self.text)
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0.5)

        # Remove the first 30000 vectors (which is the length of the fiqa test data)
        ids = range(30000)
        self.db.remove(ids)
        # The index coverage ratio should be 0 now
        coverage_ratio = self.db.trained_index_coverage_ratio
        self.assertEqual(coverage_ratio, 0)

        # Delete the database
        self.tear_down()
