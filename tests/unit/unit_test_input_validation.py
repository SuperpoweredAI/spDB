import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from input_validation import *


class TestInputParameters(unittest.TestCase):

    def test__validate_database_name_valid(self):
        db_name = 'test_db'
        is_valid, _ = validate_database_name(db_name)
        self.assertTrue(is_valid)
    
    def test__validate_database_name_invalid_period(self):
        db_name = 'test.db'
        is_valid, _ = validate_database_name(db_name)
        self.assertFalse(is_valid)
    
    def test__validate_database_name_invalid_character(self):
        db_name = 'test_db#'
        is_valid, _ = validate_database_name(db_name)
        self.assertFalse(is_valid)


    def test__validate_train_valid_inputs(self):
        is_valid, _ = validate_train(vector_dimension=768, pca_dimension=256, compressed_vector_bytes=32, opq_dimension=128)
        self.assertTrue(is_valid)

    def test__validate_train_invalid_pca(self):
        is_valid, reason = validate_train(vector_dimension=768, pca_dimension=1024, compressed_vector_bytes=32, opq_dimension=128)
        self.assertFalse(is_valid)
        # Make sure the reason is correct
        self.assertTrue('PCA is larger than the number of columns in the data. Number of columns in data' in reason)
    
    def test__validate_train_invalid_opq(self):
        is_valid, reason = validate_train(vector_dimension=768, pca_dimension=128, compressed_vector_bytes=32, opq_dimension=256)
        self.assertFalse(is_valid)
        # Make sure the reason is correct
        self.assertTrue('OPQ is larger than PCA' in reason)
