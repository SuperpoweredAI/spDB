import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from input_validation import *


class TestNameInputParameters(unittest.TestCase):

    valid_database_names = [
        "test_db",
        "test-db-2"
        "Test db"
    ]

    invalid_database_names = [
        "test.db",
        "#test_db"
        "test/db"
    ]

    ### Test valid database names ###
    def test__validate_database_name__valid_names(self):
        for database_name in self.valid_database_names:
            is_valid, _ = validate_database_name(database_name)
            self.assertTrue(is_valid)
    
    ### Test invalid database names ###
    def test__validate_database_name__invalid_names(self):
        for database_name in self.invalid_database_names:
            is_valid, _ = validate_database_name(database_name)
            self.assertFalse(is_valid)


class TestTrainInputParameters(unittest.TestCase):

    # vector_dimension, pca, opq, pq
    valid_train_parameters = [
        (768, 256, 128, 32),
        (1024, 256, 256, 64),
        (512, 512, 256, 64),
        (768, None, None, None),
    ]

    invalid_train_parameters = [
        (None, 128, 100, 32, "No vectors have been added to the database"),
        (768, None, 128, None, "compressed_vector_bytes must be set if opq_dimension is set"),
        (768, 128.3, 100, 32, "pca_dimension is not the correct type. Expected type: int. Actual type"),
        (768, '128', 100, 32, "pca_dimension is not the correct type. Expected type: int. Actual type"),
        (768, 1024, 128, 32, "pca_dimension is larger than the number of columns in the data. Number of columns in data"),
        (768, 128, 256, 32, "opq_dimension is larger than pca_dimension"),
        (768, 128, 100, 32, "opq_dimension is not divisible by compressed_vector_bytes")
    ]

    def test__validate_train__valid_parameters(self):
        for vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes in self.valid_train_parameters:
            is_valid, _ = validate_train(vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
            self.assertTrue(is_valid)

    def test__validate_train__invalid_parameters(self):
        for vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes, expected_reason in self.invalid_train_parameters:
            is_valid, reason = validate_train(vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)


class TestAddInputParameters(unittest.TestCase):

    # Create a valid numpy array, an invalid one, and a text list
    vector_array = np.random.rand(10, 768)
    invalid_array = np.random.rand(10, 512)
    text = ["test"] * 10

    valid_add_parameters = [
        (vector_array, text, 768),
        (vector_array, text, None),
    ]

    invalid_add_parameters = [
        ([1, 2, 3], text, 768, "Vectors are not the correct type. Expected type: numpy array. Actual type"),
        (invalid_array, text, 768, "Vector is not the correct size. Expected size"),
        (vector_array, text[0:5], 768, "Number of vectors does not match number of text items. Number of vectors"),
    ]

    def test__validate_add__valid_parameters(self):
        for vectors, text, vector_dimension in self.valid_add_parameters:
            is_valid, _ = validate_add(vectors, text, vector_dimension)
            self.assertTrue(is_valid)

    def test__validate_add__invalid_parameters(self):
        for vectors, text, vector_dimension, expected_reason in self.invalid_add_parameters:
            is_valid, reason = validate_add(vectors, text, vector_dimension)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)


class TestRemoveInputParameters(unittest.TestCase):

    valid_remove_parameters = [
        np.random.randint(0, 100, 10)
    ]

    invalid_remove_parameters = [
        (np.array([1.2, 2.3, 3.4, 4.5, 5.6]), "IDs are not integers"),
        (np.array([-1, -2, 0, 1, 2]), "Negative IDs found. All IDs must be positive"),
        (np.random.randint(0, 100, (10, 768)), "IDs are not 1D.")
    ]

    def test__validate_remove__valid_parameters(self):
        for ids in self.valid_remove_parameters:
            is_valid, _ = validate_remove(ids)
            self.assertTrue(is_valid)

    def test__validate_remove__invalid_parameters(self):
        for ids, expected_reason in self.invalid_remove_parameters:
            is_valid, reason = validate_remove(ids)
            self.assertFalse(is_valid)
            self.assertTrue(expected_reason in reason)
