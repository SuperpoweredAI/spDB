""" Testing thread safety of spDB faiss operations """

import logging
import numpy as np
import os
import pickle
import shutil
import sys
import threading
import time
import unittest


logger = logging.getLogger(__name__)


# get the absolute file path of this file
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

from spdb.spdb import spDB
from spdb import lmdb_utils

lmdb_utils.MAP_SIZE = 10 * 1024 * 1024 * 1024  # set testing map size to 10 GB

def get_test_data() -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:

    # Get the vectors
    with open(FILE_PATH + '/../data/fiqa_vectors.pickle', 'rb') as handle:
        vectors = pickle.load(handle)
    
    # Get the query data
    with open(FILE_PATH + '/../data/fiqa_queries.pickle', 'rb') as handle:
        queries = pickle.load(handle)
    
    # Get the text data
    with open(FILE_PATH + '/../data/fiqa_text.pickle', 'rb') as handle:
        text = pickle.load(handle)
    
    # Get the ground truths
    with open(FILE_PATH + '/../data/fiqa_ground_truths.pickle', 'rb') as handle:
        ground_truths = pickle.load(handle)

    return vectors, text, queries, ground_truths



def clean_up(db_path: str):

    # Delete the folders
    shutil.rmtree(db_path)


class TestFullSpdbEvaluation(unittest.TestCase):

    def setUp(self):
        self.db_name = "test_db"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True # This speeds up the test with a very small performance hit
        self.query_k = 100
        self.gt_k = 50

    def test__full_eval(self):
        # define helper functions for concurrent testing
        def add_async():
            db.add(long_vectors, long_text)

        def query_async():
            logger.info(f'starting short query after waiting {expected_add_time} seconds for lmdb adds to complete and long add to start')
            time.sleep(expected_add_time)  # simulate a slight delay before querying
            start_time = time.perf_counter()
            logger.info(f'starting short query, but expecting to be locked from long faiss.add()')
            result = db.query(queries[0], self.query_k, self.gt_k)
            end_time = time.perf_counter()
            time_taken = end_time - start_time

            logger.info(f'time taken for single query (initiated after long faiss.add()): {time_taken}')
            self.assertGreaterEqual(time_taken, expected_add_time, "query() should take at least as long as add()")

        # Get the test data from the pickled files
        vectors, text, queries, ground_truths = get_test_data()

        long_vectors = np.tile(vectors, (10, 1))
        long_text = text * 10

        # create the database
        db = spDB(self.db_name)

        # add the data
        db.add(vectors, text)

        # Train the index
        db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq, num_clusters=20000)

        ##################
        # THREAD SAFETY TEST
        ##################
        # Expected time for add() to complete.
        expected_add_time = 10
        # create and start add and query threads
        add_thread = threading.Thread(target=add_async)
        query_thread = threading.Thread(target=query_async)
        
        add_thread.start()
        query_thread.start()

        # wait for both threads to finish execution
        add_thread.join()
        query_thread.join()

