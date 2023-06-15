""" Testing thread safety of spDB faiss operations """

import logging
import numpy as np
import threading
import time
import unittest

import helpers


from spdb.spdb import spDB
from spdb import lmdb_utils

lmdb_utils.MAP_SIZE = 10 * 1024 * 1024 * 1024  # set testing map size to 10 GB


logger = logging.getLogger(__name__)


class TestThreadSafety(unittest.TestCase):

    def setUp(self):
        self.db_name = "test_db"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True # This speeds up the test with a very small performance hit
        self.query_k = 500
        self.gt_k = 50
        self.db = spDB(self.db_name)
        self.vectors, self.text, self.queries, self.ground_truths = helpers.faiq_test_data()
        self.db.add(self.vectors, self.text)
        self.db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq, num_clusters=20_000)

    def tearDown(self):
        self.db.delete()

    def test__full_eval(self):
        # define helper functions for concurrent testing
        def add_async():
            self.db.add(long_vectors, long_text)

        def query_async():
            logger.info(f'starting short query after waiting {expected_add_time} seconds for lmdb adds to complete and long add to start')
            time.sleep(expected_add_time)  # simulate a slight delay before querying
            start_time = time.perf_counter()
            logger.info(f'starting short query, but expecting to be locked from long faiss.add()')
            result = self.db.query(self.queries[0], self.query_k, self.gt_k)
            end_time = time.perf_counter()
            time_taken = end_time - start_time

            logger.info(f'time taken for single query (initiated after long faiss.add()): {time_taken}')
            self.assertGreaterEqual(time_taken, expected_add_time, "query() should take at least as long as add()")

        long_vectors = np.tile(self.vectors, (10, 1))
        long_text = self.text * 10

        # log expected query time for short query
        t0 = time.perf_counter()
        self.db.query(self.queries[0], self.query_k, self.gt_k)
        t1 = time.perf_counter()
        expected_query_time = t1 - t0
        logger.info(f'time to complete short query without blocking thread: {expected_query_time}')

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

