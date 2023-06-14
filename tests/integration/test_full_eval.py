""" Full end to end evaluation of the spDB using the fiqa Beir dataset """

import os
import sys
import shutil
import numpy as np
import unittest
import pickle
import time

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


def evaluate(db, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int) -> tuple[float, float, list]:

    start_time = time.time()
    all_unique_ids = []
    total_sum = 0
    for i in range(queries.shape[0]):
        _, reranked_I = db.query(queries[i], query_k, gt_k)
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]

    return recall, latency, all_unique_ids


def clean_up(db_path: str):

    # Delete the folders
    shutil.rmtree(db_path)


class TestFullSpdbEvaluation(unittest.TestCase):

    def setup(self):
        self.db_name = "test_db"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True # This speeds up the test with a very small performance hit
        self.query_k = 500
        self.gt_k = 50

    def test__full_eval(self):

        self.setup()

        # Get the test data from the pickled files
        vectors, text, queries, ground_truths = get_test_data()

        # create the database
        db = spDB(self.db_name)

        # add the data
        db.add(vectors, text)

        # Train the index
        db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq)

        # Evaluate the index
        recall, latency, all_unique_ids = evaluate(db, queries, ground_truths, self.query_k, self.gt_k)

        print ("recall", recall)

        # Delete the index, pickle file and folders
        clean_up(db.save_path)

        # Set the recall cutoff at 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertTrue(recall > 0.97)
        self.assertTrue(recall <= 1)

        # Make sure latency is less than 15ms (this includes the re-ranking from disk step)
        self.assertTrue(latency < 15)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))
