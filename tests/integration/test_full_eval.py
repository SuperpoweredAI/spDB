""" Full end to end evaluation of the spDB using the fiqa Beir dataset """

import numpy as np
import time
import unittest

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import helpers

from spdb.spdb import spDB
from spdb import lmdb_utils

lmdb_utils.MAP_SIZE = 10 * 1024 * 1024 * 1024  # set testing map size to 10 GB


def evaluate(db, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int) -> tuple[float, float, list]:

    start_time = time.time()
    all_unique_ids = []
    all_cosine_similarity = []
    total_sum = 0
    for i in range(queries.shape[0]):
        results = db.query(queries[i], query_k, gt_k)
        reranked_I = results["ids"]
        cosine_similarity = results["cosine_similarity"]

        all_cosine_similarity.append(cosine_similarity)
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]

    return recall, latency, all_unique_ids, all_cosine_similarity


class TestFullSpdbEvaluation(unittest.TestCase):

    def setUp(self):
        self.db_name = "full_eval_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True # This speeds up the test with a very small performance hit
        self.query_k = 500
        self.gt_k = 50
        self.db = spDB(self.db_name)
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
        data = [(self.vectors[i], {"text": self.text[i]}) for i in range(len(self.vectors))]
        self.db.add(data)
        self.db.train(True, self.pca_dimension, self.opq_dimension, self.compressed_vector_bytes, self.omit_opq)


    def tearDown(self):
        self.db.delete()


    def test__full_eval(self):
        # Train the index

        # Evaluate the index
        recall, latency, all_unique_ids, all_cosine_similarity = evaluate(
            self.db, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Make sure cosine similarity is between 0 and 1
        # all_cosine_similarity is a list of lists
        all_cosine_similarity = [item for sublist in all_cosine_similarity for item in sublist]
        self.assertTrue(all([x >= 0 and x <= 1 for x in all_cosine_similarity]))

        # Set the recall cutoff at 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertGreater(recall, 0.97)
        self.assertLessEqual(recall, 1)

        # Make sure latency is less than 15ms (this includes the re-ranking from disk step)
        self.assertLess(latency, 15)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))

        # Retrain the index, but without two level clustering
        self.db.train(False)

if __name__ == '__main__':
    unittest.main()