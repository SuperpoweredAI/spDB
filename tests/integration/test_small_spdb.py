""" Test for a small spDB """

import numpy as np
import unittest

import helpers

from spdb.spdb import spDB


def evaluate(db, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int) -> tuple[float, list]:

    all_unique_ids = []
    all_cosine_similarity = []
    total_sum = 0
    for i in range(queries.shape[0]):
        _, reranked_I, cosine_similarity = db.query(queries[i], query_k, gt_k)
        all_cosine_similarity.append(cosine_similarity)
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    recall = total_sum / ground_truths.shape[0]

    return recall, all_unique_ids, all_cosine_similarity


class TestSmallSpdbEvaluation(unittest.TestCase):

    def setUp(self):
        self.db_name = "small_spdb_test"
        self.query_k = 500
        self.gt_k = 50
        self.db = spDB(self.db_name)
        self.vectors, self.text, self.queries, self.ground_truths = helpers.fiqa_test_data()
    
    def test__small_eval(self):
        vectors = self.vectors[0:2500]
        text = self.text[0:2500]
        # Add a subset of the vectors
        self.db.add(vectors, text)
        # Train the index
        self.db.train(False)
        # Make sure the vectors are in the index
        self.assertTrue(self.db.faiss_index.ntotal, 2500)

        vectors = self.vectors[2500:]
        text = self.text[2500:]
        # Add the rest of the vectors
        self.db.add(vectors, text)

        recall, all_unique_ids, all_cosine_similarity = evaluate(
            self.db, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Make sure cosine similarity is between 0 and 1
        # all_cosine_similarity is a list of lists, so we need to flatten it
        all_cosine_similarity = [item for sublist in all_cosine_similarity for item in sublist]
        self.assertTrue(all([x >= 0 and x <= 1 for x in all_cosine_similarity]))

        # Recall should be 1.0
        self.assertGreaterEqual(recall, 0.999)
        self.assertLessEqual(recall, 1.001)

        # Make sure the unique ids are the same length as the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))

        self.db.delete()
