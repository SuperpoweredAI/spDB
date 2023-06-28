from fastapi import FastAPI
from fastapi.testclient import TestClient # requires httpx
import sys
import os
import numpy as np
import time
import unittest

from helpers import fiqa_test_data

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from api.fastapi import app


def evaluate(client, db_name: str, queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int):
    start_time = time.time()
    all_unique_ids = []
    total_sum = 0
    for i in range(queries.shape[0]):
        response = client.post(f"/db/{db_name}/query", json={"query_vector": queries[i].tolist(), "preliminary_top_k": query_k, "final_top_k": gt_k})
        reranked_I = np.array(response.json()['ids'])
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]

    return recall, latency, all_unique_ids


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.db_name = "fiqa_test"
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.query_k = 500
        self.gt_k = 50

        vectors, text, queries, ground_truths = fiqa_test_data()
        self.vectors = vectors.tolist()
        self.text = text
        self.queries = queries
        self.ground_truths = ground_truths


    def test__001_create(self):
        # Create a new database
        response = self.client.post("/db/create", json={"name": self.db_name})
        print (response.text)
        self.assertTrue(response.status_code == 200)

    def test__002_add(self):
        # Add vectors to the index
        batch_size = 100
        for i in range(0, len(self.vectors), batch_size):
            print (i)
            data = []
            for j in range(i, i+batch_size):
                data.append((
                    self.vectors[j],
                    {"text": self.text[j]}
                ))
            response = self.client.post(f"/db/{self.db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

    def test__003_train(self):
        # Train the index, using 2 level clustering
        response = self.client.post(f"/db/{self.db_name}/train", json={
            "use_two_level_clustering": True,
            "pca_dimension": self.pca_dimension,
            "opq_dimension": self.opq_dimension,
            "compressed_vector_bytes": self.compressed_vector_bytes,
            "omit_opq": True
        })
        self.assertTrue(response.status_code == 200)

    def test__004_query(self):
        # Test a single query
        response = self.client.post(f"/db/{self.db_name}/query", json={"query_vector": self.queries[0].tolist()})
        self.assertTrue(response.status_code == 200)

    def test__005_full_eval(self):
        # Run a full evaluation. This will tell us if everything is working properly
        recall, latency, all_unique_ids = evaluate(
            self.client, self.db_name, self.queries, self.ground_truths, self.query_k, self.gt_k
        )

        # Set the recall cutoff at above 0.97 and less than 1
        # If recall is above 1, something went wrong
        self.assertGreater(recall, 0.97)
        self.assertLessEqual(recall, 1)

        # Make sure latency is less than 25ms (higher cutoff than the other test since there's an http request)
        self.assertLess(latency, 25)

        # Make sure the length of each unique ID list is equal to the gt_k
        self.assertTrue(all([len(x) == self.gt_k for x in all_unique_ids]))

    def test__006_tear_down(self):
        response = self.client.post(f"/db/{self.db_name}/delete")
        assert response.status_code == 200
