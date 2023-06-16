from fastapi import FastAPI
from fastapi.testclient import TestClient # requires httpx
import sys
import os
import numpy as np
import time

from helpers import fiqa_test_data

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from api.fastapi import app


client = TestClient(app)

db_name = "fiqa_test"

def test_create():
    response = client.post("/db/create", json={"name": db_name})
    assert response.status_code == 200

def test_add(vectors, text):
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        print (i)
        response = client.post(f"/db/{db_name}/add", json={"vectors": vectors[i:i+batch_size], "text": text[i:i+batch_size]})
    assert response.status_code == 200

def test_train():
    response = client.post(f"/db/{db_name}/train", json={"use_two_level_clustering": True, "omit_opq": True})
    assert response.status_code == 200
    print (response.text)

def test_query(queries):
    response = client.post(f"/db/{db_name}/query", json={"query_vector": queries[0].tolist()})
    print (response.json().keys())
    assert response.status_code == 200

def test_full_eval(queries: np.ndarray, ground_truths: np.ndarray, query_k: int, gt_k: int):
    start_time = time.time()
    all_unique_ids = []
    total_sum = 0
    for i in range(queries.shape[0]):
        response = client.post(f"/db/{db_name}/query", json={"query_vector": queries[i].tolist(), "preliminary_top_k": query_k, "final_top_k": gt_k})
        reranked_I = np.array(response.json()['ids'])
        #_, reranked_I = db.query(queries[i], query_k, gt_k)
        # compute recall
        total_sum += sum([1 for x in reranked_I[:gt_k] if x in ground_truths[i, :gt_k]]) / gt_k
        unique_ids = np.unique(reranked_I)
        all_unique_ids.append(unique_ids)

    end_time = time.time()
    recall = total_sum / ground_truths.shape[0]
    latency = (end_time - start_time) * 1000 / queries.shape[0]
    print ("recall", recall)
    print ("latency", latency)

    # Set the recall cutoff at 0.97 and less than 1
    # If recall is above 1, something went wrong
    assert recall > 0.97
    assert recall < 1

    # Make sure latency is less than 25ms (higher cutoff than the other test since there's an http request)
    assert latency < 25

    # Make sure the length of each unique ID list is equal to the gt_k
    assert (all([len(x) == gt_k for x in all_unique_ids]))

def test_tear_down():
    response = client.post(f"/db/{db_name}/delete")
    assert response.status_code == 200


if __name__ == "__main__":

    vectors, text, queries, ground_truths = fiqa_test_data()
    vectors = vectors.tolist()

    test_create()
    test_add(vectors, text)
    test_train()
    test_query(queries)
    test_full_eval(queries, ground_truths, 500, 50)
    test_tear_down()

