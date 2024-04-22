from fastapi.testclient import TestClient # requires httpx
import time
import unittest
import os
import sys
import random
import string
import numpy as np
import json

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from helpers import fiqa_test_data

from api.fastapi import app


def generate_random_vectors_with_text(N, D):
    random_vectors = np.random.rand(N, D).astype(np.float32) 
    random_text = [''.join(random.choices(string.ascii_lowercase, k=D)) for _ in range(N)]
    return random_vectors, random_text


class TestAutoTrain(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.db_names = ["fiqa_test_1", "fiqa_test_2"]
        self.pca_dimension = 256
        self.opq_dimension = 128
        self.compressed_vector_bytes = 32
        self.omit_opq = True
        self.query_k = 500
        self.gt_k = 50

        vectors, text, _, _ = fiqa_test_data()
        print (len(vectors), len(text))
        self.vectors = vectors.tolist()
        self.text = text

        # Specify the number of random vectors (N) and the dimensionality (D)
        #N = 30000  # Number of random vectors
        #D = 2048  # Dimensionality of each vector

        # Generate N random vectors with D dimensions and random text strings
        #random_vectors, random_text = generate_random_vectors_with_text(N, D)
        #self.vectors = random_vectors.tolist()
        #self.text = random_text
    

    def test__001_setup_dbs(self):
        # Create a few databases and add vectors to them

        vectors = self.vectors
        text = self.text

        for db_name in self.db_names:
            response = self.client.post("/db/create", json={"name": db_name})
            self.assertTrue(response.status_code == 200)

            # Add vectors to the database
            batch_size = 1000
            for i in range(0, len(vectors), batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((vectors[j], {"text": text[j]}))
                response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)

        # View the cache
        response = self.client.get("/db/view_cache")
        print (response.json())
        cache_keys = response.json()["cache_keys"]
        self.assertTrue(len(cache_keys) == 2)

        db_name = self.db_names[0]
    

    def test__002_train(self):

        # Train the first db, then check the info of the db
        db_name = self.db_names[0]
        response = self.client.post(f"/db/{db_name}/train", json={
            "use_two_level_clustering": True,
            "pca_dimension": self.pca_dimension,
            "opq_dimension": self.opq_dimension,
            "compressed_vector_bytes": self.compressed_vector_bytes,
            "omit_opq": self.omit_opq
        })

        # Wait for the training to finish
        for i in range(20):
            response = self.client.get(f"/db/{db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            time.sleep(10)
        
        response = self.client.get(f"/db/{db_name}/info")
        db_info  = response.json()["db_info"]
        # Convert the db info from a string to a dictionary
        db_info = json.loads(db_info)
        n_total = db_info["n_total"]
        self.assertTrue(n_total == 30000)

        # View the cache
        response = self.client.get("/db/view_cache")
        print (response.json())
        current_memory_usage = response.json()["current_memory_usage"]

        # Make sure the memory usage is less than 10MB
        # If it's higher, then the cache memory wouldn't have updated the after training
        print ("current_memory_usage", current_memory_usage)
        self.assertTrue(current_memory_usage < (100 * 1024 * 1024))
    

    def test__003_auto_remove_cache(self):

        # Create 2 more DBs and add 30,000 vectors to each in order to get above 200MB
        new_db_names = ["fiqa_test_3", "fiqa_test_4"]
        for db_name in new_db_names:
            response = self.client.post("/db/create", json={"name": db_name})
            self.assertTrue(response.status_code == 200)

            # Add vectors to the database
            batch_size = 1000
            for i in range(0, len(self.vectors), batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((self.vectors[j], {"text": self.text[j]}))
                response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)
        
        # View the cache
        response = self.client.get("/db/view_cache")
        # The fiqa_test_1 db should have been removed from the cache
        cache_keys = response.json()["cache_keys"]
        print ("cache_keys", cache_keys)
        self.assertTrue(len(cache_keys) == 3)
        # Make sure the fiqa_test_2 db is not in the cache
        self.assertTrue("fiqa_test_2" not in cache_keys)
    
    def test__004_remove_from_cache(self):
        # Just testing removing a database from the cache
        
        db_name = self.db_names[0]
        response = self.client.post(f"/db/{db_name}/remove_from_cache")

        # View the cache
        response = self.client.get("/db/view_cache")
        print (response.json())
        cache_keys = response.json()["cache_keys"]
        self.assertTrue(len(cache_keys) == 2)
        print (cache_keys)
        # Make sure the fiqa_test_2 db is not in the cache
        self.assertTrue("fiqa_test_1" not in cache_keys)


    def test__005_tear_down(self):
        for db_name in self.db_names:
            response = self.client.post(f"/db/{db_name}/delete")

if __name__ == "__main__":
    unittest.main()