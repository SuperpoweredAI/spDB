from fastapi.testclient import TestClient # requires httpx
import sys
import os
import numpy as np
import time
import unittest
#import json

from helpers import fiqa_test_data

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../'))

from api.fastapi import app



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
    
    def test__001_setup_dbs(self):
        # Create a few databases and add vectors to them

        vectors = self.vectors
        text = self.text

        for db_name in self.db_names:
            response = self.client.post("/db/create", json={"name": db_name})
            assert response.status_code == 200

            # Add vectors to the database
            batch_size = 1000
            for i in range(0, len(vectors), batch_size):
                data = []
                for j in range(i, i+batch_size):
                    data.append((vectors[j], {"text": text[j]}))
                response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
            self.assertTrue(response.status_code == 200)
            
            response = self.client.get(f"/db/{db_name}/info")
            db_info  = response.json()["db_info"]
            print ("db_info", db_info)

        # Add more vectors to the last database to get over 50,000 vectors
        db_name = "fiqa_test_2"
        # Double the size of the vectors and text lists
        #vectors.extend(vectors)
        #text.extend(text)

        # Add vectors to the database
        """batch_size = 1000
        for i in range(0, 15000, batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((vectors[j], {"text": text[j]}))
            response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

        response = self.client.get(f"/db/{db_name}/info")
        db_info = response.json()["db_info"]
        print ("db_info", db_info)"""


    def test__002_auto_train(self):

        vectors = self.vectors
        text = self.text

        # Add more vectors to the first database to get close to 50,000 vectors, but not to it yet
        db_name = "fiqa_test_1"

        # Add 25,000 vectors to the database (for a total of 55,000) to get over 50,000 vectors
        batch_size = 1000
        for i in range(0, 25000, batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((vectors[j], {"text": text[j]}))
            response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

        response = self.client.get(f"/db/{db_name}/info")
        db_info  = response.json()["db_info"]
        print ("db_info", db_info)


        ### Find indexes to train ###
        response = self.client.get("/db/get_initial_training_queue")
        print (response.json())
        indexes_to_train = response.json()["initial_training_queue"]

        # The only database that should be returned is the last one
        self.assertTrue(indexes_to_train[0] == "fiqa_test_1")
        self.assertTrue(len(indexes_to_train) == 1)


        ### Add 25,000 more vectors to the second database to get over 50,000 vectors ###
        db_name = "fiqa_test_2"
        batch_size = 1000
        for i in range(0, 25000, batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((vectors[j], {"text": text[j]}))
            response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)

        response = self.client.get(f"/db/{db_name}/info")
        db_info  = response.json()["db_info"]
        print ("db_info", db_info)


        ### Find indexes to train again ###
        response = self.client.get("/db/get_initial_training_queue")
        print (response.json())
        indexes_to_train = response.json()["initial_training_queue"]

        # There should be 2 databases in the initial training queue
        self.assertTrue(indexes_to_train[1] == "fiqa_test_2")
        self.assertTrue(len(indexes_to_train) == 2)


        # Wait for the training to complete
        tries = 0
        while tries < 40:
            response = self.client.get(f"/db/fiqa_test_1/train")
            status = response.json()["status"]
            if status == "complete":
                break
            else:
                tries += 1
                time.sleep(20)
        
        ### Find indexes to train again ###
        response = self.client.get("/db/get_initial_training_queue")
        print (response.json())
        indexes_to_train = response.json()["initial_training_queue"]

        # The only database that should be returned is the 2nd one, because the first one has already been trained
        self.assertTrue(indexes_to_train[0] == "fiqa_test_2")
        self.assertTrue(len(indexes_to_train) == 1)


    def test__003_call_training_during_auto_train(self):

        # The fiqa test 2 DB will be in the initial training queue, so it shouldn't show up here
        response = self.client.get("/db/find_indexes_to_train")
        print (response.json())
        indexes_to_train = response.json()["training_queue"]

        self.assertTrue(len(indexes_to_train) == 0)
        
        
    """def test__003_auto_train_during_adding(self):

        ### Add some more vectors to the first database (to hit 50,000 vectors where it will automatically train)
        db_name = "fiqa_test_1"
        vectors = self.vectors
        text = self.text
        #vectors.extend(vectors)
        #text.extend(text)

        response = self.client.post("/db/create", json={"name": db_name})
        assert response.status_code == 200

        # Add vectors to the database
        batch_size = 1000
        for i in range(0, len(vectors), batch_size):
            data = []
            for j in range(i, i+batch_size):
                data.append((vectors[j], {"text": text[j]}))
            response = self.client.post(f"/db/{db_name}/add", json={"add_data": data})
        self.assertTrue(response.status_code == 200)


        # Check the training status
        response = self.client.get(f"/db/{db_name}/train")
        status = response.json()["status"]

        self.assertEqual(status, "in progress")

        tries = 0
        while tries < 30:
            response = self.client.get(f"/db/{db_name}/train")
            status = response.json()["status"]
            if status == "complete":
                break
            else:
                tries += 1
                time.sleep(20)"""

    
    # Call the auto train endpoint
    def test__004_tear_down(self):
        for db_name in self.db_names:
            response = self.client.post(f"/db/{db_name}/delete")

if __name__ == "__main__":
    unittest.main()