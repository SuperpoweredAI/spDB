import requests
import time
from helpers import fiqa_test_data

if __name__ == "__main__":
    db_name = "fiqa_test" #"test_db_06_30"
    url = f"https://ajolbcqoy8.execute-api.us-west-1.amazonaws.com/dev"

    response = requests.get(f"{url}/{db_name}/info")

    vectors, text, queries, ground_truths = fiqa_test_data()
    print (queries[0].shape)

    batch_size = 100
    i = 0
    #for i in range(0, len(vectors), batch_size):
    print (i)
    data = []
    for j in range(i, i+batch_size):
        data.append((
            vectors[j].tolist(),
            {"text": text[j]}
        ))
    
    #response = requests.post(f"{url}/{db_name}/add", json={"add_data": data})
    #print (response)
        
    response = requests.post(f"{url}/{db_name}/train", json={
        "use_two_level_clustering": True,
        "pca_dimension": 256,
        "opq_dimension": 128,
        "compressed_vector_bytes": 32,
        "omit_opq": True
    })
    print (response)

    #response = requests.post(url, json={"query_vector": queries[0].tolist(), "preliminary_top_k": 500, "final_top_k": 150})
    

