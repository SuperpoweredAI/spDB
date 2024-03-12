import requests
import time
from helpers import fiqa_test_data

if __name__ == "__main__":
    db_name = "fiqa_test" #"test_db_06_30"
    larger_instance_url = f"https://uu6xwycc44.execute-api.us-west-1.amazonaws.com/dev"
    url = f"https://ajolbcqoy8.execute-api.us-west-1.amazonaws.com/dev"

    response = requests.post(f"{larger_instance_url}/create", json={"name": db_name})
    print (response.json())
    # 50.18.85.107/32
    #response = requests.get(f"{url}/{db_name}/info")

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
        
    batch_size = 1000
    i=0
    #for i in range(0, 15000, batch_size):
    ids = list(range(i, i+batch_size))
    #response = requests.post(f"{url}/{db_name}/remove", json={"ids": ids})
    
    #response = requests.post(f"{url}/{db_name}/train")
    #response = requests.get(f"{url}/{db_name}/train")
    #print (response.json())

    #response = requests.post(url, json={"query_vector": queries[0].tolist(), "preliminary_top_k": 500, "final_top_k": 150})
    

