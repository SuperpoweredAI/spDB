from spdb.spdb import spDB, load_db, lmdb_utils
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import threading
import json
import time

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../'))


# Install FastAPI and Uvicorn
# pip install fastapi uvicorn


app = FastAPI()


# Load all databases located in the ~/.spdb folder into a dictionary
db_path = os.path.join(os.path.expanduser("~"), ".spdb")
if not (os.path.exists(db_path)):
    os.mkdir(db_path)
db_names = os.listdir(db_path)

# Create a dictionary of databases keyed on name
databases = {name: load_db(name) for name in db_names}

operations = {}
unassigned_vectors = {}

# Define request and response models


class AddInput(BaseModel):
    add_data: List[Tuple]

class RemoveInput(BaseModel):
    ids: List[int]

class QueryInput(BaseModel):
    query_vector: List[float]
    preliminary_top_k: Optional[int] = 500
    final_top_k: Optional[int] = 100


class QueryOutput(BaseModel):
    metadata: List[dict]
    ids: List[int]
    cosine_similarity: List[float]


class CreateDBInput(BaseModel):
    name: str
    vector_dimension: Optional[int] = None
    max_memory_usage: Optional[int] = 4 * 1024 * 1024 * 1024


class TrainDBInput(BaseModel):
    use_two_level_clustering: Optional[bool] = None
    pca_dimension: Optional[int] = None
    opq_dimension: Optional[int] = None
    compressed_vector_bytes: Optional[int] = None
    omit_opq: Optional[bool] = False


# API routes
@app.get("/health")
def read_root():
    return {"status": "healthy"}


@app.get("/db/{db_name}/info")
def get_info(db_name: str):
    # Return a json object with the class attributes
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]

    if db.faiss_index == None:
        n_total = 0
    else:
        n_total = db.faiss_index.ntotal
    db_info = {
        "name": db.name,
        "vector_dimension": db.vector_dimension,
        "num_vectors": db.num_vectors,
        "trained_index_coverage_ratio": db.trained_index_coverage_ratio,
        "max_memory_usage": db.max_memory_usage,
        "n_total": n_total,
        "max_id": db.max_id,
        "lmdb_uncompressed_vectors_path": db.lmdb_uncompressed_vectors_path,
        "unassigned_vectors": unassigned_vectors
    }

    # Turn the object into a string so it can be returned
    db_info = json.dumps(db_info)

    return {"db_info": db_info}


@app.post("/db/create")
def create_db(create_db_input: CreateDBInput):
    if create_db_input.name in databases:
        raise HTTPException(
            status_code=400, detail="Database with this name already exists")
    db = spDB(name=create_db_input.name, vector_dimension=create_db_input.vector_dimension,
              max_memory_usage=create_db_input.max_memory_usage)
    databases[create_db_input.name] = db
    return {"message": "Database created successfully"}


@app.post("/db/{db_name}/add")
def add_vectors(db_name: str, data: AddInput):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]

    training_status = "untrained"
    if (db_name in operations):
        training_status = operations[db_name]

    # If the status of the training operation is 'trained', add the vectors to the new faiss index as well
    if training_status == "trained":
        ids = db.add(data=data.add_data, add_to_new_faiss_index=True)
    else:
        ids = db.add(data=data.add_data)

    # If there is a training process happening, add the vectors to a list of unassigned vectors
    if training_status == "in progress":
        if db_name not in unassigned_vectors:
            unassigned_vectors[db_name] = []
        unassigned_vectors[db_name].extend(ids)
    
    return {"message": "Vectors and text added successfully"}


@app.post("/db/{db_name}/remove")
def remove_vectors_by_id(db_name: str, ids: RemoveInput):

    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    
    db = databases[db_name]
    db.remove(vector_ids=ids.ids)

    return {"message": f"{len(ids.ids)} vectors removed successfully"}


def cleanup_training(db_name: str):

    # One last check to make sure there are no unassigned vectors
    db = databases[db_name]
    
    if (db_name in unassigned_vectors):

        # Add the vectors to the new faiss index in batches of 10,000
        batch_size = 10000
        while len(unassigned_vectors[db_name]) > 0:
            batch_ids = unassigned_vectors[db_name][:batch_size]
            
            with db._lmdb_lock:
                vectors = lmdb_utils.get_lmdb_vectors_by_ids(db.lmdb_uncompressed_vectors_path, batch_ids)
            with db._faiss_lock:
                db.faiss_index.add_with_ids(vectors, batch_ids)
                unassigned_vectors[db_name] = unassigned_vectors[db_name][batch_size:]
            

def train_db(db_name: str, train_db_input: TrainDBInput):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]

    db.train(
        use_two_level_clustering=train_db_input.use_two_level_clustering,
        pca_dimension=train_db_input.pca_dimension,
        opq_dimension=train_db_input.opq_dimension,
        compressed_vector_bytes=train_db_input.compressed_vector_bytes,
        omit_opq=train_db_input.omit_opq
    )
    operations[db_name] = "trained"

    # Sleep for 5 seconds. This in unnecessarily long, but it makes sure we don't start adding the unassigned vectors
    # to the faiss index in the middle of an add operation. 
    time.sleep(5)
    
    # Perform the cleanup operation to make sure all of the vectors have been added to the faiss index
    if (db_name in unassigned_vectors):

        # Add the vectors to the new faiss index in batches of 1,000
        batch_size = 1000
        expected_num_batches = int(len(unassigned_vectors[db_name]) / batch_size)
        num_iters = 0
        while len(unassigned_vectors[db_name]) > 0:
            batch_ids = unassigned_vectors[db_name][:batch_size]

            if len(batch_ids) >= len(unassigned_vectors[db_name]):
                set_new_faiss_index = True
            else:
                set_new_faiss_index = False
            
            success = db.add_unassigned_vectors(vector_ids=batch_ids, set_new_faiss_index=set_new_faiss_index)

            """if not success:
                # This means we couldn't add the vectors to the new faiss index, so we should pause for a second
                time.sleep(1)
                success = db.add_unassigned_vectors(vector_ids=batch_ids, set_new_faiss_index=set_new_faiss_index)"""
            unassigned_vectors[db_name] = unassigned_vectors[db_name][batch_size:]
            num_iters += 1

            if (num_iters > expected_num_batches*2):
                # This is a safeguard to make sure we don't get stuck in an infinite loop
                # TODO: Alert us that something went wrong
                break
        
    
    # When the operation is complete, update the operation status to 'complete', set the faiss index
    # to the new faiss index, and remove the new faiss index
    with db._faiss_lock:
        db.faiss_index = db.new_faiss_index
        db.new_faiss_index = None
    operations[db_name] = "complete"

    time.sleep(5)
    # Perform the cleanup operation to make sure all of the vectors have been added to the faiss index
    cleanup_training(db_name)


@app.post("/db/{db_name}/train")
def start_train_db(db_name: str, train_db_input: TrainDBInput):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")

    # Check if this db is already training
    if db_name in operations:
        status = operations[db_name]
        if status == "in progress":
            raise HTTPException(
                status_code=400, detail="This database is in the process of training already")

    operations[db_name] = "in progress"

    thread = threading.Thread(target=train_db, args=(
        db_name, train_db_input))
    thread.start()

    return {"status": "training successfully initiated"}


@app.get("/db/{db_name}/train")
def get_operation_status(db_name: str):
    if db_name not in operations:
        raise HTTPException(status_code=404, detail="Operation not found")

    return {"status": operations[db_name]}


@app.post("/db/{db_name}/query")
def query(db_name: str, query_input: QueryInput):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]
    results = db.query(
        query_vector=query_input.query_vector, preliminary_top_k=query_input.preliminary_top_k, final_top_k=query_input.final_top_k
    )
    ids = results['ids']
    metadata = results['metadata']
    cosine_similarity = results['cosine_similarity']
    return QueryOutput(metadata=metadata, ids=ids, cosine_similarity=cosine_similarity)


@app.post("/db/{db_name}/save")
def save_db(db_name: str):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]
    db.save()
    return {"message": "Database saved successfully"}


@app.post("/db/{db_name}/reload")
def reload_db(db_name: str):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        db = load_db(db_name, db_path)
        databases[db_name] = db
        return {"message": "Database reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/db/{db_name}/delete")
def delete_db(db_name: str):
    if db_name not in databases:
        raise HTTPException(status_code=404, detail="Database not found")
    db = databases[db_name]
    db.delete()
    del databases[db_name]
    return {"message": "Database deleted successfully"}


"""
Usage


# Run the server using Uvicorn
# uvicorn app:app --host 0.0.0.0 --port 8000
"""
