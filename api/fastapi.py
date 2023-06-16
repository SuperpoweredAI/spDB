import os
import sys
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../'))

#from spdb import spDB, load_db
from spdb.spdb import spDB, load_db

# Install FastAPI and Uvicorn
# pip install fastapi uvicorn


app = FastAPI()


# Load all databases located in the ~/.spdb folder into a dictionary
db_path = os.path.join(os.path.expanduser("~"), ".spdb")
db_names = os.listdir(db_path)

# Create a dictionary of databases keyed on name
databases = {name: load_db(name) for name in db_names}


# Define request and response models
class VectorInput(BaseModel):
   vectors: List[List[float]]
   text: List[str]


class QueryInput(BaseModel):
   query_vector: List[float]
   preliminary_top_k: Optional[int] = 500
   final_top_k: Optional[int] = 100


class QueryOutput(BaseModel):
   text: List[str]
   ids: List[int]


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
@app.post("/db/create")
def create_db(create_db_input: CreateDBInput):
   if create_db_input.name in databases:
       raise HTTPException(status_code=400, detail="Database with this name already exists")
   db = spDB(name=create_db_input.name, vector_dimension=create_db_input.vector_dimension, max_memory_usage=create_db_input.max_memory_usage)
   databases[create_db_input.name] = db
   return {"message": "Database created successfully"}


@app.post("/db/{db_name}/add")
def add_vectors(db_name: str, vector_input: VectorInput):
   if db_name not in databases:
       raise HTTPException(status_code=404, detail="Database not found")
   db = databases[db_name]
   print ("db", db.name)
   db.add(vectors=vector_input.vectors, text=vector_input.text)
   return {"message": "Vectors and text added successfully"}


@app.post("/db/{db_name}/remove")
def remove_vectors(db_name: str, vector_ids: List[int]):
   if db_name not in databases:
       raise HTTPException(status_code=404, detail="Database not found")
   db = databases[db_name]
   db.remove(vector_ids=vector_ids)
   return {"message": "Vectors and text removed successfully"}


@app.post("/db/{db_name}/train")
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
   return {"message": "Database trained successfully"}


@app.post("/db/{db_name}/query")
def query(db_name: str, query_input: QueryInput):
   if db_name not in databases:
       raise HTTPException(status_code=404, detail="Database not found")
   db = databases[db_name]
   reranked_text, reranked_ids = db.query(query_vector=query_input.query_vector, preliminary_top_k=query_input.preliminary_top_k, final_top_k=query_input.final_top_k)
   return QueryOutput(text=reranked_text, ids=reranked_ids)


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

