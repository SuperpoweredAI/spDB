import numpy as np
import lmdb
import os
from typing import Callable


MAP_SIZE = 1099511627776 # 1TB


def create_lmdb(lmdb_path: str, db_name: str) -> str:
    # Create the LMDB for the vectors
    db_path = os.path.join(lmdb_path, db_name)
    os.makedirs(db_path, exist_ok=True)
    env = lmdb.open(db_path)
    env.close()
    return db_path


def add_items_to_lmdb(db_path: str, items, ids: list, encode_fn: Callable) -> None:
    # Add the text to LMDB
    env = lmdb.open(db_path, map_size=MAP_SIZE) # 1TB
    with env.begin(write=True) as txn:
        for i, t in enumerate(items):
            txn.put(str(ids[i]).encode('utf-8'), encode_fn(t))
    env.close()
    # TODO: handle the case where the text upload fails


def remove_from_lmdb(db_path: str, ids: list):
    # remove the vectors to the LMDB
    env = lmdb.open(db_path, map_size=MAP_SIZE) # 1TB
    with env.begin(write=True) as txn:
        for id in ids:
            txn.delete(str(id).encode('utf-8'))
    env.close()
    # TODO: handle the case where the delete fails


def get_ranked_vectors(full_vectors_path: str, I: np.ndarray) -> tuple[np.ndarray, dict]:
    # query lmdb for the vectors
    corpus_vectors = []
    position_to_id_map = {}
    env = lmdb.open(full_vectors_path)
    with env.begin() as txn:
        for i, id in enumerate(I[0]):
            value = txn.get(str(id).encode('utf-8'))
            value = np.frombuffer(value, dtype=np.float32)
            corpus_vectors.append(value)
            position_to_id_map[i] = id
    env.close()
    # Convert the list to a numpy array
    corpus_vectors = np.array(corpus_vectors)
    return corpus_vectors, position_to_id_map


def get_reranked_text(full_text_path: str, reranked_I: np.ndarray, position_to_id_map: dict) -> list:
    # retrieve text for top_k results from LMDB
    reranked_text = []
    reranked_ids = []
    env = lmdb.open(full_text_path)
    with env.begin() as txn:
        for position in reranked_I[0]:
            id = position_to_id_map[position]
            value = txn.get(str(id).encode('utf-8'))
            # Convert from bytes to string
            value = value.decode('utf-8')
            reranked_text.append(value)
            reranked_ids.append(id)
    env.close()
    return reranked_text, reranked_ids


def get_lmdb_index_ids(db_path: str) -> list:
    env = lmdb.open(db_path)
    # Get the ids from the LMDB
    with env.begin() as txn:
        # decode the keys from bytes to strings
        keys = [key.decode('utf-8') for key in txn.cursor().iternext(keys=True, values=False)]
    env.close()
    return keys


def get_lmdb_vectors_by_ids(full_vectors_path: str, ids: list) -> np.ndarray:
    env = lmdb.open(full_vectors_path)
    # Get the ids from the LMDB
    with env.begin() as txn:
        vectors = []
        for id in ids:
            value = txn.get(str(id).encode('utf-8'))
            value = np.frombuffer(value, dtype=np.float32)
            vectors.append(value)
    env.close()
    vectors = np.array(vectors)
    return vectors


def get_db_count(db_path: str) -> int:
    env = lmdb.open(db_path)
    with env.begin() as txn:
        count = txn.stat()['entries']
    env.close()
    return count
