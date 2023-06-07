import numpy as np
import lmdb


def create_lmdb(save_path: str, name: str) -> None:
    
    # Create the LMDB for the vectors
    env = lmdb.open(f'{save_path}{name}_full_vectors')
    env.close()

    # Create the LMDB for the text
    env = lmdb.open(f'{save_path}{name}_full_text')
    env.close()


def add_vectors_to_lmdb(save_path: str, name: str, vectors: np.ndarray, ids: list) -> None:
    
    # Add the vectors to the LMDB
    env = lmdb.open(f'{save_path}{name}_full_vectors', map_size=1099511627776) # 1TB
    with env.begin(write=True) as txn:
        for i, vector in enumerate(vectors):
            txn.put(str(ids[i]).encode('utf-8'), vector.tobytes())
    
    # TODO: handle the case where the vector upload fails


def add_text_to_lmdb(save_path: str, name: str, text: list, ids: list) -> None:
    
    # Add the text to LMDB
    env = lmdb.open(f'{save_path}{name}_full_text', map_size=1099511627776) # 1TB
    with env.begin(write=True) as txn:
        for i, t in enumerate(text):
            txn.put(str(ids[i]).encode('utf-8'), t.encode('utf-8'))
    
    # TODO: handle the case where the text upload fails


def remove_vectors_from_lmdb(save_path: str, name: str, ids: list):
    
    # Add the vectors to the LMDB
    env = lmdb.open(f'{save_path}{name}_full_vectors', map_size=1099511627776) # 1TB
    with env.begin(write=True) as txn:
        for id in ids:
            txn.delete(str(id).encode('utf-8'))
    
    # TODO: handle the case where the vector upload fails


def remove_text_from_lmdb(save_path: str, name: str, ids: list):
    
    # Add the text to LMDB
    env = lmdb.open(f'{save_path}{name}_full_text', map_size=1099511627776) # 1TB
    with env.begin(write=True) as txn:
        for id in ids:
            txn.delete(str(id).encode('utf-8'))
    
    # TODO: handle the case where the text upload fails


def get_ranked_vectors(save_path: str, name: str, I: np.ndarray) -> tuple[np.ndarray, dict]:

    # query lmdb for the vectors
    corpus_vectors = []
    position_to_id_map = {}
    env = lmdb.open(f'{save_path}{name}_full_vectors')
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


def get_reranked_text(save_path: str, name: str, reranked_I: np.ndarray, position_to_id_map: dict) -> list:
    
    # retrieve text for top_k results from LMDB
    reranked_text = []
    env = lmdb.open(f'{save_path}{name}_full_text')
    with env.begin() as txn:
        for position in reranked_I[0]:
            id = position_to_id_map[position]
            value = txn.get(str(id).encode('utf-8'))
            # Convert from bytes to string
            value = value.decode('utf-8')
            reranked_text.append(value)
    env.close()
    return reranked_text

def get_lmdb_index_ids(save_path: str, name: str) -> list:
    env = lmdb.open(f'{save_path}{name}_full_vectors')
    # Get the ids from the LMDB
    with env.begin() as txn:
        # decode the keys from bytes to strings
        keys = [key.decode('utf-8') for key in txn.cursor().iternext(keys=True, values=False)]
    
    env.close()
    return keys

def get_lmdb_vectors_by_ids(save_path: str, name: str, ids: list) -> np.ndarray:
    env = lmdb.open(f'{save_path}{name}_full_vectors')
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
