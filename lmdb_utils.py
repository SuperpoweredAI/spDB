import numpy as np
import lmdb

def get_ranked_vectors(save_path, name, I):

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


def get_reranked_text(save_path, name, reranked_I, position_to_id_map):
    
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