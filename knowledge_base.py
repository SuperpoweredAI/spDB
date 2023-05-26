import pickle
import faiss
from faiss.contrib.exhaustive_search import knn
import numpy as np
import lmdb

import utils # get_num_clusters, get_n_probe, create_faiss_index_ids
import lmdb_utils
import input_validation

class KnowledgeBase:
    def __init__(self, name, save_path=None):
        self.name = name
        self.save_path = save_path
        self.faiss_index = None
        #self.memmap = None # Not using this for now
        self.vector_size = None
        #self.embedding_function = None # Also not using this for now
        self.max_id = -1

        # Create the LMDB for the vectors
        env = lmdb.open(f'{save_path}{self.name}_full_vectors')
        env.close()

        # Create the LMDB for the text
        env = lmdb.open(f'{save_path}{self.name}_full_text')
        env.close()

    def save(self):
        # save faiss index and delete (so it doesn't get pickled)
        faiss.write_index(self.faiss_index, f'{self.save_path}{self.name}.index')
        tmp = self.faiss_index
        self.faiss_index = None

        # save object to pickle file
        pickle.dump(self, open(f'{self.save_path}{self.name}.pickle', 'wb'))

        self.faiss_index = tmp

    def train(self, data: np.ndarray, pca: int = 256, pq_bytes: int = 32, opq_dimension: int = 128):
        # Validate the inputs
        is_valid, reason = input_validation.validate_train(data, pca, pq_bytes, opq_dimension)
        if not is_valid:
            raise ValueError(reason)
        
        # Get the parameters for training the index
        num_clusters = utils.get_num_clusters(data)
        index_factory_parameters = [f'PCA{pca}', f'OPQ{pq_bytes}_{opq_dimension}', f'IVF{num_clusters}', f'PQ{pq_bytes}']
        index_factory_parameter_string = ','.join(index_factory_parameters)
        dimension = data.shape[1]

        # Define the dimension of the vectors
        self.vector_size = dimension

        # TODO: Check the size of the data to make sure it's not too big for the machine
        # If it is, then we will train on a random subset of the data

        # create the index
        index = faiss.index_factory(dimension, index_factory_parameter_string)
        index.train(data)

        # Set the n_probe parameter (I think it makes sense here since n_probe is dependent on num_clusters)
        n_probe = utils.get_n_probe(num_clusters)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

        self.faiss_index = index

        self.save()

    def add(self, vectors: np.ndarray, text: list):
        # add vector to faiss index

        # Validate the inputs
        is_valid, reason = input_validation.validate_add(vectors, text, self.vector_size)
        if not is_valid:
            raise ValueError(reason)

        ids = utils.create_faiss_index_ids(self.max_id, vectors.shape[0])
        self.max_id = ids[-1]
        self.faiss_index.add_with_ids(vectors, ids)

        # Add the vectors to the LMDB
        env = lmdb.open(f'{self.save_path}{self.name}_full_vectors', map_size=1099511627776) # 1TB
        with env.begin(write=True) as txn:
            for i, vector in enumerate(vectors):
                txn.put(str(ids[i]).encode('utf-8'), vector.tobytes())

        # Add the text to LMDB
        env = lmdb.open(f'{self.save_path}{self.name}_full_text', map_size=1099511627776) # 1TB
        with env.begin(write=True) as txn:
            for i, t in enumerate(text):
                txn.put(str(ids[i]).encode('utf-8'), t.encode('utf-8'))
        
        self.save()

    def remove(self, vector_ids):
        # remove vector from faiss index
        # remove text from LMDB

        pass

    def query(self, query_vector: np.ndarray, top_k=100):

        # query_vector needs to be a 1D array

        is_valid, reason = input_validation.validate_query(query_vector)
        if not is_valid:
            raise ValueError(reason)

        # Check if we need to reshape the query vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape((-1, self.vector_size))

        # query faiss index
        _, I = self.faiss_index.search(query_vector, 500)
        print (I.shape)

        corpus_vectors, position_to_id_map = lmdb_utils.get_ranked_vectors(self.save_path, self.name, I)
        
        # brute force search full vectors to find true top_k
        _, reranked_I = knn(query_vector, corpus_vectors, top_k)

        reranked_text = lmdb_utils.get_reranked_text(self.save_path, self.name, reranked_I, position_to_id_map)
    
        return reranked_text


def create_knowledge_base(name):
    kb = KnowledgeBase(name)

    return kb

def load_knowledge_base(name, save_path):
    # load KnowledgeBase object from pickle file
    kb = pickle.load(open(f'{save_path}{name}.pickle', 'rb'))

    # load faiss index from save path
    kb.faiss_index = faiss.read_index(f'{kb.save_path}{name}.index')

    return kb
