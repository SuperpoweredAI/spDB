import pickle
import faiss
from faiss.contrib.exhaustive_search import knn
import numpy as np
import lmdb

import utils
import lmdb_utils
import input_validation
from custom_k_means_clustering import train_ivf_index_with_2level

class KnowledgeBase:
    def __init__(self, name, save_path=None):
        self.name = name
        self.save_path = save_path
        self.faiss_index = None
        #self.memmap = None # Not using this for now
        self.vector_dimension = None
        #self.embedding_function = None # Also not using this for now
        self.max_id = -1
        self.max_memory_usage = 4 * 1024 * 1024 * 1024 # 4 GB

        # Create the LMDB for the vectors
        env = lmdb.open(f'{save_path}{self.name}_full_vectors')
        env.close()

        # Create the LMDB for the text
        env = lmdb.open(f'{save_path}{self.name}_full_text')
        env.close()

    def save(self):
        # save faiss index and delete (so it doesn't get pickled)
        tmp = self.faiss_index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f'{self.save_path}{self.name}.index')
            self.faiss_index = None

        # save object to pickle file
        pickle.dump(self, open(f'{self.save_path}{self.name}.pickle', 'wb'))

        self.faiss_index = tmp
    
    def train_with_2_level_clustering(self, pca: int = 256, pq_bytes: int = 32, opq_dimension: int = 128):

        # TODO: Figure out a better way of getting the number of vectors
        vector_ids = lmdb_utils.get_lmdb_index_ids(self.save_path, self.name)
        num_vectors = len(vector_ids)
        print ("num_vectors", num_vectors)

        # Validate the inputs
        is_valid, reason = input_validation.validate_train(self.vector_dimension, pca, pq_bytes, opq_dimension)
        if not is_valid:
            raise ValueError(reason)

        # Get the parameters for training the index
        num_clusters = utils.get_num_clusters(num_vectors)
        index_factory_parameters = [f'PCA{pca}', f'OPQ{pq_bytes}_{opq_dimension}', f'IVF{num_clusters}', f'PQ{pq_bytes}']
        index_factory_parameter_string = ','.join(index_factory_parameters)

        # create the index
        index = faiss.index_factory(self.vector_dimension, index_factory_parameter_string)
        self.faiss_index = index

        # Train the index
        index, ivf_index = train_ivf_index_with_2level(self.faiss_index, num_clusters, self.vector_dimension, self.save_path, self.name)
                    
        self.faiss_index = index
        self.faiss_index.index = ivf_index
        print ("done training index")

        self.save()


    def train(self, pca: int = 256, pq_bytes: int = 32, opq_dimension: int = 128):

        # Load the vectors from the LMDB
        vector_ids = lmdb_utils.get_lmdb_index_ids(self.save_path, self.name)
        num_vectors = len(vector_ids)

        # Validate the inputs
        is_valid, reason = input_validation.validate_train(self.vector_dimension, pca, pq_bytes, opq_dimension)
        if not is_valid:
            raise ValueError(reason)
        
        # Get the parameters for training the index
        num_clusters = utils.get_num_clusters(num_vectors)
        index_factory_parameters = [f'PCA{pca}', f'OPQ{pq_bytes}_{opq_dimension}', f'IVF{num_clusters}', f'PQ{pq_bytes}']
        index_factory_parameter_string = ','.join(index_factory_parameters)
        print (index_factory_parameter_string)

        # Get a subset of the vectors
        memory_usage = utils.get_training_memory_usage(self.vector_dimension, num_vectors)
        print ("memory_usage", memory_usage)
        # Define the percentage to train on based off the max memory usage and memory usage
        percentage_to_train_on = min(1, self.max_memory_usage / memory_usage)
        num_vectors_to_train = int(num_vectors * percentage_to_train_on)
        print ("num vectors to train", num_vectors_to_train)

        # Get a random subset of the vectors
        random_indices = np.random.choice(vector_ids, num_vectors_to_train, replace=False)
        vectors = lmdb_utils.get_lmdb_vectors_by_ids(self.save_path, self.name, random_indices)
        print ("num vectors", len(vectors))

        # create the index
        index = faiss.index_factory(self.vector_dimension, index_factory_parameter_string)
        index.train(vectors)

        print ("adding vectors to the index now")

        # Add all of the vectors to the index. We need to know the number of batches to do this in
        num_batches = utils.get_num_batches(num_vectors, self.vector_dimension, self.max_memory_usage)
        # Calculate the number of vectors per batch
        num_per_batch = np.ceil(num_vectors / num_batches).astype(int)
        for i in range(num_batches):
            # Get the batch ids (based off the number of batches and the current i)
            batch_ids = vector_ids[i * num_per_batch : (min((i + 1) * num_per_batch, num_vectors))]
            vectors = lmdb_utils.get_lmdb_vectors_by_ids(self.save_path, self.name, batch_ids)
            # Add the vectors to the index
            index.add_with_ids(vectors, batch_ids)

        # Set the n_probe parameter (I think it makes sense here since n_probe is dependent on num_clusters)
        n_probe = utils.get_n_probe(num_clusters)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

        self.faiss_index = index
        self.save()

    def add(self, vectors: np.ndarray, text: list):
        # add vector to faiss index

        # Validate the inputs
        is_valid, reason = input_validation.validate_add(vectors, text, self.vector_dimension)
        if not is_valid:
            raise ValueError(reason)

        ids = utils.create_faiss_index_ids(self.max_id, vectors.shape[0])
        self.max_id = ids[-1]

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
        
        # If the index is not trained, don't add the vectors to the index
        if self.faiss_index is not None:
            # TODO: transform vectors if necessary
            self.faiss_index.add_with_ids(vectors, ids)
        
        self.vector_dimension = vectors.shape[1]
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
            query_vector = query_vector.reshape((-1, self.vector_dimension))

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
    try:
        index = faiss.read_index(f'{kb.save_path}{name}.index')
    except:
        index = None
    kb.faiss_index = index

    return kb
