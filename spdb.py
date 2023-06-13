import faiss
import numpy as np
import os
import pickle
from faiss.contrib.exhaustive_search import knn

import utils
import lmdb_utils
import input_validation
import train


def get_spdb_path(name: str, save_path: str):
    """
    Get the path to the spDB directory.

    :param name: The name of the database.
    :param save_path: The path where the database files will be saved. Defaults to ~/.spdb/{name}. If this directory does not exist, it will be created.
    """
    # Set the save path to the current directory if it is not specified
    if save_path is None:
        save_path = os.path.join(os.path.expanduser("~"), '.spdb', name)
    return save_path


class spDB:
    """
    A class representing a searchable database using Faiss and LMDB for efficient storage and retrieval of text and vectors.
    """
    def __init__(self, name: str, save_path: str = None, vector_dimension: int = None, max_memory_usage: int = 4*1024*1024*1024):
        """
        Initialize the spDB object.

        :param name: The name of the database.
        :param save_path: The path where the database files will be saved. Defaults to ~/.spdb/{name}. If this directory does not exist, it will be created.
        :param vector_dimension: The dimension of the vectors to be stored in the database. 
        :param max_memory_usage: The maximum memory usage allowed for the construction and querying of the database, in bytes. Defaults to 4 GB.
        """
        self.name = name
        self.faiss_index = None
        self._vector_dimension = vector_dimension
        self.max_id = -1
        self.max_memory_usage = max_memory_usage

        # Set the save path to the current directory if it is not specified
        self._save_path = get_spdb_path(name, save_path)

        # Create the save directory if it doesn't exist and return an exception if it already does exist
        if os.path.exists(self.save_path):
            raise Exception(f"Database with name {self.name} already exists. Please choose a different name.")
        
        os.makedirs(self.save_path)

        # set the lmdb path
        self._lmdb_path = os.path.join(self.save_path, 'lmdb')

        # create the lmdb databases
        self._lmdb_full_vectors_db_path = lmdb_utils.create_lmdb(self.lmdb_path, 'full_vectors')
        self._lmdb_full_text_db_path = lmdb_utils.create_lmdb(self.lmdb_path, 'full_text')

    @property
    def vector_dimension(self):
        return self._vector_dimension
    
    @property
    def save_path(self):
        return self._save_path
    
    @property
    def lmdb_path(self):
        return self._lmdb_path
    
    @property
    def lmdb_full_vector_db_path(self):
        return self._lmdb_full_vectors_db_path
    
    @property
    def lmdb_full_text_db_path(self):
        return self._lmdb_full_text_db_path

    def add(self, vectors: np.ndarray, text: list) -> None:
        """
        Add vectors and their corresponding text to the database.

        :param vectors: A numpy array of vectors to be added.
        :param text: A list of text corresponding to the vectors.
        """

        # Validate the inputs
        is_valid, reason = input_validation.validate_add(
            vectors, text, self.vector_dimension)
        if not is_valid:
            raise ValueError(reason)

        ids = utils.create_faiss_index_ids(self.max_id, vectors.shape[0])
        self.max_id = ids[-1]

        lmdb_utils.add_items_to_lmdb(
            db_path=self.lmdb_full_vector_db_path,
            items=vectors,
            ids=ids,
            encode_fn=np.ndarray.tobytes
        )
        lmdb_utils.add_items_to_lmdb(
            db_path=self.lmdb_full_text_db_path,
            items=text,
            ids=ids,
            encode_fn=str.encode
        )

        # If the index is not trained, don't add the vectors to the index
        if self.faiss_index is not None:
            # TODO: transform vectors if necessary
            self.faiss_index.add_with_ids(vectors, ids)

        self._vector_dimension = vectors.shape[1]
        self.save()

    def train(self, use_two_level_clustering: bool = False, pca_dimension: int = None, opq_dimension: int = None, compressed_vector_bytes: int = None, omit_opq: bool = False) -> None:
        """
        Train the Faiss index for efficient vector search.

        :param use_two_level_clustering: Whether to use two-level clustering for training. If None, the optimal method will be determined based on memory usage and number of vectors.
        :param pca_dimension: The target dimension for PCA dimensionality reduction. If None, a default value will be used.
        :param opq_dimension: The target dimension for OPQ dimensionality reduction. If None, a default value will be used.
        :param compressed_vector_bytes: The number of bytes to use for compressed vectors. If None, a default value will be used.
        :param omit_opq: Whether to omit the OPQ step during training. This reduces training time with a slight drop in accuracy. Defaults to False.
        """

        # get default parameters
        default_params = utils.get_default_faiss_params(self.vector_dimension)
        if pca_dimension is None:
            pca_dimension = default_params['pca_dimension']
        if opq_dimension is None:
            opq_dimension = default_params['opq_dimension']
        if compressed_vector_bytes is None:
            compressed_vector_bytes = default_params['compressed_vector_bytes']

        # Validate the inputs
        is_valid, reason = input_validation.validate_train(
            self.vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes)
        if not is_valid:
            raise ValueError(reason)

        # Load the vectors from the LMDB
        vector_ids = lmdb_utils.get_lmdb_index_ids(self.lmdb_full_vector_db_path)
        num_vectors = len(vector_ids)

        if not use_two_level_clustering:
            # Figure out which training method is optimal based off the max memory usage and number of vectors
            training_method = utils.determine_optimal_training_method(
                max_memory_usage=self.max_memory_usage,
                vector_dimension=self.vector_dimension,
                num_vectors=num_vectors
            )

        if use_two_level_clustering or training_method == 'two_level_clustering':
            print('Training with two level clustering')
            self.faiss_index = train.train_with_two_level_clustering(
                self.lmdb_full_vector_db_path,
                self.vector_dimension,
                pca_dimension,
                opq_dimension,
                compressed_vector_bytes,
                self.max_memory_usage,
                omit_opq
            )
        else:
            print('Training with subsampling')
            self.faiss_index = train.train_with_subsampling(
                self.lmdb_full_vector_db_path, self.vector_dimension, pca_dimension, opq_dimension, compressed_vector_bytes, self.max_memory_usage, omit_opq)

        self.save()

    def query(self, query_vector: np.ndarray, preliminary_top_k: int = 500, final_top_k: int = 100) -> list:
        """
        Query the database to find the most similar text to the given query vector.

        :param query_vector: A 1D numpy array representing the query vector.
        :param preliminary_top_k: The number of preliminary results to retrieve from the compressed Faiss index. Should be 5-10x higher than final_top_k. Defaults to 500.
        :param final_top_k: The number of final results to return after reranking. Defaults to 100.

        :return: two lists containing the reranked text and their corresponding IDs, respectively.
        """

        # query_vector needs to be a 1D array
        is_valid, reason = input_validation.validate_query(query_vector, self.vector_dimension)
        if not is_valid:
            raise ValueError(reason)

        # Check if we need to reshape the query vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape((-1, self.vector_dimension))

        # query faiss index
        _, I = self.faiss_index.search(query_vector, preliminary_top_k)

        corpus_vectors, position_to_id_map = lmdb_utils.get_ranked_vectors(
            self.lmdb_full_vector_db_path, I)

        # brute force search full vectors to find true top_k
        _, reranked_I = knn(query_vector, corpus_vectors, final_top_k)

        reranked_text, reranked_ids = lmdb_utils.get_reranked_text(
            self.lmdb_full_text_db_path, reranked_I, position_to_id_map)

        return reranked_text, reranked_ids
    
    def remove(self, vector_ids: np.ndarray) -> None:
        """
        Remove vectors and their corresponding text from the database.

        :param vector_ids: A numpy array or list of vector IDs to be removed.
        """

        if isinstance(vector_ids, list):
            vector_ids = np.array(vector_ids)

        # Validate the inputs
        is_valid, reason = input_validation.validate_remove(vector_ids)
        if not is_valid:
            raise ValueError(reason)

        # Remove the vectors from the faiss index (has to be done first)
        self.faiss_index.remove_ids(vector_ids)
        # Save here in case something fails in the LMDB removal.
        # We can't have ids in the faiss index that don't exist in the LMDB
        self.save()

        # remove vectors from LMDB
        lmdb_utils.remove_from_lmdb(
            db_path=self.lmdb_full_vector_db_path,
            ids=vector_ids
        )

        # remove text from LMDB
        lmdb_utils.remove_from_lmdb(
            db_path=self.lmdb_full_text_db_path,
            ids=vector_ids
        )

    def save(self) -> None:
        """
        Save the spDB object and its associated Faiss index to disk.
        """

        # Save the faiss index to a tmp variable, then set it to None so it doesn't get pickled
        tmp = self.faiss_index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(self.save_path, f'{self.name}.index'))
            self.faiss_index = None

        # save object to pickle file
        with open(os.path.join(self.save_path, f'{self.name}.pickle'), 'wb') as f:
            pickle.dump(self, f)

        # Reset the faiss index
        self.faiss_index = tmp


def load_db(name: str, save_path: str = None) -> spDB:
    """
    Load an existing spDB object and its associated Faiss index from disk.

    :param name: The name of the database to load.
    :param save_path: The path where the database files are saved. Defaults to the .spdb folder in the the current directory.

    :return: An spDB object.
    """

    # use default save path if none is provided
    save_path = get_spdb_path(name, save_path)

    # load spDB object from pickle file
    with open(os.path.join(save_path, f'{name}.pickle'), 'rb') as f:
        db = pickle.load(f)

    # load faiss index from save path
    try:
        index = faiss.read_index(os.path.join(db.save_path, f'{name}.index'))
    except:
        index = None
    db.faiss_index = index

    return db