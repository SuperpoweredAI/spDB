import faiss
import numpy as np

import two_level_clustering
import lmdb_utils
import utils


def train_with_two_level_clustering(vector_db_path: str, vector_dimension: int, pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int, max_memory_usage: int, omit_opq: bool) -> faiss.IndexPreTransform:

    # TODO: Figure out a better way of getting the number of vectors
    vector_ids = lmdb_utils.get_lmdb_index_ids(vector_db_path)
    num_vectors = len(vector_ids)

    # Get the parameters for training the index
    num_clusters = utils.get_num_clusters(num_vectors)
    index_factory_parameter_string = utils.create_index_factory_parameter_string(pca_dimension, opq_dimension, compressed_vector_bytes, num_clusters, vector_dimension, omit_opq)
    print ("index_factory_parameter_string", index_factory_parameter_string)

    # create the index
    faiss_index = faiss.index_factory(
        vector_dimension, index_factory_parameter_string)

    # Train the index
    index = two_level_clustering.train_ivf_index_with_two_level_clustering(
        faiss_index, num_clusters, max_memory_usage, vector_dimension, vector_db_path)
    print("done training index")

    index = add_vectors_to_faiss(
        vector_db_path, index, vector_ids, num_vectors, vector_dimension, max_memory_usage)

    # Set the n_probe parameter
    n_probe = utils.get_n_probe(num_clusters)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

    return index


def train_with_subsampling(vector_db_path: str, vector_dimension: int, pca_dimension: int, opq_dimension: int, compressed_vector_bytes: int, max_memory_usage: int, omit_opq: bool) -> faiss.IndexPreTransform:

    # Load the vectors from the LMDB
    vector_ids = lmdb_utils.get_lmdb_index_ids(vector_db_path)
    num_vectors = len(vector_ids)

    # Get the parameters for training the index
    num_clusters = utils.get_num_clusters(num_vectors)
    index_factory_parameter_string = utils.create_index_factory_parameter_string(pca_dimension, opq_dimension, compressed_vector_bytes, num_clusters, vector_dimension, omit_opq)
    print ("index_factory_parameter_string", index_factory_parameter_string)

    # Get a subset of the vectors
    memory_usage = utils.get_training_memory_usage(
        vector_dimension, num_vectors)
    print("memory_usage", memory_usage)
    # Define the percentage to train on based off the max memory usage and memory usage
    percentage_to_train_on = min(1, max_memory_usage / memory_usage)
    num_vectors_to_train = int(num_vectors * percentage_to_train_on)
    print("num vectors to train", num_vectors_to_train)

    # Get a random subset of the vectors
    random_indices = np.random.choice(
        vector_ids, num_vectors_to_train, replace=False)
    vectors = lmdb_utils.get_lmdb_vectors_by_ids(
        vector_db_path, random_indices)

    # create the index
    index = faiss.index_factory(
        vector_dimension, index_factory_parameter_string)
    index.train(vectors)

    print("adding vectors to the index now")
    index = add_vectors_to_faiss(
        vector_db_path, index, vector_ids, num_vectors, vector_dimension, max_memory_usage)

    # Set the n_probe parameter (I think it makes sense here since n_probe is dependent on num_clusters)
    n_probe = utils.get_n_probe(num_clusters)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)

    return index


def add_vectors_to_faiss(vector_db_path: str, index: faiss.IndexPreTransform, vector_ids: list, num_vectors: int, vector_dimension: int, max_memory_usage: int) -> faiss.IndexPreTransform:
    
    # Add all of the vectors to the index. We need to know the number of batches to do this in
    num_batches = utils.get_num_batches(
        num_vectors, vector_dimension, max_memory_usage)
    # Calculate the number of vectors per batch
    num_per_batch = np.ceil(num_vectors / num_batches).astype(int)
    for i in range(num_batches):
        # Get the batch ids (based off the number of batches and the current i)
        batch_ids = vector_ids[i *
                               num_per_batch: (min((i + 1) * num_per_batch, num_vectors))]
        vectors = lmdb_utils.get_lmdb_vectors_by_ids(
            vector_db_path, batch_ids)
        # Add the vectors to the index
        index.add_with_ids(vectors, batch_ids)

    return index