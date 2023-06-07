import numpy as np

def get_num_clusters(num_vectors: int) -> int:
    # Get the number of clusters to use for the IVF index, based on the number of vectors
    scaling_factor = 0.2
    num_clusters = int((num_vectors**0.75) * scaling_factor)
    return num_clusters

def get_n_probe(num_clusters: int) -> int:
    # Get the number of probes to use for the IVF index, based on the number of clusters
    # This is a piecewise linear function, based on the log of the number of clusters

    log_num_clusters = np.log(num_clusters)

    # Define the piecewise linear function breakpoints and y values
    x_breakpoints = [np.log(200), np.log(1000), np.log(6350), np.log(200000)]
    y_values = [0.5, 0.25, 0.07, 0.03]

    if log_num_clusters <= x_breakpoints[0]:
        n_probe_factor = y_values[0]
    elif log_num_clusters <= x_breakpoints[1]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[0], x_breakpoints[1]], [y_values[0], y_values[1]])
    elif log_num_clusters <= x_breakpoints[2]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[1], x_breakpoints[2]], [y_values[1], y_values[2]])
    elif log_num_clusters <= x_breakpoints[3]:
        n_probe_factor = np.interp(log_num_clusters, [x_breakpoints[2], x_breakpoints[3]], [y_values[2], y_values[3]])
    else:
        n_probe_factor = y_values[3]

    return int(n_probe_factor * num_clusters)

def create_faiss_index_ids(max_id: int, num_new_vectors: int) -> list:
    # Create a sequential list of IDs for the new vectors
    # The IDs start at max_id + 1 and go up to max_id + num_new_vectors
    new_ids = range(max_id + 1, max_id + num_new_vectors + 1)
    return new_ids

def get_training_memory_usage(vector_dimension: int, num_vectors: int) -> int:
    # 1M 768 dimension vectors uses ~10GB of memory
    memory_usage = int(num_vectors * vector_dimension * 4 * 3) # 4 bytes per float, with a 3x multiplier for overhead
    return memory_usage

def get_num_batches(num_vectors: int, vector_dimension: int, max_memory_usage: int) -> int:
    memory_usage = num_vectors * vector_dimension * 4
    # We don't really need to push memory requirements here, so we'll just use 1/4 of the max memory usage
    num_batches = int(np.ceil(memory_usage / (max_memory_usage / 4)))
    return num_batches

def determine_optimal_training_method(max_memory_usage: int, vector_dimension: int, num_vectors: int) -> str:

    memory_usage = get_training_memory_usage(vector_dimension, num_vectors)
    max_num_vectors = int((max_memory_usage / memory_usage) * num_vectors)
    num_clusters = get_num_clusters(num_vectors)
    num_vectors_per_cluster = int(max_num_vectors / num_clusters)

    # faiss recommends a minimum of 39 vectors per cluster
    if num_vectors_per_cluster < 39:
        # We can use the subsampling method
        return 'clustering'
    else:
        # We need to use the clustering method
        return 'subsample'
    