import numpy as np

def get_num_clusters(data):
    # Get the number of clusters to use for the IVF index, based on the number of vectors
    scaling_factor = 0.2
    num_vectors = data.shape[0]
    num_clusters = int((num_vectors**0.75) * scaling_factor)
    return num_clusters

def get_n_probe(num_clusters):
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

def create_faiss_index_ids(max_id, num_new_vectors):
    # Create a sequential list of IDs for the new vectors
    # The IDs start at max_id + 1 and go up to max_id + num_new_vectors
    new_ids = range(max_id + 1, max_id + num_new_vectors + 1)
    return new_ids
