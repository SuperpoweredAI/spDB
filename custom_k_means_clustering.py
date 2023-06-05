import faiss
import numpy as np
import time
import lmdb_utils
from utils import get_n_probe

def print_nop(*arg, **kwargs):
    pass


def assign_to_centroids(batches, km, save_path, name, all_vector_transforms):
    # Assign the data to the coarse centroids in batches
    print (len(batches))
    all_centroid_assignments = []
    for i in range(len(batches)):
        x = lmdb_utils.get_lmdb_vectors_by_ids(save_path, name, batches[i])
        x = apply_pre_transforms(x, all_vector_transforms)
        _, centroid_assignment = km.assign(x)
        all_centroid_assignments.append(centroid_assignment)
    all_centroid_assignments = np.concatenate(all_centroid_assignments)
    return all_centroid_assignments


def train_sub_clusters(
    num_coarse_clusters, sub_clusters_per_coarse_cluster, d, sorted_centroid_assignments, bin_count,
    iteration_stats, save_path, name, batches, all_vector_transforms
):

    # Convert the batches list from a list of lists to a single list
    # This is a list of vector ids
    extended_batches = []
    for batch in batches:
        extended_batches.extend(batch)

    # Train the sub-clusters
    index_0 = 0
    sub_clusters = []
    for cluster_num in range(num_coarse_clusters):

        num_sub_clusters = int(sub_clusters_per_coarse_cluster[cluster_num])
        if num_sub_clusters == 0:
            continue
        index_1 = index_0 + bin_count[cluster_num]
        batch_indices = sorted_centroid_assignments[index_0:index_1]
        batch_ids = [extended_batches[i] for i in batch_indices]

        # Get a random subset of the batch ids (64 * num_sub_clusters)
        np.random.shuffle(batch_ids)
        batch_ids = batch_ids[:64*num_sub_clusters]

        data_subset = lmdb_utils.get_lmdb_vectors_by_ids(save_path, name, batch_ids)
        data_subset = apply_pre_transforms(data_subset, all_vector_transforms)
        
        km = faiss.Kmeans(d, num_sub_clusters)
        km.train(data_subset)
        iteration_stats.append(km.iteration_stats)
        sub_clusters.append(km.centroids)
        del km
        index_0 = index_1

    return np.vstack(sub_clusters), iteration_stats


def two_level_clustering(
        num_coarse_clusters, num_total_clusters, save_path, name, all_vector_transforms, rebalance=True, clustering_niter=25):

    # Define the number of vectors to pull in from the lmdb at a time (can't hold all vectors in memory at once)
    num_vectors_per_batch = 250000

    max_samples = num_coarse_clusters*256  # max of 256 samples per centroid
    random_sub_sample, _ = get_random_vectors(max_samples, save_path, name)
    random_sub_sample = apply_pre_transforms(
        random_sub_sample, all_vector_transforms)

    d = random_sub_sample.shape[1]
    km = faiss.Kmeans(
        d, num_coarse_clusters, niter=clustering_niter,
        max_points_per_centroid=2000
    )
    start_time = time.time()
    km.train(random_sub_sample)
    print ("time taken to train km", time.time() - start_time)

    iteration_stats = [km.iteration_stats]

    start_time = time.time()
    batches = break_into_batches(save_path, name, num_vectors_per_batch)
    extended_batches = []
    for batch in batches:
        extended_batches.extend(batch)
    print ("time taken to break into batches", time.time() - start_time)

    start_time = time.time()
    all_centroid_assignments = assign_to_centroids(
        batches, km, save_path, name, all_vector_transforms)
    print ("time taken to assign to centroids", time.time() - start_time)
    bin_count = np.bincount(all_centroid_assignments,
                            minlength=num_coarse_clusters)
    # The centroids are sorted by the coarse cluster number they belong to
    # So the first N centroids belong to the first coarse cluster, etc.
    # This is where the bin counts come in handy. sorted_centroid_assignments[bin_count[0]:bin_count[1]]
    # will give you the indices of the centroids that belong to the first coarse cluster
    sorted_centroid_assignments = all_centroid_assignments.argsort()

    bc_sum = np.cumsum(bin_count)
    # sub_clusters_per_coarse_cluster is number of clusters inside each coarse cluster, adjusted for how many
    # vectors are in each coarse cluster
    sub_clusters_per_coarse_cluster = bc_sum * num_total_clusters // bc_sum[-1]
    sub_clusters_per_coarse_cluster[1:] -= sub_clusters_per_coarse_cluster[:-1]

    start_time = time.time()
    sub_clusters, iteration_stats = train_sub_clusters(
        num_coarse_clusters, sub_clusters_per_coarse_cluster, d, sorted_centroid_assignments, bin_count,
        iteration_stats, save_path, name, batches, all_vector_transforms
    )
    print ("time taken to train sub clusters", time.time() - start_time)

    return sub_clusters, batches


def handle_pre_transforms(index, vector_dimension, save_path, name):
    # handle PreTransforms

    # index is the faiss index
    # vector_size is the dimensionality of the vectors

    start_time = time.time()
    random_sub_sample, _ = get_random_vectors(vector_dimension*100, save_path, name)
    print ("time taken to get random vectors inside handle_pre_transforms", time.time() - start_time)
    all_vector_transforms = []
    for i in range(index.chain.size()):
        vector_transform = index.chain.at(i)
        vector_transform.train(random_sub_sample)
        random_sub_sample = vector_transform.apply(random_sub_sample)
        all_vector_transforms.append(vector_transform)

    index.is_trained = True
    return faiss.downcast_index(index.index), all_vector_transforms


def train_ivf_index_with_2level(index, num_total_clusters, vector_dimension, save_path, name):
    """
    Applies 2-level clustering to an index_ivf embedded in an index.
    """

    start_time = time.time()
    ivf_index, all_vector_transforms = handle_pre_transforms(index, vector_dimension, save_path, name)
    print ("time taken to handle pre transforms", time.time() - start_time)
    assert isinstance(ivf_index, faiss.IndexIVF)
    assert ivf_index.metric_type == faiss.METRIC_L2

    # now do 2-level clustering
    num_coarse_clusters = int(np.sqrt(ivf_index.nlist))  # number of clusters at top level
    
    start_time = time.time()
    centroids, batches = two_level_clustering(
        num_coarse_clusters, num_total_clusters, save_path, name, all_vector_transforms)
    print ("time taken to do two level clustering", time.time() - start_time)
    ivf_index.quantizer.train(centroids)
    ivf_index.quantizer.add(centroids)

    # finish training (PQ and PCA)
    # 256 centroids times 64 samples per centroid for PQ; assumed good enough for PCA
    max_samples = 64*256
    # get samples from disk
    start_time = time.time()
    random_sub_sample, _ = get_random_vectors(max_samples, save_path, name)
    random_sub_sample = apply_pre_transforms(random_sub_sample, all_vector_transforms)
    print ("time taken to get random vectors and apply_pre_transforms", time.time() - start_time)

    start_time = time.time()
    ivf_index.train(random_sub_sample)
    print ("time taken to train ivf index", time.time() - start_time)
    
    # Get all of the vectors from the knowledge base and add them to the index (in batches)
    start_time = time.time()
    for i in range(len(batches)):
        batch_ids = batches[i]
        data_subset = lmdb_utils.get_lmdb_vectors_by_ids(save_path, name, batch_ids)
        data_subset = apply_pre_transforms(data_subset, all_vector_transforms)
        ivf_index.add_with_ids(data_subset, batch_ids)
    print ("num per batch", len(batch_ids))
    print ("time taken to add vectors to index", time.time() - start_time)
    
    # Set the n_probe parameter for the index
    n_probe = get_n_probe(ivf_index.nlist)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", n_probe)
    
    return index, ivf_index


def get_random_vectors(n, save_path, name):

    lmdb_ids = lmdb_utils.get_lmdb_index_ids(save_path, name)
    # Cap the number of vectors to the number of ids in the lmdb
    n = min(n, len(lmdb_ids))
    # Create a random vector of size n, with each element being a random integer between min(lmdb_ids) and max(lmdb_ids)
    random_integers = np.random.randint(min(lmdb_ids), max(lmdb_ids), n)
    # Create a mapping of the random integers to lmdb_ids
    random_vector_ids = np.array([lmdb_ids[i] for i in random_integers])
    # Get the vectors from the lmdb for these ids
    vectors = lmdb_utils.get_lmdb_vectors_by_ids(
        save_path, name, random_vector_ids)

    return vectors, random_vector_ids


def apply_pre_transforms(vectors, all_vt):
    for vt in all_vt:
        vectors = vt.apply(vectors)
    return vectors


def break_into_batches(save_path, name, num_vectors_per_batch):
    # Break the vectors into batches of size batch_size
    lmdb_ids = lmdb_utils.get_lmdb_index_ids(save_path, name)
    lmdb_ids = [int(i) for i in lmdb_ids]
    lmdb_ids.sort()

    # define the number of elements per batch
    num_batches = np.ceil(len(lmdb_ids)/num_vectors_per_batch).astype(int)

    # define the batches
    batches = []
    for i in range(num_batches):
        batches.append(
            lmdb_ids[i*num_vectors_per_batch:(i+1)*num_vectors_per_batch])

    return batches
