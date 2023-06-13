import faiss
import numpy as np
import time

from . import lmdb_utils
from . import utils


def assign_to_centroids(batches: list, km: faiss.extra_wrappers.Kmeans, vector_db_path: str, all_vector_transforms: list) -> np.ndarray:
    # Assign the data to the coarse centroids in batches
    all_centroid_assignments = []
    for i in range(len(batches)):
        x = lmdb_utils.get_lmdb_vectors_by_ids(vector_db_path, batches[i])
        x = apply_pre_transforms(x, all_vector_transforms)
        _, centroid_assignment = km.assign(x)
        all_centroid_assignments.append(centroid_assignment)
    all_centroid_assignments = np.concatenate(all_centroid_assignments)
    return all_centroid_assignments


def train_sub_clusters(
    num_coarse_clusters: int, sub_clusters_per_coarse_cluster: int, vector_dimension: int, sorted_centroid_assignments: np.ndarray,
    bin_count: np.ndarray, vector_db_path: str, batches: list, all_vector_transforms: list
) -> np.ndarray:

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

        data_subset = lmdb_utils.get_lmdb_vectors_by_ids(vector_db_path, batch_ids)
        data_subset = apply_pre_transforms(data_subset, all_vector_transforms)

        km = faiss.Kmeans(vector_dimension, num_sub_clusters)
        km.train(data_subset)
        sub_clusters.append(km.centroids)
        del km
        index_0 = index_1

    return np.vstack(sub_clusters)


def two_level_clustering(num_coarse_clusters: int, num_total_clusters: int, vector_db_path: str, 
    all_vector_transforms: list, max_memory_usage: int, vector_dimension: int, clustering_niter: int = 25) -> np.ndarray:

    # Define the number of vectors to pull in from the lmdb at a time (can't hold all vectors in memory at once)
    num_vectors_per_batch = utils.get_num_vectors_per_batch(max_memory_usage, vector_dimension)

    max_samples = num_coarse_clusters*256  # max of 256 samples per centroid
    random_sub_sample, _ = get_random_vectors(max_samples, vector_db_path)
    random_sub_sample = apply_pre_transforms(
        random_sub_sample, all_vector_transforms)

    d = random_sub_sample.shape[1]
    km = faiss.Kmeans(
        d, num_coarse_clusters, niter=clustering_niter,
        max_points_per_centroid=2000
    )
    start_time = time.time()
    km.train(random_sub_sample)
    print("time taken to train km", time.time() - start_time)

    start_time = time.time()
    batches = break_into_batches(vector_db_path, num_vectors_per_batch)
    extended_batches = []
    for batch in batches:
        extended_batches.extend(batch)
    print("time taken to break into batches", time.time() - start_time)

    start_time = time.time()
    all_centroid_assignments = assign_to_centroids(
        batches, km, vector_db_path, all_vector_transforms)
    print("time taken to assign to centroids", time.time() - start_time)
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
    sub_clusters = train_sub_clusters(
        num_coarse_clusters, sub_clusters_per_coarse_cluster, d, sorted_centroid_assignments, bin_count,
        vector_db_path, batches, all_vector_transforms
    )
    print("time taken to train sub clusters", time.time() - start_time)

    return sub_clusters


def handle_pre_transforms(index: faiss.IndexPreTransform, vector_dimension: int, vector_db_path: str) -> tuple[faiss.Index, list]:
    # handle PreTransforms

    # index is the faiss index
    # vector_dimension is the dimensionality of the vectors

    start_time = time.time()
    random_sub_sample, _ = get_random_vectors(
        vector_dimension*100, vector_db_path)
    print("time taken to get random vectors inside handle_pre_transforms",
          time.time() - start_time)
    all_vector_transforms = []
    for i in range(index.chain.size()):
        start_time = time.time()
        vector_transform = index.chain.at(i)
        vector_transform.train(random_sub_sample)
        random_sub_sample = vector_transform.apply(random_sub_sample)
        all_vector_transforms.append(vector_transform)
        print (f"time taken to train and apply vector transform {i}", time.time() - start_time)

    index.is_trained = True
    return faiss.downcast_index(index.index), all_vector_transforms


def train_ivf_index_with_two_level_clustering(index: faiss.IndexPreTransform, num_total_clusters: int,max_memory_usage: int, vector_dimension: int, vector_db_path: str) -> faiss.IndexPreTransform:
    """
    Applies 2-level clustering to an index_ivf embedded in an index.
    """

    start_time = time.time()
    ivf_index, all_vector_transforms = handle_pre_transforms(
        index, vector_dimension, vector_db_path)
    print("time taken to handle pre transforms", time.time() - start_time)

    # TODO: check these instead of using assert, and if they're not true, raise an error
    # although, I don't think it is possible for these to not be true, so these should go in a test
    assert isinstance(ivf_index, faiss.IndexIVF)
    assert ivf_index.metric_type == faiss.METRIC_L2

    # now do 2-level clustering
    # number of clusters at top level
    num_coarse_clusters = int(np.sqrt(ivf_index.nlist))

    start_time = time.time()
    centroids = two_level_clustering(
        num_coarse_clusters, num_total_clusters, vector_db_path, all_vector_transforms, max_memory_usage, vector_dimension)
    print("time taken to do two level clustering", time.time() - start_time)
    ivf_index.quantizer.train(centroids)
    ivf_index.quantizer.add(centroids)

    # finish training (PQ and PCA)
    # 256 centroids times 64 samples per centroid for PQ; assumed good enough for PCA
    max_samples = 64*256
    # get samples from disk
    start_time = time.time()
    random_sub_sample, _ = get_random_vectors(max_samples, vector_db_path)
    random_sub_sample = apply_pre_transforms(
        random_sub_sample, all_vector_transforms)
    print("time taken to get random vectors and apply_pre_transforms",
          time.time() - start_time)

    start_time = time.time()
    ivf_index.train(random_sub_sample)
    print("time taken to train ivf index", time.time() - start_time)

    index.index = ivf_index

    return index


def get_random_vectors(n: int, vector_db_path: str) -> tuple[np.ndarray, np.ndarray]:

    lmdb_ids = lmdb_utils.get_lmdb_index_ids(vector_db_path)
    # Cap the number of vectors to the number of ids in the lmdb
    n = min(n, len(lmdb_ids))
    # Create a random vector of size n, with each element being a random integer between min(lmdb_ids) and max(lmdb_ids)
    random_integers = np.random.randint(min(lmdb_ids), max(lmdb_ids), n)
    # Create a mapping of the random integers to lmdb_ids
    random_vector_ids = np.array([lmdb_ids[i] for i in random_integers])
    # Get the vectors from the lmdb for these ids
    vectors = lmdb_utils.get_lmdb_vectors_by_ids(
        vector_db_path, random_vector_ids)

    return vectors, random_vector_ids


def apply_pre_transforms(vectors: np.ndarray, all_vt: list) -> np.ndarray:
    for vt in all_vt:
        vectors = vt.apply(vectors)
    return vectors


def break_into_batches(vector_db_path: str, num_vectors_per_batch: int) -> list:
    # Break the vectors into batches of size batch_size
    lmdb_ids = lmdb_utils.get_lmdb_index_ids(vector_db_path)
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
