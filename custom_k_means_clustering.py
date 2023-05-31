import faiss
import numpy as np
import time
import lmdb_utils


def print_nop(*arg, **kwargs):
    pass


def assign_to_centroids(batches, km, save_path, name, nc1):
    # Assign the data to the coarse centroids in batches
    num_batches = 100
    all_centroid_assignments = []
    for i in range(num_batches):
        x = get_batch(save_path, name, batches, i)
        _, centroid_assignment = km.assign(x)
        all_centroid_assignments.append(centroid_assignment)
    all_centroid_assignments = np.concatenate(all_centroid_assignments)
    return all_centroid_assignments


def train_sub_clusters(
    num_coarse_clusters, sub_clusters_per_coarse_cluster, d, sorted_centroid_assignments, bin_count,
    iteration_stats, save_path, name, batches
):

    # train sub-clusters
    i0 = 0
    sub_clusters = []
    for c1 in range(num_coarse_clusters):
        num_sub_clusters = int(sub_clusters_per_coarse_cluster[c1])
        i1 = i0 + bin_count[c1]
        subset_ids = sorted_centroid_assignments[i0:i1]
        # assert np.all(assign1[subset] == c1)
        km = faiss.Kmeans(d, num_sub_clusters)
        # xtsub = xt[subset]
        data_subset = get_batch(save_path, name, batches, subset_ids)
        km.train(data_subset)
        iteration_stats.append(km.iteration_stats)
        sub_clusters.append(km.centroids)
        del km
        i0 = i1
    return np.vstack(sub_clusters), iteration_stats


def two_level_clustering(nc1, nc2, save_path, name, rebalance=True, clustering_niter=25, **args):
    """
    perform 2-level clustering on a training set xt
    nc1 and nc2 are the number of clusters at each level, the final number of
    clusters is nc2. Additional arguments are passed to the Kmeans object.

    Rebalance allocates the number of sub-clusters depending on the number of
    first-level assignment.
    """

    # get training set for top level
    max_samples = nc1*256  # max of 256 samples per centroid
    xt = get_random_vectors(max_samples, save_path, name)

    d = xt.shape[1]

    verbose = args.get("verbose", False)

    log = print if verbose else print_nop
    log(f"2-level clustering of {xt.shape} nb 1st level clusters = {nc1} total {nc2}")
    log("perform coarse training")

    km = faiss.Kmeans(
        d, nc1, niter=clustering_niter,
        max_points_per_centroid=2000,
        **args
    )
    km.train(xt)

    iteration_stats = [km.iteration_stats]
    log()

    # coarse centroids
    centroids1 = km.centroids

    """
    log("assigning the training set")
    t0 = time.time()
    _, assign1 = km.assign(xt) # TODO: make this work with subsampling - we can't assign all the vectors at once
    bc = np.bincount(assign1, minlength=nc1)
    log(f"done in {time.time() - t0:.2f} s. Sizes of clusters {min(bc)}-{max(bc)}")
    o = assign1.argsort()
    del km
    """

    batches = break_into_batches(save_path, name, 100)

    # Assign the data to the coarse centroids in batches
    """num_batches = 100
    all_assign1 = []
    for i in range(num_batches):
        x = get_batch(save_path, name, batches, i)
        #_, I = km.index.search(x, 1)
        _, assign1 = km.assign(x)
        all_assign1.append(assign1)
    all_assign1 = np.concatenate(all_assign1)
    bc = np.bincount(all_assign1, minlength=nc1)
    o = all_assign1.argsort()
    del km"""
    all_centroid_assignments = assign_to_centroids(
        batches, km, save_path, name, nc1)

    bin_count = np.bincount(all_centroid_assignments, minlength=nc1)
    sorted_centroid_assignments = all_centroid_assignments.argsort()

    if not rebalance:
        # make sure the sub-clusters sum up to exactly nc2
        cc = np.arange(nc1 + 1) * nc2 // nc1
        all_nc2 = cc[1:] - cc[:-1]
    else:
        bc_sum = np.cumsum(bin_count)
        all_nc2 = bc_sum * nc2 // bc_sum[-1]
        all_nc2[1:] -= all_nc2[:-1]
        assert sum(all_nc2) == nc2
        log(f"nb 2nd-level centroids {min(all_nc2)}-{max(all_nc2)}")

    # train sub-clusters
    i0 = 0
    c2 = []
    t0 = time.time()
    for c1 in range(nc1):
        nc2 = int(all_nc2[c1])
        log(f"[{time.time() - t0:.2f} s] training sub-cluster {c1}/{nc1} nc2={nc2}\r",
            end="", flush=True)
        i1 = i0 + bin_count[c1]
        subset = sorted_centroid_assignments[i0:i1]
        assert np.all(assign1[subset] == c1)
        km = faiss.Kmeans(d, nc2, **args)
        xtsub = xt[subset]
        km.train(xtsub)
        iteration_stats.append(km.iteration_stats)
        c2.append(km.centroids)
        del km
        i0 = i1
    log(f"done in {time.time() - t0:.2f} s")
    return np.vstack(c2), iteration_stats


def handle_pre_transforms(index, vector_size, save_path, name):

    # handle PreTransforms
    xt = get_random_vectors(vector_size*500, save_path, name)
    print(xt.shape)
    for i in range(index.chain.size()):
        print("i", i)
        vt = index.chain.at(i)
        vt.train(xt)
        xt = vt.apply(xt)
    # train_ivf_index_with_2level(index.index, **args)
    index.is_trained = True
    return faiss.downcast_index(index.index), vt


def train_ivf_index_with_2level(index, vector_size, save_path, name, **args):
    """
    Applies 2-level clustering to an index_ivf embedded in an index.
    """

    # handle PreTransforms
    # index = faiss.downcast_index(index)
    """if isinstance(index, faiss.IndexPreTransform):
        xt = get_random_vectors(vector_size*500, save_path, name)
        for i in range(index.chain.size()):
            vt = index.chain.at(i)
            vt.train(xt)
            #xt = vt.apply(xt)
        train_ivf_index_with_2level(index.index, **args)
        index.is_trained = True
        return
    index = faiss.downcast_index(index)"""

    ivf_index, vt = handle_pre_transforms(index, vector_size, save_path, name)
    assert isinstance(ivf_index, faiss.IndexIVF)
    assert ivf_index.metric_type == faiss.METRIC_L2

    # now do 2-level clustering
    nc1 = int(np.sqrt(index.nlist))  # number of clusters at top level
    print("REBALANCE=", args)
    centroids, _ = two_level_clustering(
        nc1, index.nlist, save_path, name, **args)
    index.quantizer.train(centroids)
    index.quantizer.add(centroids)

    # finish training (PQ and PCA)
    # 256 centroids times 256 samples per centroid for PQ; assumed good enough for PCA
    max_samples = 256*256
    # get samples from disk
    xt = get_random_vectors(max_samples, save_path, name)
    index.train(xt)


def get_random_vectors(n, save_path, name):

    lmdb_ids = lmdb_utils.get_lmdb_index_ids(save_path, name)
    n = min(n, len(lmdb_ids))
    # Create a random vector of size n, with each element being a random integer between min(lmdb_ids) and max(lmdb_ids)
    random_integers = np.random.randint(min(lmdb_ids), max(lmdb_ids), n)
    # Create a mapping of the random integers to lmdb_ids
    random_vector_ids = np.array([lmdb_ids[i] for i in random_integers])
    # Get the vectors from the lmdb for these ids
    vectors = lmdb_utils.get_lmdb_vectors_by_ids(
        save_path, name, random_vector_ids)
    # Convert the vectors to a numpy array
    vectors = np.array(vectors)

    return vectors


def get_batch(save_path, name, batches, i):
    batch_ids = batches[i]
    # Get the vectors from the lmdb for these ids
    vectors = lmdb_utils.get_lmdb_vectors_by_ids(save_path, name, batch_ids)
    return vectors


def break_into_batches(save_path, name, batch_size):
    # Break the vectors into batches of size batch_size
    lmdb_ids = lmdb_utils.get_lmdb_index_ids(save_path, name)

    # define the number of elements per batch
    num_elements_per_batch = np.ceil(len(lmdb_ids)/batch_size).astype(int)

    # define the batches
    batches = []
    for i in range(batch_size):
        batches.append(
            lmdb_ids[i*num_elements_per_batch:(i+1)*num_elements_per_batch])

    return batches
