import faiss
import numpy as np
import time

def print_nop(*arg, **kwargs):
    pass

def two_level_clustering(nc1, nc2, rebalance=True, clustering_niter=25, **args):
    """
    perform 2-level clustering on a training set xt
    nc1 and nc2 are the number of clusters at each level, the final number of
    clusters is nc2. Additional arguments are passed to the Kmeans object.

    Rebalance allocates the number of sub-clusters depending on the number of
    first-level assignment.
    """
    d = xt.shape[1]

    verbose = args.get("verbose", False)

    log = print if verbose else print_nop

    log(f"2-level clustering of {xt.shape} nb 1st level clusters = {nc1} total {nc2}")
    log("perform coarse training")

    # get training set for top level
    max_samples = nc1*256 # max of 256 samples per centroid
    xt = get_random_vectors(max_samples)

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

    # Assign the data to the coarse centroids in batches
    num_batches = 100
    all_assign1 = []
    for i in range(num_batches):
        x = get_batch(i, num_batches)
        #_, I = km.index.search(x, 1)
        _, assign1 = km.assign(x)
        all_assign1.append(assign1)
    all_assign1 = np.concatenate(all_assign1)
    bc = np.bincount(all_assign1, minlength=nc1)
    o = all_assign1.argsort()
    del km

    if not rebalance:
        # make sure the sub-clusters sum up to exactly nc2
        cc = np.arange(nc1 + 1) * nc2 // nc1
        all_nc2 = cc[1:] - cc[:-1]
    else:
        bc_sum = np.cumsum(bc)
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
        log(f"[{time.time() - t0:.2f} s] training sub-cluster {c1}/{nc1} nc2={nc2}\r", end="", flush=True)
        i1 = i0 + bc[c1]
        subset = o[i0:i1]
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

def train_ivf_index_with_2level(index, **args):
    """
    Applies 2-level clustering to an index_ivf embedded in an index.
    """

    # handle PreTransforms
    index = faiss.downcast_index(index)
    if isinstance(index, faiss.IndexPreTransform):
        for i in range(index.chain.size()):
            vt = index.chain.at(i)
            vt.train(xt)
            xt = vt.apply(xt)
        train_ivf_index_with_2level(index.index, **args)
        index.is_trained = True
        return
    assert isinstance(index, faiss.IndexIVF)
    assert index.metric_type == faiss.METRIC_L2
    
    # now do 2-level clustering
    nc1 = int(np.sqrt(index.nlist)) # number of clusters at top level
    print("REBALANCE=", args)
    centroids, _ = two_level_clustering(nc1, index.nlist, **args)
    index.quantizer.train(centroids)
    index.quantizer.add(centroids)
    
    # finish training (PQ and PCA)
    max_samples = 256*256 # 256 centroids times 256 samples per centroid for PQ; assumed good enough for PCA
    # get samples from disk
    xt = get_random_vectors(max_samples)
    index.train(xt)

# TODO: implement this function
def get_random_vectors(n):
    pass

# TODO: implement this function
def get_batch(i, num_batches):
    pass