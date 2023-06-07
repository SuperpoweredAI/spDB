# spDB -  an extremely memory-efficient vector database
Existing open source vector databases are built on HNSW indexes that must be held entirely in memory to be used. This uses an extremely large amount of memory, which severely limits the sizes of vector DBs that can be used locally, and creates very high costs for cloud deployments.

It’s possible to build a vector database with extremely low memory requirements that still has high recall and low latency. The key is to use a highly compressed search index, combined with reranking from disk, as demonstrated in the [Zoom](https://arxiv.org/abs/1809.04067) paper. This project implements the core technique introduced in that paper. We also implement a novel adaptation of Faiss's two-level k-means clustering algorithm that only requires a small subset of vectors to be held in memory at an given point.

With spDB, you can index and query 100M 768d vectors with peak memory usage of around 3GB. With an in-memory vector DB, you would need ~340GB of RAM.

## Usage
```python
import spdb

db = spdb.spDB()
db.add(vectors)
db.train()

results = db.query(query_vector)
```

## Architecture overview
spDB uses a two-step process to perform approximate nearest neighbors search. First, a highly compressed Faiss index is searched to find the `preliminary_top_k` (set to 500 by default) results. Then the full uncompressed vectors for these results are retrieved from a key-value store on disk, and a k-nearest neighbors search is performed on these vectors to arrive at the `final_top_k` results.

## Parameters
spDB has a few parameters you can adjust to control the tradeoff between memory usage, recall, and latency. You may also need to adjust certain parameters based on the size of your vectors. The default parameters were designed around 768d vectors, with a high focus placed on low memory usage.

- `pca_dimension`: Principal Component Analysis (PCA) is the first step in the compression process, and this parameter defines how many dimensions to reduce the vector to with PCA. You can usually do a 2-4x reduction without substantially hurting recall.
- `opq_dimension`: Optimized Product Quantization (OPQ) is the second compression step. The main purpose of this step is to prepare the vector for the product quantization step that follows, but it's also common to do a 2x dimensionality reduction in this step. One thing to keep in mind is that the value for this parameter has to be divisible by the following parameter, `compressed_vector_bytes`, and it's recommended that this value is 4x that value.
- `compressed_vector_bytes`: The final, and most important, compression step is Product Quantization (PQ). This parameter controls the size of the final compressed vectors, measured in bytes. Since the compressed vectors are the primary thing stored in memory, this parameter has a direct impact on memory usage. 32 and 64 are generally the best values for this parameter, but you could also try 16 if you want an extremely compressed index, or 128 if you don't need aggressive compression. For reference, uncompressed vectors use four bytes per dimension, so uncompressed 768d vectors use 3,072 bytes per vector.

## Contributing
We are open to contributions of all types.
