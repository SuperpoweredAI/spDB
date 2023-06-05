# spDB -  an extremely memory-efficient vector database

Existing open source vector databases are built on HNSW indexes that must be held entirely in memory to be used. This uses an extremely large amount of memory, which severely limits the sizes of vector DBs that can be used locally, and creates very high costs for cloud deployments.

It’s possible to build a vector database with extremely low memory requirements that still has high recall and low latency. The key is to use a highly compressed search index, combined with reranking from disk, as demonstrated in the [Zoom](https://arxiv.org/abs/1809.04067) paper. This project implements the core technique introduced in that paper. We also implement a novel adaptation of Faiss's two-level k-means clustering algorithm that only requires a small subset of vectors to be held in memory at an given point.

With spDB, you can index and query 100M 768d vectors with peak memory usage of around 3GB. With an in-memory vector DB, you would need ~340GB of RAM.
