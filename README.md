# spDB -  an extremely memory-efficient vector database
Existing open source vector databases are built on HNSW indexes that must be held entirely in memory to be used. This uses an extremely large amount of memory, which severely limits the sizes of vector DBs that can be used locally, and creates very high costs for cloud deployments.

It’s possible to build a vector database with extremely low memory requirements that still has high recall and low latency. The key is to use a highly compressed search index, combined with reranking from disk, as demonstrated in the [Zoom](https://arxiv.org/abs/1809.04067) paper. This project implements the core technique introduced in that paper. We also implement a novel adaptation of Faiss's two-level k-means clustering algorithm that only requires a small subset of vectors to be held in memory at an given point.

With spDB, you can index and query 100M 768d vectors with peak memory usage of around 3GB. With an in-memory vector DB, you would need ~340GB of RAM. This means you could easily index and query all of Wikipedia on an average Macbook.

## Architecture overview
spDB uses a two-step process to perform approximate nearest neighbors search. First, a highly compressed Faiss index is searched to find the `preliminary_top_k` (set to 500 by default) results. Then the full uncompressed vectors for these results are retrieved from a key-value store on disk, and a k-nearest neighbors search is performed on these vectors to arrive at the `final_top_k` results.

## Basic usage guide
Install using pip: `pip install spdb`

```python
from spdb import spDB

db = spDB(name="Example")
db.add(vectors, text)
db.train()

results = db.query(query_vector)
```

By default, all spDB databases are saved to the ~/.spdb directory. This directory is created automatically if it doesn’t exist when you initialize an spDB object. You can override this path by specifying a save_path when you create your spDB object.

## Adding and removing items
To add vectors to your database, use the db.add() method. This method takes a list of (vector, metadata) tuples, where each vector is itself a list, and each metadata item is a dictionary with keys of your choosing.

## How and when to train the index
In order to query your spDB database, you’ll need to train the search index. You can do this by running the db.train() method. The index training process exploits patterns in your vectors to enable more efficient search. In general, you want to add your vectors and then train the index. For more details on exactly when to train and potentially retrain your index, check out our wiki page [here](https://github.com/SuperpoweredAI/spDB/wiki/Search-index-training).

## Metadata
You can add metadata to each vector by including a metadata dictionary. You can include whatever metadata fields you want, but the keys and values should all be serializable.

Metadata filtering is the next major feature that will be added. This will allow you to use SQL-like statements to control which items get searched over.

## FastAPI server deployment
To deploy your database as a server with a REST API, you can just run fastapi.py as a script. This will start a FastAPI server instance. You can then make API calls to it using the following endpoints:
/db/create
…

## Limitations
- spDB uses a simple embedded database architecture, not a client-server architecture, so it may not be ideal for certain kinds of large-scale production applications.
- One of the main dependencies, Faiss, doesn't play nice with Apple M1/M2 chips. You may be able to get it to work by building it from source, but we haven't successfully done so yet.
- We haven't tested it on datasets larger than 35M vectors yet. It should still work well up to 100-200M vectors, but beyond that performance may start to deteriorate.

## Additional documentation
- [Tunable parameters](https://github.com/SuperpoweredAI/spDB/wiki/Tunable-parameters)
- [Contributing](https://github.com/SuperpoweredAI/spDB/wiki/Contributing)
- [Development roadmap](https://github.com/SuperpoweredAI/spDB/wiki/Development-roadmap)
