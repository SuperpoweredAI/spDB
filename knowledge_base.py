import pickle
import faiss

class KnowledgeBase:
    def __init__(self, name, save_path=None):
        self.name = name
        self.save_path = save_path
        self.faiss_index = None
        self.memmap = None
        self.vector_size = None
        self.embedding_function = None

    def save(self):
        # save faiss index and delete (so it doesn't get pickled)
        faiss.write_index(self.faiss_index, self.save_path + f'{self.name}.index')

        # save object to pickle file
        pickle.dump(self, open(self.save_path, 'wb'))

    def train(self, data):
        # train faiss index
        pass

    def add(self, vector):
        # add vector to memmap
        # add vector to faiss index
        # add text to LMDB

        pass

    def remove(self, vector_ids):
        # remove vector from memmap
        # remove vector from faiss index
        # remove text from LMDB

        pass

    def query(self, query_vectors, top_k=100):
        # query faiss index
        # query memmap
        # brute force search full vectors to find true top_k
        # retrieve text for top_k results from LMDB
        pass


def create_knowledge_base(name):
    kb = KnowledgeBase(name)

    return kb

def load_knowledge_base(name, save_path):
    # load KnowledgeBase object from pickle file
    kb = pickle.load(open(name, 'rb'))

    # load faiss index from save path
    kb.faiss_index = faiss.read_index(kb.save_path + f'{name}.index')

    # load memmap

    return kb
