import numpy as np


def validate_train(vector_dimension, pca_dimension, compressed_vector_bytes, opq_dimension):

    # Make sure pca, pq_bytes, and opq_dimension are integers and are all positive
    if not isinstance(pca_dimension, int):
        return False, "PCA is not the correct type. Expected type: int. Actual type: " + str(type(pca_dimension))
    if not isinstance(compressed_vector_bytes, int):
        return False, "PQ is not the correct type. Expected type: int. Actual type: " + str(type(compressed_vector_bytes))
    if not isinstance(opq_dimension, int):
        return False, "OPQ is not the correct type. Expected type: int. Actual type: " + str(type(opq_dimension))
    
    if pca_dimension < 1:
        return False, "PCA is not positive. PCA: " + str(pca_dimension)
    if compressed_vector_bytes < 1:
        return False, "PQ is not positive. PQ: " + str(compressed_vector_bytes)
    if opq_dimension < 1:
        return False, "OPQ is not positive. OPQ: " + str(opq_dimension)

    # Make sure PCA is less than the number of columns in the data
    if pca_dimension > vector_dimension:
        return False, "PCA is larger than the number of columns in the data. Number of columns in data: " + str(vector_dimension) + " PCA: " + str(pca_dimension)
    
    # OPQ has to be less than or equal to PCA
    if opq_dimension > pca_dimension:
        return False, "OPQ is larger than PCA. PCA: " + str(pca_dimension) + " OPQ: " + str(opq_dimension)
    
    # PCA has to be divisible by py_bytes
    if opq_dimension % compressed_vector_bytes != 0:
        return False, "OPQ is not divisible by PQ. PCA: " + str(opq_dimension) + " PQ: " + str(compressed_vector_bytes)
    
    return True, "Success"


def validate_add(vectors, text, vector_dimension):
        
    # Make sure the data is the correct type (probably a numpy array)
    if not isinstance(vectors, np.ndarray):
        return False, "Vectors are not the correct type. Expected type: numpy array. Actual type: " + str(type(vectors))

    # Double check that the vector is the right size first
    if vector_dimension != None and vectors.shape[1] != vector_dimension:
        return False, "Vector is not the correct size. Expected size: " + str(vector_dimension) + " Actual size: " + str(vectors.shape[1])

    # Check that the number of vectors is the same as the number of text items
    if vectors.shape[0] != len(text):
        return False, "Number of vectors does not match number of text items. Number of vectors: " + str(vectors.shape[0]) + " Number of text items: " + str(len(text))

    return True, "Success"


def validate_query(query_vector, vector_dimension):

    # Make sure the data is the correct type (numpy array)
    if not isinstance(query_vector, np.ndarray):
        return False, "Query vectors are not the correct type. Expected type: numpy array. Actual type: " + str(type(query_vector))
    
    # Make sure the query vector is the correct size. The 

    return True, "Success"