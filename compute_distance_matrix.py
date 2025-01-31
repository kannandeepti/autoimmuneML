import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
import time
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def log1p_normalization(arr):
    """  Apply log1p normalization to the given array """

    scale_factor = 100
    return np.log1p((arr / np.sum(arr, axis=1, keepdims=True)) * scale_factor)

def nearest_neighbor_predict_chunk(X_train, y_train, X_test, k=200, chunk_size=100):
    """
    Memory-efficient k-nearest neighbors implementation that processes test samples in small chunks
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training embeddings
    y_train : np.ndarray 
        Training labels/values
    X_test : np.ndarray
        Test embeddings
    k : int
        Number of nearest neighbors
    chunk_size : int
        Size of chunks for processing test samples (smaller = less memory usage)
    """
    n_test = X_test.shape[0]
    n_features = y_train.shape[1]
    predictions = np.zeros((n_test, n_features))
    
    # Process test samples in small chunks to conserve memory
    for i in range(0, n_test, chunk_size):
        chunk_end = min(i + chunk_size, n_test)
        chunk_size_actual = chunk_end - i
        
        # Pre-allocate distance matrix for current chunk
        distances = np.zeros((chunk_size_actual, len(X_train)))
        
        # Calculate distances one test sample at a time
        for j in range(chunk_size_actual):
            # Compute squared distances efficiently using numpy operations
            diff = X_train - X_test[i + j]
            distances[j] = np.sum(diff * diff, axis=1)
            
        # Find k nearest neighbors for all samples in chunk at once
        neighbor_indices = np.argpartition(distances, k, axis=1)[:, :k]
        
        # Calculate mean predictions for chunk
        chunk_predictions = np.mean(y_train[neighbor_indices], axis=1)
        predictions[i:chunk_end] = chunk_predictions
        
        # Clear some memory
        del distances, neighbor_indices, chunk_predictions
        
    return predictions

def nearest_neighbor_predict_regular(X_train, y_train, X_test, k=200):
    """Find k nearest neighbors in training set for each test point and return average of their y values"""
    predictions = []
    for i, test_embedding in enumerate(X_test):
        # Calculate distances to all training points
        distances = np.sqrt(np.sum((X_train - test_embedding) ** 2, axis=1))
        
        # If the test point is in the training set, exclude it
        if X_test is X_train:
            distances[i] = np.inf
            
        # Find k nearest neighbors
        nn_indices = np.argpartition(distances, k)[:k]
        # Use average of their gene expressions as prediction
        predictions.append(np.mean(y_train[nn_indices], axis=0))

    return np.array(predictions)


def process_chunk(chunk_data, X_train, y_train, k):
    """
    Process a single chunk of test data
    
    Parameters:
    -----------
    chunk_data : tuple
        (start_idx, chunk_X_test) containing the start index and chunk of test data
    X_train : np.ndarray
        Training embeddings
    y_train : np.ndarray
        Training labels/values
    k : int
        Number of nearest neighbors
    """
    start_idx, chunk_X_test = chunk_data
    chunk_size = len(chunk_X_test)
    chunk_predictions = np.zeros((chunk_size, y_train.shape[1]))
    
    # Process each sample in the chunk
    for i in range(chunk_size):
        # Calculate distances to all training points
        diff = X_train - chunk_X_test[i]
        distances = np.sum(diff * diff, axis=1)
        
        # Find k nearest neighbors
        nn_indices = np.argpartition(distances, k)[:k]
        # Calculate mean prediction
        chunk_predictions[i] = np.mean(y_train[nn_indices], axis=0)
    
    return (start_idx, chunk_predictions)

def nearest_neighbor_predict_multiprocessing(X_train, y_train, X_test, k=200, chunk_size=100, n_processes=12):
    """
    Parallel implementation of k-nearest neighbors using multiprocessing
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training embeddings
    y_train : np.ndarray 
        Training labels/values
    X_test : np.ndarray
        Test embeddings
    k : int
        Number of nearest neighbors
    chunk_size : int
        Size of chunks for parallel processing
    n_processes : int
        Number of parallel processes to use
    """
    n_test = X_test.shape[0]
    n_features = y_train.shape[1]
    predictions = np.zeros((n_test, n_features))
    
    # Prepare chunks of data
    chunks = []
    for start_idx in range(0, n_test, chunk_size):
        end_idx = min(start_idx + chunk_size, n_test)
        chunks.append((start_idx, X_test[start_idx:end_idx]))
    
    # Create a partial function with fixed arguments
    process_chunk_partial = partial(process_chunk, X_train=X_train, y_train=y_train, k=k)
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        for start_idx, chunk_predictions in tqdm(pool.imap(process_chunk_partial, chunks), 
                                               total=len(chunks)):
            end_idx = min(start_idx + chunk_size, n_test)
            predictions[start_idx:end_idx] = chunk_predictions
    
    return predictions

# Load the NPZ file
print("loading data")
embeddings_table = np.load(os.path.join("broad-1-aecv1/embeddings_dataset", f"combined_dataset.npz"))

# Load and normalize the data
print("normalizing data")
embeddings = embeddings_table['embeddings'][::50]
gene_expression = embeddings_table['gene_expression'][::50]
gene_expression_normalized = log1p_normalization(gene_expression)

# Create train/test split
print("creating train/test split")
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, gene_expression_normalized, 
    test_size=0.2, random_state=42
)


print("computing distance matrix using all three methods")

# Test regular method
# start_time = time.time()
# predictions_regular = nearest_neighbor_predict_regular(X_train, y_train, X_test)
# regular_time = time.time() - start_time
# print(f"Regular prediction took {regular_time:.2f} seconds")

# # Test chunked method
# start_time = time.time()
# predictions_chunked = nearest_neighbor_predict_chunk(X_train, y_train, X_test)
# chunked_time = time.time() - start_time
# print(f"Chunked prediction took {chunked_time:.2f} seconds")

# Test multiprocessing method
start_time = time.time()
predictions_multiprocessing = nearest_neighbor_predict_multiprocessing(X_train, y_train, X_test)
multiprocessing_time = time.time() - start_time
print(f"Multiprocessing prediction took {multiprocessing_time:.2f} seconds")

# Verify results are the same (within numerical precision)
# print("\nVerifying results match:")
# print("Regular vs Chunked:", np.allclose(predictions_regular, predictions_chunked, rtol=1e-5, atol=1e-8))
# print("Regular vs Multiprocessing:", np.allclose(predictions_regular, predictions_multiprocessing, rtol=1e-5, atol=1e-8))
# print("Chunked vs Multiprocessing:", np.allclose(predictions_chunked, predictions_multiprocessing, rtol=1e-5, atol=1e-8))

print("\nDone")
