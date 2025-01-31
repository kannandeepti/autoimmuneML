import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import warnings
from scipy.stats import ConstantInputWarning
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

patch_size = int(sys.argv[1])

def log1p_normalization(arr):
    """  Apply log1p normalization to the given array """
    scale_factor = 100
    return np.log1p((arr / np.sum(arr, axis=1, keepdims=True)) * scale_factor)

def nearest_neighbor_predict(X_train, y_train, X_test, k=200):
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
data = np.load(f'embeddings_dataset/combined_dataset_hoptimus0.npz')

# Load and normalize the data
embeddings = data['embeddings']
gene_expression = data['gene_expression']
gene_expression_normalized = log1p_normalization(gene_expression)

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, gene_expression_normalized, 
    test_size=0.2, random_state=42
)

print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)

print("\nTraining Nearest Neighbor model...")
# Make predictions
# train_preds = nearest_neighbor_predict(X_train, y_train, X_train)
test_preds = nearest_neighbor_predict_multiprocessing(X_train, y_train, X_test, n_processes=40)

print("test_preds.shape: ", test_preds.shape)

print(test_preds)
print(y_test)

# Calculate L2 error
# train_l2 = float(np.mean((train_preds - y_train) ** 2))
test_l2 = float(np.mean((test_preds - y_test) ** 2))

# Calculate correlations (averaged across all genes)
# train_correlations = [spearmanr(y_train[:, i], train_preds[:, i])[0] for i in range(y_train.shape[1])]
# test_correlations = [spearmanr(y_test[:, i], test_preds[:, i])[0] for i in range(y_test.shape[1])]

train_correlations = []
test_correlations = []

# for i in range(y_train.shape[1]):
#     # Compute Pearson correlation
#     with warnings.catch_warnings():
#         warnings.filterwarnings("error", category=ConstantInputWarning)
#         try:
#             pearson_corr, _ = pearsonr(y_train[:, i], train_preds[:, i])
#             pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
#         except ConstantInputWarning:
#             pearson_corr = 0.0
#         train_correlations.append(pearson_corr)

for i in range(y_test.shape[1]):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConstantInputWarning)
        try:
            pearson_corr, _ = pearsonr(y_test[:, i], test_preds[:, i])
            pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
        except ConstantInputWarning:
            pearson_corr = 0.0
        test_correlations.append(pearson_corr)

# train_spearman = np.mean([corr for corr in train_correlations if not np.isnan(corr)])
test_spearman = np.mean([corr for corr in test_correlations if not np.isnan(corr)])

print("Nearest Neighbor Results:")
# print(f"  Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}")
print(f"  Test L2: {test_l2:.4f}")
# print(f"  Train Spearman: {train_spearman:.4f}, Test Spearman: {test_spearman:.4f}")
print(f"  Test Spearman: {test_spearman:.4f}")

# # Plot results
# plt.figure(figsize=(6, 4))
# x = np.arange(1)
# width = 0.35

# plt.bar(x - width/2, [train_l2], width, label='Train')
# plt.bar(x + width/2, [test_l2], width, label='Test')

# plt.xlabel('Model')
# plt.ylabel('L2 Error')
# plt.title('Nearest Neighbor: Train vs Test L2 Error')
# plt.xticks(x, ['Nearest Neighbor'])
# plt.legend()
# plt.tight_layout()
# plt.savefig('nn_comparison.png', bbox_inches='tight', dpi=300)
# plt.close()