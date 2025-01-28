import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import warnings
from scipy.stats import ConstantInputWarning

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

# Load the NPZ file
data = np.load('embeddings_dataset/combined_dataset.npz')

# Load and normalize the data
embeddings = data['embeddings'][::10]
gene_expression = data['gene_expression'][::10]
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
test_preds = nearest_neighbor_predict(X_train, y_train, X_test)

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