import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


patch_size = int(sys.argv[1])

# Load the NPZ file
data = np.load(f'embeddings_dataset/combined_dataset_hoptimus0.npz')

# Print all keys in the dataset
print("Keys in the dataset:")
for key in data.keys():
    print(f"- {key}")
    # Get the shape of each array
    # array = data[key]
    # print(f"  Shape: {array.shape}")
    # print(f"  Number of datapoints: {len(array)}")

print(data['embeddings'].shape)
print(data['gene_expression'].shape)

def log1p_normalization(arr):
    """  Apply log1p normalization to the given array """

    scale_factor = 100
    return np.log1p((arr / np.sum(arr, axis=1, keepdims=True)) * scale_factor)

# Load and normalize the data
embeddings = data['embeddings']
gene_expression = data['gene_expression']
gene_expression_normalized = log1p_normalization(gene_expression)

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, gene_expression_normalized, 
    test_size=0.2, random_state=42
)

# Initialize models
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42
    )
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate L2 error
    train_l2 = float(np.mean((train_preds - y_train) ** 2))
    test_l2 = float(np.mean((test_preds - y_test) ** 2))
    
    # Calculate correlations (averaged across all genes)
    train_spearman = np.mean([spearmanr(y_train[:, i], train_preds[:, i])[0] for i in range(y_train.shape[1])])
    test_spearman = np.mean([spearmanr(y_test[:, i], test_preds[:, i])[0] for i in range(y_test.shape[1])])
    
    results[name] = {
        'train_l2': train_l2, 
        'test_l2': test_l2,
        'train_spearman': train_spearman,
        'test_spearman': test_spearman
    }
    print(f"{name}:")
    print(f"  Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}")
    print(f"  Train Spearman: {train_spearman:.4f}, Test Spearman: {test_spearman:.4f}")

"""
# Plot results
plt.figure(figsize=(10, 6))
x = np.arange(len(results))
width = 0.35

plt.bar(x - width/2, [r['train_l2'] for r in results.values()], width, label='Train')
plt.bar(x + width/2, [r['test_l2'] for r in results.values()], width, label='Test')

plt.xlabel('Models')
plt.ylabel('L2 Error')
plt.title('Model Comparison: Train vs Test L2 Error')
plt.xticks(x, results.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

"""

