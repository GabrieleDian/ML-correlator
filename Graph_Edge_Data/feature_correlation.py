import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Path to your features folder
features_dir = "features_loop_9"

# Load all .npy features
features = {}
for fname in os.listdir(features_dir):
    if fname.endswith(".npy"):
        key = os.path.splitext(fname)[0]
        features[key] = np.load(os.path.join(features_dir, fname))

print(f"Loaded {len(features)} features.")

# Step 1: Check exact duplicates
print("\n=== Checking for identical features ===")
for f1, f2 in itertools.combinations(features.keys(), 2):
    if np.array_equal(features[f1], features[f2]):
        print(f"⚠️ {f1} and {f2} are identical")

# Step 2: Compute correlations between features
print("\n=== Computing correlations ===")
feature_names = list(features.keys())
n = len(feature_names)

corr_matrix = np.zeros((n, n))

def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

for i in range(n):
    for j in range(n):
        corr_matrix[i, j] = safe_corr(features[feature_names[i]].ravel(),
                                      features[feature_names[j]].ravel())

# Step 3: Report highly correlated pairs
threshold = 0.99
for i in range(n):
    for j in range(i + 1, n):
        if abs(corr_matrix[i, j]) >= threshold:
            print(f"🔗 {feature_names[i]} and {feature_names[j]} correlation = {corr_matrix[i,j]:.4f}")

# Step 4: Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(n), feature_names, rotation=90)
plt.yticks(range(n), feature_names)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
