import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import umap

# 1. Generate synthetic classification data with 4 classes and 100 features
X, y = make_classification(n_samples=500, n_features=100, n_informative=20,
                           n_redundant=10, n_classes=2, n_clusters_per_class=1,
                           random_state=42)

# 2. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. UMAP reduction to 2D
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 4. Logistic Regression for feature importance
clf = LogisticRegression(multi_class='ovr', solver='liblinear')
clf.fit(X_scaled, y)
importance = np.mean(np.abs(clf.coef_), axis=0)  # average importance across classes

# 5. Plot UMAP
plt.figure(figsize=(5, 8))

plt.subplot(2, 1, 2)
for i in range(2):
    plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=f'Class {i}', s=20)
plt.title("UMAP Projection")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(loc='upper left', fontsize=10)

# 6. Plot Feature Importance
plt.subplot(2, 1, 1)
top_indices = np.argsort(importance)[-10:][::-1]
plt.bar(range(len(top_indices)), importance[top_indices])
# plt.xticks(range(len(top_indices)), top_indices, rotation=90)
plt.title("Top 10 Feature Importances")
# plt.xlabel("Feature Index")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()
