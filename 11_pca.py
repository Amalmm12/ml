import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('iris.csv').dropna()
X = df.drop(columns='label')
y = df['label']
X = StandardScaler().fit_transform(X)

model = PCA(n_components=2)
X_pca = model.fit_transform(X)
print('Variance ratios:', model.explained_variance_ratio_)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()
