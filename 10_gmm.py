import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

df = pd.read_csv('iris.csv').dropna()
X = df[['petal_length', 'petal_width']]
X = StandardScaler().fit_transform(X)

model = GaussianMixture(n_components=3, random_state=42)
clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Scaled petal length')
plt.ylabel('Scaled petal width')
plt.title('GMM Clustering')
plt.show()
