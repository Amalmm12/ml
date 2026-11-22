import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv('iris.csv').dropna()
X = df.drop(columns='label')
y = df['label']
X = StandardScaler().fit_transform(X)

model = LinearDiscriminantAnalysis(n_components=2)
X_lda = model.fit_transform(X, y)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection')
plt.show()
