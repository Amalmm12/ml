import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

df = pd.read_csv('iris.csv').dropna()
A = df[['sepal_length', 'sepal_width']]
B = df[['petal_length', 'petal_width']]
A = StandardScaler().fit_transform(A)
B = StandardScaler().fit_transform(B)

model = CCA(n_components=1)
A_cca, B_cca = model.fit_transform(A, B)
corr = np.corrcoef(A_cca[:, 0], B_cca[:, 0])[0, 1]
print('Canonical correlation:', corr)

plt.scatter(A_cca[:, 0], B_cca[:, 0])
plt.xlabel('Canonical score A')
plt.ylabel('Canonical score B')
plt.title('CCA')
plt.show()
