import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('iris.csv').dropna()
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def fuzzy(X):
    return np.hstack([np.exp(-(X - c) ** 2) for c in (-1, 0, 1)])

model = MLPClassifier(hidden_layer_sizes=(12,),
                      max_iter=800, random_state=42)
model.fit(fuzzy(X_train), y_train)
print('Accuracy:', model.score(fuzzy(X_test), y_test))

plt.plot(model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Fuzzy-Feature Neural Network')
plt.show()
