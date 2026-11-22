import pandas as pd
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

model = MLPClassifier(hidden_layer_sizes=(16, 8),
                      max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print('Accuracy:', model.score(X_test, y_test))

plt.plot(model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('FFNN Training Loss')
plt.show()
