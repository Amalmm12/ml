import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('iris.csv').dropna()
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print('Accuracy:', model.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=list(X.columns), filled=True)
plt.title('Decision Tree')
plt.show()
