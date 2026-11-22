import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('iris.csv').dropna()
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', model.score(X_test, y_test))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title('Naive Bayes Confusion Matrix')
plt.show()
