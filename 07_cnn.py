import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

df = pd.read_csv('digits.csv').dropna()
X = df.drop(columns='label').to_numpy().reshape(-1, 8, 8, 1)
X = (X / 16.0).astype('float32')
y = df['label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5,
                    validation_data=(X_test, y_test), verbose=0)
print('Accuracy:', model.evaluate(X_test, y_test, verbose=0)[1])

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Accuracy')
plt.show()
