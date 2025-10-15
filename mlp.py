import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print(x_train.shape, x_test.shape)

def build_model(hidden_units=64, activation='relu'):
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_64 = build_model(hidden_units = 64)
history_64 = model_64.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

model_256 = build_model(hidden_units = 256)
history_256 = model_256.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

model_sigmoid = build_model(hidden_units = 64, activation = 'sigmoid')
history_sigmoid = model_sigmoid.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

test_loss, test_acc = model_64.evaluate(x_test, y_test)
print(f"Model (64 neurons, ReLU): accuracy = {test_acc:.4f}")

test_loss, test_acc = model_256.evaluate(x_test, y_test)
print(f"Model (256 neurons, ReLU): accuracy = {test_acc:.4f}")

test_loss, test_acc = model_sigmoid.evaluate(x_test, y_test)
print(f"Model (64 neurons, Sigmoid): accuracy = {test_acc:.4f}")

plt.plot(history_64.history['val_accuracy'], label='64 neurons (ReLU)')
plt.plot(history_256.history['val_accuracy'], label='256 neurons (ReLU)')
plt.plot(history_sigmoid.history['val_accuracy'], label='64 neurons (Sigmoid)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
