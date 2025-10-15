# Multilayer Perceptron (MLP)

## Topic
**Multilayer Neural Networks (Multilayer Perceptron, MLP)**

---

## Objective
To understand the architecture of multilayer neural networks and the **backpropagation** algorithm used to train them.

---

## Theoretical Background
A **Multilayer Perceptron (MLP)** is a feedforward neural network that contains one or more **hidden layers** between the input and output layers.  
These hidden layers allow the network to learn more complex, non-linear relationships.

**Backpropagation** is the algorithm used to minimize the model’s error by propagating it backward from the output layer to the input layer, updating the weights at each step.

MLPs are widely used for **classification tasks**, such as handwritten digit recognition using the **MNIST dataset**.

---

### How to Run
```bash
     pip install tensorflow matplotlib
     python lab2_mlp.py
```


## Steps of Implementation

### 1. Data Preparation (MNIST)
```python
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28*28)).astype("float32") / 255.0

```

### 2. Building the Model
```python
def build_model(hidden_units=64, activation='relu'):
    model = keras.Sequential([
        layers.Dense(hidden_units, activation=activation, input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

### 3. Training
```python
model = build_model(hidden_units=64)
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

model_256 = build_model(hidden_units=256)
history_256 = model_256.fit(x_train, y_train, epochs=5, validation_split=0.2)

model_sigmoid = build_model(hidden_units=64, activation='sigmoid')
history_sigmoid = model_sigmoid.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

### 4. Evaluation
```python
test_loss, test_acc = model_64.evaluate(x_test, y_test)
print(f"Model (64 neurons, ReLU): accuracy = {test_acc:.4f}")

test_loss, test_acc = model_256.evaluate(x_test, y_test)
print(f"Model (256 neurons, ReLU): accuracy = {test_acc:.4f}")

test_loss, test_acc = model_sigmoid.evaluate(x_test, y_test)
print(f"Model (64 neurons, Sigmoid): accuracy = {test_acc:.4f}")
```

### 5. Visualization
```python
plt.plot(history_64.history['val_accuracy'], label='64 neurons (ReLU)')
plt.plot(history_256.history['val_accuracy'], label='256 neurons (ReLU)')
plt.plot(history_sigmoid.history['val_accuracy'], label='64 neurons (Sigmoid)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```


Author:  Miras Tleusserik
Course: Neural Networks
Lab #2: Multilayer Perceptron (MLP)
