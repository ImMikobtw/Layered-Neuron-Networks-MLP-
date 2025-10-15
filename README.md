# Layered-Neuron-Networks-MLP
# Laboratory Work №2 — Multilayer Perceptron (MLP)

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

## ⚙️ Steps of Implementation

### 1. Data Preparation (MNIST)
```python
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28*28)).astype("float32") / 255.0

```
