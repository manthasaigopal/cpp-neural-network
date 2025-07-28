# C++ Neural Network Framework
This repository provides a lightweight, dependency-free implementation of a fully connected neural network, built entirely from scratch in modern C++ — including all matrix operations without using any external libraries. It allows you to construct custom model architectures by stacking layers with configurable activation functions and train them end-to-end using mini-batch gradient descent and backpropagation. Designed for classification tasks (like digit recognition), it can be easily extended for regression problems with minimal changes. The framework supports loading datasets in CSV format (<label, features>), and includes functionality to train, evaluate, save, and load models — all in pure C++.  

## Table of Contents
- [Features](#features)
  - [Matrix Class](#matrix-class)
  - [Layer Class](#layer-class)
  - [NeuralNet Class](#neuralnet-class)
  - [Utilities](#utilities)
- [Sample Usage](#sample-usage)
- [Build and Run](#build-and-run)

## Features
### Matrix Class
A custom `Matrix` class powering all numerical operations in the network. It includes:
- Matrix creation, resizing
- Element-wise operations: `add`, `subtract`, `multiply`
- Matrix multiplication
- Transpose
- Broadcasting support for vectors (e.g. adding biases)
- Printing and shape utilities

### Layer Class
Each layer in the neural network is implemented as a modular class with full support for forward and backward propagation, storing intermediate values(during the forward propagation) required for backpropagation.
* **Weights and Bias Storage:**
  Each `Layer` instance maintains its own:
  * `weights`: weight matrix (`output_size × input_size`)
  * `bias`: bias vector (`output_size × 1`)

* **Activation Support:**
  Activation functions are applied during forward pass:
  * Supports `sigmoid`, `relu` and `softmax`.

* **Forward Propagation:**

* **Backward Propagation:**
  * Uses chain rule: computes gradients of loss w\.r.t. weights, biases, and inputs.
  * Updates parameters using gradient descent and returns `dA` for the previous layer.

* **Flexible Initialization**
  * Can either initialize layers from scratch or load them with pre-defined parameters—making the class suitable for training from scratch or inference with saved models.


### NeuralNet Class
The `NeuralNet` class is a self-contained framework for training and evaluating fully connected neural networks from scratch — **no deep learning libraries used**. It offers:
* **Modular Layer Construction**
  * Easily build custom architectures by stacking layers using `add_layer()`. Mix and match activation functions and dimensions.

* **End-to-End Forward Pass**
  * Executes data through all layers.

* **Backpropagation & Learning**
  * Implements full backpropagation from scratch with gradient descent. Adjusts weights using only linear algebra and calculus.

* **Mini-Batch Training with Validation**
  `train()` method supporting:
  * Mini-batch gradient descent
  * Setting Learning rate
  * Validation set evaluation at each epoch

* **Predictions**
  `predict()` returns:
  * Class Label
  * Confidence score

* **Model Persistence**
  * Supports saving trained models to disk and reloading them later for inference or further training.

### Utilities
- **`load_csv()`**  
  Parses CSV datasets into feature (`X`) and label (`Y`) matrices.  
  Includes an optional `limit` parameter to load only a subset of the data — useful for quicker testing and debugging.

- **`compute_accuracy()`** 
  Computes classification accuracy by comparing predicted outputs with ground-truth labels.



## Sample Usage
```cpp
#include "neuralnet.hpp"
#include "utils.hpp"

using namespace std;

int main() {
    NeuralNet net;

    // Define a fully connected neural network
    net.add_layer(Layer(784, 512, "relu"));  
    net.add_layer(Layer(512, 512, "relu"));
    net.add_layer(Layer(512, 10, "softmax"));

    // Load training data (with optional sample limit)
    Matrix X_train, Y_train;
    load_csv("data/mnist/mnist_train.csv", X_train, Y_train, 50000);

    Matrix X_val, Y_val;
    load_csv("data/mnist/mnist_test.csv", X_val, Y_val, -1);

    // Train the model
    net.train(X_train, Y_train, X_val, Y_val,
              epochs = 15,
              learning_rate = 0.01,
              batch_size = 32);

    // Save the trained model for either inference or continued training
    net.save_model("models/model.txt");

    return 0;
}
```

## Build and Run
Using CMake:
```bash
git clone https://github.com/manthasaigopal/cpp-neural-network.git
cd cpp-neural-network
mkdir build && cd build
cmake ..
cmake --build .
```

Finally run the executable 
```bash
neural_net.exe
```
