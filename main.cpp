#include "neuralnet.hpp"
#include "utils.hpp"

using namespace std;

int main() {
    NeuralNet net;
    net.add_layer(Layer(784, 512, "relu"));  
    net.add_layer(Layer(512, 512, "relu"));
    net.add_layer(Layer(512, 10, "softmax"));

    Matrix X_train, Y_train;
    load_csv("D:/Job Prep/NeuralNetwork/data/mnist/mnist_train.csv", X_train, Y_train, 50);

    Matrix X_val, Y_val;
    load_csv("D:/Job Prep/NeuralNetwork/data/mnist/mnist_test.csv", X_val, Y_val, 200);

    net.train(X_train, Y_train, X_val, Y_val, 15, 0.01, 256);
    return 0;
}

