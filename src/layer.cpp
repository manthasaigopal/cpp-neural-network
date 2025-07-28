#include "layer.hpp"
#include <iostream>
using namespace std;

Layer::Layer(int input_size, int output_size, std::string activation_type) : weights(output_size, input_size), bias(output_size, 1), activation_type(activation_type) {
    weights.xavier_init(input_size);
    bias.random_init(0.0, 0.0);
}

Layer::Layer(const Matrix &w, const Matrix &b, const std::string &activation_type) : weights(w), bias(b), activation_type(activation_type) {}


Matrix Layer::forward(const Matrix &input) {
    this->input = input;

    z = Matrix::multiply(this->weights, input);
    z.add(bias);

    a = z;
    if (activation_type == "sigmoid") {
        a.sigmoid();
    }
    else if (activation_type == "softmax") {
        a.softmax();
    }
    else if (activation_type == "relu") {
        a.relu();
    }
    return a;
}

Matrix Layer::backward(Matrix &dA, double learning_rate) {
    Matrix dZ = dA;
    if (activation_type == "sigmoid") {
        Matrix sigmoid_der = z;
        sigmoid_der.sigmoid_derivative();
        dZ = dZ.elementwise_multiply(sigmoid_der); // dZ = dA * (sigmoid(z))'
    }

    else if (activation_type == "relu") {
        Matrix relu_der = z;
        relu_der.relu_derivative();
        dZ = dZ.elementwise_multiply(relu_der); // dZ = dA * relu'(z)
    }

    // for softmax dZ = dA

    Matrix A_prev_T = input.transpose();
    Matrix dW = Matrix::multiply(dZ, A_prev_T);
    Matrix db = dZ.sum_columns();

    int batch_size = dZ.cols;
    dW.multiply_scalar(1.0 / batch_size);
    db.multiply_scalar(1.0 / batch_size);

    dW.multiply_scalar(learning_rate);
    db.multiply_scalar(learning_rate);

    weights.subtract_inplace(dW);
    bias.subtract_inplace(db);

    Matrix W_T = weights.transpose();
    Matrix dA_prev = Matrix::multiply(W_T, dZ);
    return dA_prev;
}