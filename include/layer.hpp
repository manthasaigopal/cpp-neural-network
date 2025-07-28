#pragma once
#include "matrix.hpp"
#include <string>

class Layer{
    public:
    Matrix weights, bias;
    Matrix input, z, a; // will be used in backprop
    std::string activation_type;

    Layer(int input_size, int output_size, std::string activation_type="sigmoid");

    Layer(const Matrix &w, const Matrix &b, const std::string &activation_type);

    Matrix forward(const Matrix &input);
    Matrix backward(Matrix &dA, double learning_rate);
};
