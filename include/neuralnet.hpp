#pragma once
#include "layer.hpp"
#include <vector>
#include <numeric>
#include <random>
#include<algorithm>
#include <cmath>

class NeuralNet {
    public:
    std::vector<Layer> layers;

    void add_layer(const Layer &l);

    Matrix forward(const Matrix &input);

    void backward(const Matrix &predicted, const Matrix &actual, double learning_rate); 

    void train(Matrix &X, Matrix &Y, Matrix &val_X, Matrix &val_Y, int epochs, double learning_rate, int batch_size);

    std::vector<std::pair<int, double>> predict(const Matrix& input);

    void save_model(const std::string &filename);

    void loadModel(const std::string& filename);
};