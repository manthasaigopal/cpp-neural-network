#pragma once
#include <iostream>
#include <vector>

class Matrix{
    public:
    int rows, cols;
    std::vector<std::vector<double>> data;

    Matrix(int rows=0, int cols=0);

    void print() const;

    void shape();

    void add(const Matrix &b);

    Matrix subtract(const Matrix &other) const;

    void subtract_inplace(const Matrix &other);

    static Matrix multiply(const Matrix &a, const Matrix &b);

    Matrix elementwise_multiply(const Matrix &other) const;

    void multiply_scalar(double scalar);

    Matrix transpose() const;

    Matrix sum_columns() const;

    void random_init(double lower, double upper);
    
    void xavier_init(int input_size);

    void softmax();

    void sigmoid();

    void relu();

    void sigmoid_derivative();

    void relu_derivative();

    int argmax_col(int col) const;

    Matrix get_cols_by_indices(const std::vector<int>& indices, int start, int count) const;
};