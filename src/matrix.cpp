#include "matrix.hpp"
#include <cmath>
#include <random>
#include <algorithm>
using namespace std;

// defining the constructor
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data.resize(rows, vector<double> (cols, 0.0));
}

// display the matrix
void Matrix::print() const {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            cout << data[i][j] << " ";
        }
        cout << endl;
    }
}

// matrix multiplication
Matrix Matrix::multiply(const Matrix &a, const Matrix &b) {
    if (a.cols != b.rows) {
        throw invalid_argument("Matrix dimensions did not mathch for multiplication.");
    }

    Matrix result(a.rows, b.cols);

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            for (int k = 0; k < a.cols; ++k) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

void Matrix::sigmoid() {
    for(int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            data[i][j] = 1.0 / (1.0 + exp(-data[i][j]));
        }
    }
}

void Matrix::relu() {
    for(int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            if(data[i][j] < 0) {
                data[i][j] = 0;
            }
        }
    }
}

// broadcasting addition
void Matrix::add(const Matrix &b) {
    if (b.cols != 1 || b.rows != rows) {
        throw invalid_argument("Incompatible dimensions for broadcasting addition");
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] += b.data[i][0];
        }
    }
}

Matrix Matrix::subtract(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

void Matrix::random_init(double lower, double upper) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<> dist(lower, upper);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

void Matrix::xavier_init(int input_size) {
    random_device rd;
    mt19937_64 gen(rd());
    double limit = 1.0 / sqrt(input_size);
    uniform_real_distribution<> dist(-limit, limit);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

void Matrix::softmax() {
    for (int j=0; j<cols; j++){
        // find max value in entire col
        double max_value = data[0][j];
        for (int i=1; i<rows; i++) {
            max_value = max(max_value, data[i][j]);
        }

        //compute denomitator
        double sum = 0.0;
        for (int i=0; i<rows; i++) {
            sum += exp(data[i][j] - max_value);
        }

        for (int i=0; i<rows; i++) {
            data[i][j] = exp(data[i][j] - max_value) / sum;
        }
    }
}

void Matrix::sigmoid_derivative() {
    this->sigmoid();
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            data[i][j] = data[i][j] * (1.0 - data[i][j]);
        }
    }
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);  // Notice: flipped dimensions
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::sum_columns() const {
    Matrix result(rows, 1);
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += data[i][j];
        }
        result.data[i][0] = sum;
    }
    return result;
}

void Matrix::multiply_scalar(double scalar) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] *= scalar;
        }
    }
}

void Matrix::subtract_inplace(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for in-place subtraction.");
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] -= other.data[i][j];
        }
    }
}

void Matrix::shape() {
    cout << "(" << rows << ", " << cols << ")" << endl;
}

Matrix Matrix::elementwise_multiply(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication.");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = this->data[i][j] * other.data[i][j];
        }
    }
    return result;
}

int Matrix::argmax_col(int col) const{
    double max_val = data[0][col];
    int max_idx = 0;
    for (int i = 1; i < rows; ++i) {
        if (data[i][col] > max_val) {
            max_val = data[i][col];
            max_idx = i;
        }
    }
    return max_idx;
}

void Matrix::relu_derivative() {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            data[i][j] = (data[i][j] > 0) ? 1.0 : 0.0;
        }
    }
}

Matrix Matrix::get_cols_by_indices(const std::vector<int>& indices, int start, int count) const {
    Matrix result(this->rows, count);
    for (int i = 0; i < count; ++i) {
        int col_idx = indices[start + i];
        for (int row = 0; row < this->rows; ++row) {
            result.data[row][i] = this->data[row][col_idx];
        }
    }
    return result;
}
