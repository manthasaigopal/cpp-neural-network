#include "loss.hpp"
#include <cmath>
#include <cassert>

double cross_entropy_loss(const Matrix &predicted, const Matrix &actual) {
    if (predicted.rows != actual.rows || predicted.cols != actual.cols) {
        throw std::runtime_error("Shape mismatch: predicted and actual must have the same dimensions.");
    }

    double loss = 0.0;
    for (int i=0; i<predicted.rows; i++) {
        for (int j=0; j<predicted.cols; j++) {
            loss += -actual.data[i][j] * log(std::max(predicted.data[i][j], 1e-10));
        }
    }
    return loss/predicted.cols;
}