#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "matrix.hpp"

void load_csv(const std::string &filename, Matrix &X, Matrix &Y, int limit = -1);

double compute_accuracy(const Matrix &predictions, const Matrix &targets);
