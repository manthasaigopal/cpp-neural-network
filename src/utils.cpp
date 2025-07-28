#include "utils.hpp"
#include "matrix.hpp"

void load_csv(const std::string& filename, Matrix& X, Matrix& Y, int limit) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> labels;
    int count = 0;

    while (std::getline(file, line)) {
        if (limit > 0 && count >= limit) break;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;

        if (!std::getline(ss, cell, ',')) continue;
        if (cell.empty()) continue;

        int label_val;
        try {
            label_val = std::stoi(cell);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid label found. Skipping line.\n";
            continue;
        }

        std::vector<double> input_vec;
        while (std::getline(ss, cell, ',')) {
            try {
                input_vec.push_back(std::stod(cell) / 255.0); // normalize
            } catch (...) {
                std::cerr << "Invalid pixel found. Skipping line.\n";
                input_vec.clear();
                break;
            }
        }

        if (input_vec.size() != 784) continue; 
        std::vector<double> label_vec(10, 0.0);
        label_vec[label_val] = 1.0;

        inputs.push_back(input_vec);
        labels.push_back(label_vec);
        count++;
    }

    // converting vectors to Matrix format
    int num_samples = inputs.size();
    int num_features = inputs[0].size();

    X = Matrix(num_features, num_samples); // each column is a sample
    Y = Matrix(10, num_samples);           // one-hot labels

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            X.data[j][i] = inputs[i][j];
        }
        for (int k = 0; k < 10; ++k) {
            Y.data[k][i] = labels[i][k];
        }
    }
}

double compute_accuracy(const Matrix &predictions, const Matrix &targets) {
    int correct = 0;
    for (int col = 0; col < predictions.cols; ++col) {
        int predicted_label = predictions.argmax_col(col);
        int true_label = targets.argmax_col(col);
        if (predicted_label == true_label) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predictions.cols;
}
