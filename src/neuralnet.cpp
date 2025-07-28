#include "neuralnet.hpp"
#include "loss.hpp"
#include "utils.hpp"

void NeuralNet::add_layer(const Layer &l) {
    layers.push_back(l);
}

Matrix NeuralNet::forward(const Matrix &input) {
    Matrix out = input;
    for (auto &layer : layers) {
        out = layer.forward(out);
    }
    return out;
}

void NeuralNet::backward(const Matrix &predicted, const Matrix &actual, double learning_rate) {
    Matrix dA = predicted.subtract(actual);

    for (int i=layers.size()-1; i>=0; i--) {
        dA = layers[i].backward(dA, learning_rate);
    }
}

void NeuralNet::train(Matrix &X, Matrix &Y, Matrix &val_X, Matrix &val_Y, int epochs, double learning_rate, int batch_size) {
    int total_samples = X.cols;
    std::vector<int> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0); 

    double train_loss = 0.0;
    double train_accuracy = 0.0;
    int num_batches = static_cast<int>(std::ceil(static_cast<double>(total_samples) / batch_size));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        train_loss = 0.0;
        train_accuracy = 0.0;
        for (int start = 0; start < total_samples; start += batch_size) {
            int actual_batch = std::min(batch_size, total_samples - start);

            Matrix batch_X = X.get_cols_by_indices(indices, start, actual_batch); // features × batch
            Matrix batch_Y = Y.get_cols_by_indices(indices, start, actual_batch); // 10 × batch

            Matrix output = this->forward(batch_X);
            train_loss += cross_entropy_loss(output, batch_Y);
            train_accuracy += compute_accuracy(output, batch_Y);

            this->backward(output, batch_Y, learning_rate);
        }

        train_loss = train_loss / static_cast<double>(num_batches);
        train_accuracy = train_accuracy / static_cast<double>(num_batches);

        Matrix val_output = this->forward(val_X);
        double val_loss = cross_entropy_loss(val_output, val_Y);
        double val_accuracy = compute_accuracy(val_output, val_Y);

        std::cout << "Epoch " << (epoch + 1) << " / " << epochs << std::endl;
        std::cout << "-------------------------------" << std::endl;
        std::cout << "Train Loss:      " << train_loss << std::endl;
        std::cout << "Train Accuracy:  " << train_accuracy * 100 << "%" << std::endl;
        std::cout << "Val Loss:        " << val_loss << std::endl;
        std::cout << "Val Accuracy:    " << val_accuracy * 100 << "%" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Training complete." << std::endl;
}

std::vector<std::pair<int, double>> NeuralNet::predict(const Matrix& input) {
    Matrix output = this->forward(input);

    std::vector<std::pair<int, double>> result;

    for (int i = 0; i < output.cols; ++i) {
        int pred_class = output.argmax_col(i);
        double confidence = output.data[pred_class][i]; 
        result.emplace_back(pred_class, confidence);
    }

    return result;
}


void NeuralNet::save_model(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for saving model.\n";
        return;
    }

    for (const auto& layer : layers) {
        outFile << layer.activation_type << "\n";

        outFile << layer.weights.rows << " " << layer.weights.cols << "\n";
        for (int i = 0; i < layer.weights.rows; ++i) {
            for (int j = 0; j < layer.weights.cols; ++j) {
                outFile << layer.weights.data[i][j] << " ";
            }
            outFile << "\n";
        }

        outFile << layer.bias.rows << " " << layer.bias.cols << "\n";
        for (int i = 0; i < layer.bias.rows; ++i) {
            for (int j = 0; j < layer.bias.cols; ++j) {
                outFile << layer.bias.data[i][j] << " ";
            }
            outFile << "\n";
        }
    }
    outFile.close();
}


void NeuralNet::loadModel(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open file for loading model.\n";
        return;
    }

    layers.clear();

    while (!inFile.eof()) {
        std::string activation;
        if (!(inFile >> activation)) break;

        int wRows, wCols;
        inFile >> wRows >> wCols;
        Matrix weights(wRows, wCols);
        for (int i = 0; i < wRows; ++i)
            for (int j = 0; j < wCols; ++j)
                inFile >> weights.data[i][j];

        int bRows, bCols;
        inFile >> bRows >> bCols;
        Matrix biases(bRows, bCols);
        for (int i = 0; i < bRows; ++i)
            for (int j = 0; j < bCols; ++j)
                inFile >> biases.data[i][j];

        Layer layer(weights, biases, activation);
        layers.push_back(layer);
    }

    inFile.close();
}
