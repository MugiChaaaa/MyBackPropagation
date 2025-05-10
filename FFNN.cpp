//
// Created by young on 2025-05-08.
//

#include "FFNN.h"
#include <iostream>
#include <random>

FFNN::FFNN(int input_size, int hidden_size, int output_size)
    : I_input_size(input_size), J_hidden_size(hidden_size), K_output_size(output_size),
      I_input_size_bias(add_size_bias(input_size)), J_hidden_size_bias(add_size_bias(hidden_size)) {
    init_weights();
}

std::vector<double> FFNN::forward(std::vector<double> x) {

    // hidden layer
    for (int j = 0; j < J_hidden_size; ++j) {
        b_hidden[j] = 0.0;
        for (int i = 0; i < I_input_size_bias; ++i) {
            b_hidden[j] += W[j][i] * x[i];
        }
    }

    // activate function
    for (int j = 0; j < J_hidden_size; ++j) {
        z_hidden[j] = sigmoid(b_hidden[j]);
    }

    // output layer
    std::vector<double> a_output(K_output_size);
    for (int k = 0; k < K_output_size; ++k) {
        a_output[k] = 0.0;
        for (int j = 0; j < J_hidden_size_bias; ++j) {
            a_output[k] += V[k][j] * z_hidden[j];
        }
    }

    // activate function
    std::vector<double> y_output(K_output_size);
    for (int k = 0; k < K_output_size; ++k) {
        y_output[k] = softmax(a_output[k], a_output);
    }

    // return b_hidden for calculation C_1 at backward.
    return y_output;
}

void FFNN::backward(std::vector<double> x, std::vector<double> y_output, const std::vector<double> t, std::vector<std::vector<double>>& dW, std::vector<std::vector<double>>& dV) {

    // output layer
    std::vector<double> C_2(K_output_size);
    for (int k = 0; k < K_output_size; ++k) {
        C_2[k] = y_output[k] - t[k];
    }

    // hidden layer
    std::vector<double> C_1(J_hidden_size);
    for (int j = 0; j < J_hidden_size; ++j) {
        C_1[j] = 0.0;
        for (int k = 0; k < K_output_size; ++k) {
            C_1[j] += V[k][j] * C_2[k];
        }
        C_1[j] *= sigmoid_der(b_hidden[j]);
    }

    // calculate gradients
    // output layer
    for (int k = 0; k < K_output_size; ++k) {
        for (int j = 0; j < J_hidden_size_bias; ++j) {
            dV[k][j] += C_2[k] * z_hidden[j];
        }
    }

    // hidden layer
    for (int j = 0; j < J_hidden_size; ++j) {
        for (int i = 0; i < I_input_size_bias; ++i) {
            dW[j][i] += C_1[j] * x[i];
        }
    }
}

void FFNN::fit(std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& T, int epochs, double learning_rate) {
    // add bias
    const double bias = 1.0;
    for (int n = 0; n < X.size(); ++n) {
        X[n].push_back(bias);
    }
    z_hidden.push_back(bias);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        double accuracy = 0.0;

        // initialize gradients
        std::vector<std::vector<double>> dW(J_hidden_size_bias, std::vector<double>(I_input_size_bias));
        std::vector<std::vector<double>> dV(K_output_size, std::vector<double>(J_hidden_size_bias));



        // loop over each training sample
        for (int n = 0; n < X.size(); ++n) {
            // Forward pass
            std::vector<double> y = forward(X[n]);
            std::vector<double> t = T[n];
            loss += cross_entropy(y, t);
            int y_pred = std::max_element(y.begin(), y.end()) - y.begin();
            int t_true = std::max_element(t.begin(), t.end()) - t.begin();
            accuracy += (y_pred == t_true) ? 1.0 : 0.0;

            // Backward pass
            backward(X[n], y, t, dW, dV);
        }
        // Update weights and biases
        // input data size
        const int N = X.size();

        // output layer
        for (int k = 0; k < K_output_size; ++k) {
            for (int j = 0; j < J_hidden_size_bias; ++j) {
                V[k][j] -= learning_rate * dV[k][j] / N;
            }
        }

        // hidden layer
        for (int j = 0; j < J_hidden_size; ++j) {
            for (int i = 0; i < I_input_size_bias; ++i) {
                W[j][i] -= learning_rate * dW[j][i] / N;
            }
        }

        // Calculate average loss and accuracy
        std::cout << "Epoch: " << epoch + 1 << " Loss: " << loss / N << " Accuracy: " << accuracy / N << std::endl;
    }
}

void FFNN::predict(const std::vector<double>& X, std::vector<double>& Y) {

    // hidden layer
    for (int j = 0; j < J_hidden_size; ++j) {
        b_hidden[j] = 0.0;
        for (int i = 0; i < I_input_size_bias; ++i) {
            b_hidden[j] += W[j][i] * X[i];
        }
    }
    // activate function
    for (int j = 0; j < J_hidden_size; ++j) {
        z_hidden[j] = sigmoid(b_hidden[j]);
    }

    // output layer
    std::vector<double> a_output(K_output_size);
    for (int k = 0; k < K_output_size; ++k) {
        a_output[k] = 0.0;
        for (int j = 0; j < J_hidden_size_bias; ++j) {
            a_output[k] += V[k][j] * z_hidden[j];
        }
    }
    // activate function
    std::vector<double> y_output(K_output_size);
    for (int k = 0; k < K_output_size; ++k) {
        y_output[k] = softmax(a_output[k], a_output);
    }
    Y = y_output;
}

void FFNN::predict(std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& T, std::vector<std::vector<double>>& Y) {
    const int N = X.size();
    Y.resize(N, std::vector<double>(K_output_size, 0.0));
    double accuracy = 0.0;
    double loss = 0.0;

    // add bias to test input data
    const double bias = 1.0;
    for (int n = 0; n < N; ++n) {
        X[n].push_back(bias);
    }

    for (int n = 0; n < N; ++n) {
        predict(X[n], Y[n]);
        std::vector<double> y = Y[n];
        std::vector<double> t = T[n];
        loss += cross_entropy(y, t);
        int y_pred = std::max_element(y.begin(), y.end()) - y.begin();
        int t_true = std::max_element(t.begin(), t.end()) - t.begin();
        accuracy += (y_pred == t_true) ? 1.0 : 0.0;
    }
    // Calculate average loss and accuracy
    std::cout << "Loss: " << loss / N << " Accuracy: " << accuracy / N << std::endl;
}

void FFNN::init_weights() {
    // Initialize weights and biases with random values between -1.0 and 1.0
    // Also initialize b_hidden and z_hidden for the calculation of C_1

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> weight_w;
    std::vector<std::vector<double>> weight_v;

    for (int j = 0; j < J_hidden_size; j++) {
        std::vector<double> row;
        for (int i = 0; i < I_input_size + 1; i++) {
            row.push_back(dist(gen));
        }
        weight_w.push_back(row);
    }
    W = weight_w;
    for (int k = 0; k < K_output_size; k++) {
        std::vector<double> row;
        for (int j = 0; j < J_hidden_size + 1; j++) {
            row.push_back(dist(gen));
        }
        weight_v.push_back(row);
    }
    V = weight_v;

    // Initialize b_hidden and z_hidden
    b_hidden.resize(J_hidden_size, 0.0);
    z_hidden.resize(J_hidden_size, 0.0);
    for (int j = 0; j < J_hidden_size; ++j) {
        b_hidden[j] = 0.0;
        z_hidden[j] = 0.0;
    }
}

double FFNN::sigmoid(const double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double FFNN::sigmoid_der(const double x) {
    const double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double FFNN::softmax(const double x, const std::vector<double>& X) {
    double softmax = 0.0;
    double sum = 0.0;
    for (int i = 0; i < X.size(); i++) {
        sum += std::exp(X[i]);
    }
    softmax = std::exp(x) / sum;
    return softmax;
}

double FFNN::cross_entropy(const std::vector<double>& Y, const std::vector<double>& T) {
    double sum = 0.0;
    for (int k = 0; k < Y.size(); k++) {
        sum += T[k] * std::log(Y[k]);
    }
    return -sum;
}