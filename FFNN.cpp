//
// Created by young on 2025-05-08.
//

#include "FFNN.h"
#include <iostream>
#include <random>

FFNN::FFNN(int input_size, int hidden_size, int output_size)
    : I_input_size(input_size), J_hidden_size(hidden_size), K_output_size(output_size) {
    init_weights();
}

int FFNN::get_size_include_bias(const int size) {
    return size + 1; // include bias
}

std::vector<double> FFNN::forward(std::vector<double> x) {
    int I_input_size_bias = get_size_include_bias(I_input_size); //consider bias input size of 1
    int J_hidden_size_bias = get_size_include_bias(J_hidden_size); //consider bias hidden size of 1

    double bias = 1.0;

    // input layer
    x.push_back(bias);

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

    // add bias
    z_hidden.push_back(bias);

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
    const int I_input_size_bias = get_size_include_bias(I_input_size);
    const int J_hidden_size_bias = get_size_include_bias(J_hidden_size);

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

void FFNN::fit(const std::vector<std::vector<double>> X, const std::vector<std::vector<double>> T, int epochs, double learning_rate) {
    int I_input_size_bias = get_size_include_bias(I_input_size); //consider bias input size of 1
    int J_hidden_size_bias = get_size_include_bias(J_hidden_size); //consider bias hidden size of 1

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        double accuracy = 0.0;

        // initialize gradients
        std::vector<std::vector<double>> dW(J_hidden_size_bias, std::vector<double>(I_input_size_bias));
        std::vector<std::vector<double>> dV(K_output_size, std::vector<double>(J_hidden_size_bias));

        // loop over each training example
        for (int n = 0; n < X.size(); ++n) {
            // Forward pass
            std::vector<double> y = forward(X[n]);
            std::vector<double> t = T[n];
            loss += cross_entropy(y, t);
            accuracy += (y == t) ? 1.0 : 0.0;

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
        std::cout << "Epoch: " << epoch << " Loss: " << loss / N << " Accuracy: " << accuracy / N << std::endl;
    }
}

void FFNN::predict(const std::vector<double> X, std::vector<double>& Y) {
    int I_input_size_bias = get_size_include_bias(I_input_size); //consider bias input size of 1
    int J_hidden_size_bias = get_size_include_bias(J_hidden_size); //consider bias hidden size of 1
    double bias = 1.0;

    // input layer
    std::vector<double> x = X;
    // add bias
    x.push_back(bias);

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
    // add bias
    z_hidden.push_back(bias);

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

void FFNN::predict(const std::vector<std::vector<double>> X, const std::vector<std::vector<double>>& T, std::vector<std::vector<double>>& Y) {
    const int N = X.size();
    Y.resize(N, std::vector<double>(K_output_size, 0.0));
    double accuracy = 0.0;
    double loss = 0.0;
    for (int n = 0; n < N; ++n) {
        predict(X[n], Y[n]);
        std::vector<double> y = Y[n];
        std::vector<double> t = T[n];
        loss += cross_entropy(y, t);
        accuracy += (y == t) ? 1.0 : 0.0;
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

    for (int j = 0; j < J_hidden_size; j++)
    {
        std::vector<double> row;
        for (int i = 0; i < I_input_size + 1; i++)
        {
            row.push_back(dist(gen));
        }
        weight_w.push_back(row);
    }
    W = weight_w;
    for (int k = 0; k < K_output_size; k++)
    {
        std::vector<double> row;
        for (int j = 0; j < J_hidden_size + 1; j++)
        {
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