//
// Created by young on 2025-05-08.
//

#ifndef FFNN_H
#define FFNN_H

#include <vector>

class FFNN {
public:
    FFNN(const int input_size, const int hidden_size, const int output_size);

    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& T, int epochs, double learning_rate);
    void predict(const std::vector<double>& X, std::vector<double>& Y);
    void predict(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& T, std::vector<std::vector<double>>& Y);

    static int argmax(const std::vector<double>& v);

private:
    void init_weights();

    static constexpr int get_size_include_bias(int size) { return size + 1; }

    std::vector<double> forward(std::vector<double> x);

    void backward(std::vector<double> x, std::vector<double> y_output, const std::vector<double> t, std::vector<std::vector<double>>& dW, std::vector<std::vector<double>>& dV);

    double softmax(const double x, const std::vector<double>& X);

    double sigmoid(const double x);

    double sigmoid_der(const double x);

    double cross_entropy(const std::vector<double>& Y, const std::vector<double>& T);

    int I_input_size;
    int J_hidden_size;
    int K_output_size;

    std::vector<std::vector<double>> W;
    std::vector<std::vector<double>> V;

    std::vector<double> b_hidden;
    std::vector<double> z_hidden;
};



#endif //FFNN_H
