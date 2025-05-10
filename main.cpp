#include "FFNN.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <random>

bool load_data(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& T, int& input_size);
bool save_predictions(const std::string& filename, const std::vector<std::vector<double>>& X_test, const std::vector<std::vector<double>>& T_test, const std::vector<std::vector<double>>& Y_pred);

int main() {
    std::cout << "This code represents 1 hidden-layer simple FFNN" << std::endl;

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> T;
    int input_size;
    int output_size = 6;
    const int n_hidden = 12;
    double learning_rate = 0.1;
    int epochs = 1000;

    if (!load_data("../Dataset/Stars_train.csv", X, T, input_size)) {
        std::cerr << "Error loading data" << std::endl;
        return 1;
    }
    std::cout << "File Loaded" << std::endl;

    FFNN ffnn(input_size, n_hidden, output_size);
    ffnn.fit(X, T, epochs, learning_rate);

    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> T_test;
    std::vector<std::vector<double>> Y;
    if (!load_data("../Dataset/Stars_test.csv", X_test, T_test, input_size)) {
        std::cerr << "Error loading data" << std::endl;
        return 1;
    }
    ffnn.predict(X_test, T_test, Y);

    save_predictions("../Result/Stars_pred.csv", X_test, T_test, Y);

    return 0;
}

bool load_data(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& T, int& input_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    bool skip_row = true; // skip the first row
    while (std::getline(file, line)) {
        if (skip_row) {
            skip_row = false;
            continue;
        }
        std::vector<double> row;
        std::vector<double> row_t(6, 0.0);
        std::string value;
        std::stringstream ss(line);
        int idx = 0;
        while (std::getline(ss, value, ',')) {
            if (idx == 5) { // target value

                row_t[std::stoi(value)] = 1.0; // one-hot encoding
                break;
            }
            else if (idx != 0) { // index number
                row.push_back(std::stod(value));
            }
            ++idx;
        }
        X.push_back(row);
        T.push_back(row_t);
    }

    input_size = X[0].size();

    return true;
}

bool save_predictions(const std::string& filename, const std::vector<std::vector<double>>& X_test, const std::vector<std::vector<double>>& T_test, const std::vector<std::vector<double>>& Y_pred) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    file << "X_1,X_2,X_3,X_4,T_1,T_2,T_3,T_4,T_5,T_6,Y_1,Y_2,Y_3,Y_4,Y_5,Y_6,Predicted_Class,True_Class" << std::endl;

    for (int i = 0; i < X_test.size(); i++) {
        for (int j = 0; j < X_test[i].size() - 1; j++) {
            file << X_test[i][j] << ",";
        }
        for (int j = 0; j < T_test[i].size(); j++) {
            file << T_test[i][j] << ",";
        }
        for (int j = 0; j < Y_pred[i].size(); j++) {
            file << Y_pred[i][j] << ",";
        }
        file << std::max_element(Y_pred[i].begin(), Y_pred[i].end()) - Y_pred[i].begin() << ",";
        file << std::max_element(T_test[i].begin(), T_test[i].end()) - T_test[i].begin();
        file << std::endl;
    }

    return true;
}