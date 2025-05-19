// predit
// fit
// cost fun
// show learning curve
// gradient descent
// gradient
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct CostFuncParams {
    MatrixXd train_data;
    VectorXd y_values;
    VectorXd weights;
    double bias;
};

double cost(CostFuncParams params) {
    const int m = params.train_data.rows();
    double sum = 0.0;
    const auto dot_product = params.train_data * params.weights;
    for (int i = 0; i < m; i++) {
        const double diff = std::pow(((dot_product(i) + params.bias) - params.y_values(i)), 2);
        sum += diff;
    }
    return sum / (2 * m);
}

struct ComputeGradientParams {
    MatrixXd train_data;
    VectorXd y_values;
    VectorXd weights;
    double bias;
};

struct GradientResult {
    VectorXd weights;
    double bias;

    GradientResult(VectorXd weights, double bias) : weights{weights}, bias{bias} {}
};

std::ostream& operator<<(std::ostream &os, const GradientResult &result) {
    std::cout << "Weights: " << result.weights << std::endl;
    std::cout << "Bias: " << result.bias << std::endl;
    return os;
}

GradientResult compute_gradient(ComputeGradientParams params) {
    const auto dot_product = params.train_data * params.weights;
    const int m = params.train_data.rows();
    VectorXd dj_dw(params.weights.rows());
    dj_dw.setZero();
    double updated_bias = 0.0;
    for (int i = 0; i < m; i++) {
        const double err_i = (dot_product(i) +  params.bias) - params.y_values(i);
        for (int j = 0; j < dj_dw.size(); j++) {
            dj_dw(j) += err_i * params.train_data(i, j);
        }
        updated_bias += err_i;
    }
    dj_dw /= m;
    updated_bias /= m;
    return GradientResult(dj_dw, updated_bias);
}