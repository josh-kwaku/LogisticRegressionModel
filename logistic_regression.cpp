//
// Created by joshu on 5/24/2025.
//
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "include/common.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
struct SigmoidParams {
    MatrixXd train_data;
    VectorXd weights;
    double bias;

    SigmoidParams(const MatrixXd &train_data, const VectorXd &weights, const double bias): train_data{train_data},
                                                                                           weights{weights}, bias{bias} {
    }
};

VectorXd sigmoid(const SigmoidParams& params) {
    auto dot_product = params.train_data * params.weights;
    int m = dot_product.size();
    VectorXd result(m);
    for (int i = 0; i < m; i++) {
        result(i) = 1.0 / (1.0 + exp(-(dot_product(i) + params.bias)));
    }
    return result;
}

/**
 * Cost function computation
 * @param params
 * @return double
 */
double logistic_cost(CostFuncParams& params) {
    const int m = params.train_data.rows();
    double sum = 0.0;
    const SigmoidParams sigmoid_params(params.train_data, params.weights, params.bias);
    const VectorXd sigmoid_values = sigmoid(sigmoid_params);
    const double epsilon = 1e-10;
    for (int i = 0; i < m; i++) {
        const double first_half = std::log(sigmoid_values(i) + epsilon) * params.y_values(i);
        const double second_half = std::log(1 - sigmoid_values(i) + epsilon) * (1 - params.y_values(i));
        sum += first_half + second_half;
    }
    return (sum * -1) / m;
}

/**
 * Computes the derivative of w and b given the training data, weights, y_values and bias
 * @param params
 * @return GradientResult
 */
GradientResult compute_logistic_gradient(const ComputeGradientParams& params) {
    const int m = params.train_data.rows();
    VectorXd dj_dw(params.weights.rows());
    dj_dw.setZero();
    double updated_bias = 0.0;
    SigmoidParams sigmoid_params = SigmoidParams(params.train_data, params.weights, params.bias);
    const VectorXd sigmoid_values = sigmoid(sigmoid_params);
    for (int i = 0; i < m; i++) {
        const double err_i = sigmoid_values(i) - params.y_values(i);
        for (int j = 0; j < dj_dw.size(); j++) {
            dj_dw(j) += err_i * params.train_data(i, j);
        }
        updated_bias += err_i;
    }
    dj_dw /= m;
    updated_bias /= m;
    return {dj_dw, updated_bias};
}

/**
 * Modifies the input weights and bias values
 * @param params
 * @return GradientDescentResult
 */
GradientDescentResult logistic_gradient_descent(GradientDescentParams& params) {
    VectorXd new_weights(params.weights.rows());
    new_weights.setZero();
    double new_bias = 0.0;
    RowVectorXd cost_values(params.iterations);
    cost_values.setZero();
    for (int i = 0; i < params.iterations; i++) {
        ComputeGradientParams compute_gradient_params(params.train_data, params.y_values, params.weights, params.bias);
        CostFuncParams cost_func_params(params.train_data, params.y_values, params.weights, params.bias);
        cost_values(i) = logistic_cost(cost_func_params);
        auto result = compute_logistic_gradient(compute_gradient_params);
        new_weights = params.weights - (params.learning_rate * result.weights);
        new_bias = params.bias - (params.learning_rate * result.bias);
        params.weights = new_weights;
        params.bias = new_bias;
    }
    return {cost_values};
}
