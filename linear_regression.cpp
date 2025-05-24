// predit
// fit
// cost fun
// show learning curve
// gradient descent
// gradient
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <matplot/matplot.h>
#include <vector>
#include "include/common.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXi;
using Eigen::RowVectorXd;
using Eigen::Block;

/**
 * Cost function computation
 * @param params
 * @return double
 */
double cost(CostFuncParams& params) {
    const int m = params.train_data.rows();
    double sum = 0.0;
    const auto dot_product = params.train_data * params.weights;
    for (int i = 0; i < m; i++) {
        const double diff = std::pow(((dot_product(i) + params.bias) - params.y_values(i)), 2);
        sum += diff;
    }
    return sum / (2 * m);
}

/**
 * Computes the derivative of w and b given the training data, weights, y_values and bias
 * @param params
 * @return GradientResult
 */
GradientResult compute_gradient(const ComputeGradientParams& params) {
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
    return {dj_dw, updated_bias};
}

/**
 * Modifies the input weights and bias values
 * @param params
 * @return GradientDescentResult
 */
GradientDescentResult gradient_descent(GradientDescentParams& params) {
    VectorXd new_weights(params.weights.rows());
    new_weights.setZero();
    double new_bias = 0.0;
    RowVectorXd cost_values(params.iterations);
    cost_values.setZero();
    for (int i = 0; i < params.iterations; i++) {
        ComputeGradientParams compute_gradient_params(params.train_data, params.y_values, params.weights, params.bias);
        CostFuncParams cost_func_params(params.train_data, params.y_values, params.weights, params.bias);
        cost_values(i) = cost(cost_func_params);
        auto result = compute_gradient(compute_gradient_params);
        new_weights = params.weights - (params.learning_rate * result.weights);
        new_bias = params.bias - (params.learning_rate * result.bias);
        params.weights = new_weights;
        params.bias = new_bias;
    }
     return {cost_values};
}




