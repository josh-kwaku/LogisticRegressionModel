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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXi;
using Eigen::RowVectorXd;
using Eigen::Block;

struct CostFuncParams {
    MatrixXd train_data;
    VectorXd y_values;
    VectorXd weights;
    double bias;

    CostFuncParams(const MatrixXd &train_data, const VectorXd &y_values, const VectorXd &weights,
                   const double bias): train_data{train_data},
                                       y_values{y_values}, weights{weights}, bias{bias} {
    }
};

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

struct ComputeGradientParams {
    MatrixXd train_data;
    VectorXd y_values;
    VectorXd weights;
    double bias;

    ComputeGradientParams(const MatrixXd &train_data, const VectorXd &y_values, const VectorXd &weights,
                          const double bias): train_data{
                                                  train_data
                                              }, y_values{y_values}, weights{weights}, bias{bias} {
    }
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

struct GradientDescentParams {
    MatrixXd train_data;
    VectorXd y_values;
    VectorXd weights;
    double bias;
    double learning_rate;
    int iterations;

    GradientDescentParams(MatrixXd& train_data, VectorXd& y_values, VectorXd& weights, double bias, double learning_rate,
                          int iterations) : train_data{train_data}, y_values{y_values}, weights{weights},
                                            bias{bias}, learning_rate{learning_rate}, iterations{iterations} {
    }
};

struct GradientDescentResult {
    RowVectorXd cost_values;
};

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

void show_learning_curve(const GradientDescentResult& result, const int iterations) {
     using namespace matplot;
     auto x_values = RowVectorXi::LinSpaced(iterations, 0, iterations - 1);
     std::vector<int> iters(x_values.begin(), x_values.end());
     std::vector<double> costs(result.cost_values.begin(), result.cost_values.end());
     plot(iters, costs, "-o");
     show();
 }

double stddev(const MatrixXd& mat) {
     const double variance = (mat.array() - mat.mean()).square().sum() / mat.size();
     return sqrt(variance);
}

struct NormalizationScalingValues {
     double mean;
     double stddev;
 };
struct NormalizeResult {
    std::vector<NormalizationScalingValues> scalings;
    MatrixXd normalized_values;
};

/**
 * Z-score normalization
 * @param mat
 * @return NormalizeResult
 */
NormalizeResult normalize(MatrixXd &mat) {
    const int m = mat.cols();
    std::vector<NormalizationScalingValues> scaling_values;
    MatrixXd result(mat.rows(), m);
    for (int i = 0; i < m; i++) {
        auto col = mat.col(i);
        double mean = col.mean();
        double standard_deviation = stddev(col);
        result.col(i) = (col.array() - mean) / standard_deviation;
        scaling_values.push_back({mean, standard_deviation});
    }
    return {scaling_values, result};
}

/**
 * Normalize single input data using the mean and std dev for each feature computed from the training data
 * @param vec
 * @param scaling_values
 * @return VectorXd
 */
VectorXd normalize_with_scaling_values(const VectorXd& vec, const std::vector<NormalizationScalingValues>& scaling_values) {
    int m = vec.size();
    VectorXd result(m);
    for (int i = 0; i < m; i++) {
        result(i) = (vec(i) - scaling_values[i].mean) / scaling_values[i].stddev;
    }
    return result;
}

struct PredictParams {
     VectorXd x_values;
     VectorXd weights;
     double bias;
 };
double predict(const PredictParams& params) {
     return params.x_values.dot(params.weights) + params.bias;
 }




