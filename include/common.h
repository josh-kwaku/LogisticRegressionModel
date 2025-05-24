//
// Created by joshu on 5/24/2025.
//

#ifndef COMMON_H
#define COMMON_H
#include <Eigen/Dense>
#include <matplot/matplot.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::RowVectorXi;

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

void show_learning_curve(const GradientDescentResult& result, const int iterations) {
    using namespace matplot;
    auto x_values = RowVectorXi::LinSpaced(iterations, 0, iterations - 1);
    std::vector<int> iters(x_values.begin(), x_values.end());
    std::vector<double> costs(result.cost_values.begin(), result.cost_values.end());
    plot(iters, costs, "-o");
    show();
}
#endif //COMMON_H
