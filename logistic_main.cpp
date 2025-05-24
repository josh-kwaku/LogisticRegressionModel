//
// Created by joshu on 5/24/2025.
//
#include "logistic_regression.cpp"
#include "include/common.h"

int main() {
    Eigen::MatrixXd train_data(10, 4);
    train_data << 50.0, 1.0, 24.0, 45.0,
                  120.0, 5.0, 6.0, 90.0,
                  80.0, 2.0, 12.0, 30.0,
                  30.0, 4.0, 3.0, 25.0,
                  150.0, 0.0, 36.0, 60.0,
                  90.0, 3.0, 6.0, 85.0,
                  60.0, 1.0, 18.0, 50.0,
                  130.0, 6.0, 3.0, 100.0,
                  70.0, 0.0, 36.0, 40.0,
                  40.0, 4.0, 6.0, 65.0;

    Eigen::VectorXd y_values(10);
    y_values << 0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1;

    VectorXd weights(4);
    weights.setZero();
    double bias = 0;

    CostFuncParams cfParams = {
        train_data,
        y_values,
        weights,
        bias
    };

    ComputeGradientParams cgParams = {
        train_data,
        y_values,
        weights,
        bias
    };

    // Train
    int iterations = 2500;
    auto normalization_result = normalize(train_data);
    auto data = GradientDescentParams(normalization_result.normalized_values, y_values, weights, bias, 0.1, iterations);
    auto gd_result = logistic_gradient_descent(data);

    // show_learning_curve(gd_result, iterations);


    // Predict
    VectorXd x_values(4);
    x_values << 40.0, 4.0, 6.0, 65.0;
    auto x_values_normalized = normalize_with_scaling_values(x_values, normalization_result.scalings);
    PredictParams ppParams = {
        x_values_normalized,
        data.weights,
        data.bias,
    };
    double prediction = predict(ppParams);
    int ans = prediction <= 0.5 ? 0 : 1;
    std::cout << "prediction = " << ans << '\n';
}