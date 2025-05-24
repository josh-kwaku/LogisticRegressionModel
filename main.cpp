#include <iostream>
#include "linear_regression.cpp"

int main() {
    MatrixXd train_data {
        {2104.0, 5.0, 1.0, 45.0},
            {1416.0, 3.0, 2.0, 40.0},
            {852.0, 2.0, 1.0, 35.0}
    };
    VectorXd y_values(3);
    y_values << 460.0, 232.0, 178.0;
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
    int iterations = 100;
    auto normalization_result = normalize(train_data);
    auto data = GradientDescentParams(normalization_result.normalized_values, y_values, weights, bias, 0.1, iterations);
    auto gd_result = gradient_descent(data);

    // show_learning_curve(gd_result, iterations);

    // Predict
    VectorXd x_values(4);
    x_values << 2104.0, 5.0, 1.0, 45.0;
    auto x_values_normalized = normalize_with_scaling_values(x_values, normalization_result.scalings);
    PredictParams ppParams = {
        x_values_normalized,
        data.weights,
        data.bias,
    };
    double prediction = predict(ppParams);
    std::cout << "prediction = " << prediction << '\n';
}
