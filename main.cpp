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

    // std::cout << cost(cfParams) << '\n';
    int iterations = 5000;
    GradientResult result = compute_gradient(cgParams);
    // std::cout << result << "\n\n\n";
    auto data = GradientDescentParams(train_data, y_values, weights, bias, 0.0001, iterations);
    auto gd_result = gradient_descent(data);
    // std::cout << "updated weights = " << weights << '\n';
    // std::cout << "costs = " << gd_result.cost_values << '\n';
    show_learning_curve(gd_result, iterations);
}

// training data - set of examples
// fit a model to that data so that it can generalize
    // by that we mean getting values for parameters w and b to fit the best curve to the data
    // to get values for parameters w and b, we need gradient descent
    // how accurate w and b are depends 
        // on how much we can minimize the cost function J
        // the learning rate alpha
    //