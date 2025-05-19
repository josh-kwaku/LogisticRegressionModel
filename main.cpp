#include <iostream>
#include "linear_regression.cpp"

int main() {
    MatrixXd train_data = MatrixXd::Random(2,3);
    VectorXd y_values = VectorXd::Random(2);
    VectorXd weights = VectorXd::Random(3);
    double bias = 0.123;

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

    std::cout << cost(cfParams) << '\n';
    GradientResult result = compute_gradient(cgParams);
    std::cout << result << '\n';
}

// training data - set of examples
// fit a model to that data so that it can generalize
    // by that we mean getting values for parameters w and b to fit the best curve to the data
    // to get values for parameters w and b, we need gradient descent
    // how accurate w and b are depends 
        // on how much we can minimize the cost function J
        // the learning rate alpha
    //