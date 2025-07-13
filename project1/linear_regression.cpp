#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class LinearRegression {
private:
    double slope;
    double intercept;
    bool trained;

public:
    LinearRegression() : slope(0), intercept(0), trained(false) {}

    void fit(const std::vector<double>& X, const std::vector<double>& y) {
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have the same size");
        }
        if (X.empty()) {
            throw std::invalid_argument("Input arrays cannot be empty");
        }

        size_t n = X.size();
        double sum_x = std::accumulate(X.begin(), X.end(), 0.0);
        double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
        double sum_xy = 0.0;
        double sum_xx = 0.0;

        for (size_t i = 0; i < n; ++i) {
            sum_xy += X[i] * y[i];
            sum_xx += X[i] * X[i];
        }

        double x_mean = sum_x / n;
        double y_mean = sum_y / n;

        slope = (sum_xy - n * x_mean * y_mean) / (sum_xx - n * x_mean * x_mean);
        intercept = y_mean - slope * x_mean;
        trained = true;
    }

    double predict(double x) const {
        if (!trained) {
            throw std::runtime_error("Model must be trained before prediction");
        }
        return slope * x + intercept;
    }

    std::vector<double> predict_vector(const std::vector<double>& X) const {
        if (!trained) {
            throw std::runtime_error("Model must be trained before prediction");
        }
        std::vector<double> predictions;
        predictions.reserve(X.size());
        for (const auto& x : X) {
            predictions.push_back(predict(x));
        }
        return predictions;
    }

    double get_slope() const { return slope; }
    double get_intercept() const { return intercept; }
    bool is_trained() const { return trained; }
};

PYBIND11_MODULE(linear_regression, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("predict", &LinearRegression::predict)
        .def("predict_vector", &LinearRegression::predict_vector)
        .def("get_slope", &LinearRegression::get_slope)
        .def("get_intercept", &LinearRegression::get_intercept)
        .def("is_trained", &LinearRegression::is_trained);
}