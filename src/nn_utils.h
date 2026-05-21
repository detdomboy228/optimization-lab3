#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace nn {

// Sigmoid activation function and its derivative
inline double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + std::exp(-x));
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

inline double sigmoid_derivative(double s) {
    // s is already sigmoid(x), so derivative is s * (1 - s)
    return s * (1.0 - s);
}

// ReLU activation function and its derivative
inline double relu(double x) {
    return std::max(0.0, x);
}

inline double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Matrix operations
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    size_t m = A.size();
    size_t n = A[0].size();
    size_t p = B[0].size();
    
    Matrix C(m, Vector(p, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < p; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    size_t m = A.size();
    size_t n = A[0].size();
    Matrix T(n, Vector(m));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix add_bias(const Matrix& X, const Vector& b) {
    Matrix result = X;
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[0].size(); ++j) {
            result[i][j] += b[j];
        }
    }
    return result;
}

Vector matrix_vector_multiply(const Matrix& A, const Vector& x) {
    Vector result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Apply activation element-wise to a matrix
Matrix apply_activation(Matrix& X, double (*activation)(double)) {
    for (auto& row : X) {
        for (auto& val : row) {
            val = activation(val);
        }
    }
    return X;
}

// Cross-entropy loss for binary classification
double binary_cross_entropy(const Vector& y_true, const Vector& y_pred) {
    double eps = 1e-15;
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double yp = std::max(eps, std::min(1.0 - eps, y_pred[i]));
        loss -= y_true[i] * std::log(yp) + (1.0 - y_true[i]) * std::log(1.0 - yp);
    }
    return loss / y_true.size();
}

} // namespace nn
