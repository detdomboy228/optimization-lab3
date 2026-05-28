#pragma once

#include "optimizer_base.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

namespace nn {

// Adam optimizer (Adaptive Moment Estimation)
class AdamOptimizer : public Optimizer {
private:
    double beta1;      // First moment decay rate
    double beta2;      // Second moment decay rate
    double epsilon;    // Small constant for numerical stability
    std::map<const void*, int> t_by_matrix;  // Time step
    std::map<const void*, std::vector<std::vector<double>>> m_by_matrix;  // First moment
    std::map<const void*, std::vector<std::vector<double>>> v_by_matrix;  // Second moment
    std::map<const void*, int> t_by_vector;
    std::map<const void*, std::vector<double>> m_by_vector;
    std::map<const void*, std::vector<double>> v_by_vector;
    
public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {}
    
    void step(std::vector<std::vector<double>>& weights,
              std::vector<std::vector<double>>& gradients) override {
        const void* key = static_cast<const void*>(&weights);
        auto& m = m_by_matrix[key];
        auto& v = v_by_matrix[key];
        auto& t = t_by_matrix[key];

        if (m.size() != weights.size() ||
            (weights.size() > 0 && m[0].size() != weights[0].size())) {
            m.resize(weights.size());
            v.resize(weights.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                m[i].resize(weights[i].size(), 0.0);
                v[i].resize(weights[i].size(), 0.0);
            }
            t = 0;  // Reset time step on re-initialization
        }
        
        t++;
        
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                double g = gradients[i][j];
                
                // Update biased first moment estimate
                m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;
                
                // Update biased second raw moment estimate
                v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g;
                
                // Compute bias-corrected first moment estimate
                double m_hat = m[i][j] / bias_correction1;
                
                // Compute bias-corrected second raw moment estimate
                double v_hat = v[i][j] / bias_correction2;
                
                // Update weights
                weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }

    void step(std::vector<double>& values,
              std::vector<double>& gradients) override {
        const void* key = static_cast<const void*>(&values);
        auto& m = m_by_vector[key];
        auto& v = v_by_vector[key];
        auto& t = t_by_vector[key];

        if (m.size() != values.size()) {
            m.assign(values.size(), 0.0);
            v.assign(values.size(), 0.0);
            t = 0;
        }

        t++;

        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);

        for (size_t i = 0; i < values.size(); ++i) {
            double g = gradients[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            double m_hat = m[i] / bias_correction1;
            double v_hat = v[i] / bias_correction2;

            values[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};

} // namespace nn
