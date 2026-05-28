#pragma once

#include "optimizer_base.h"
#include <vector>
#include <cmath>
#include <map>

namespace nn {

// Stochastic Gradient Descent optimizer
class SGDOptimizer : public Optimizer {
private:
    double momentum;
    std::map<const void*, std::vector<std::vector<double>>> velocity_by_matrix;
    std::map<const void*, std::vector<double>> velocity_by_vector;
    
public:
    SGDOptimizer(double lr = 0.01, double mom = 0.9) 
        : Optimizer(lr), momentum(mom) {}
    
    void step(std::vector<std::vector<double>>& weights,
              std::vector<std::vector<double>>& gradients) override {
        const void* key = static_cast<const void*>(&weights);
        auto& velocity = velocity_by_matrix[key];

        if (velocity.size() != weights.size() ||
            (weights.size() > 0 && velocity[0].size() != weights[0].size())) {
            velocity.resize(weights.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                velocity[i].resize(weights[i].size(), 0.0);
            }
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                velocity[i][j] = momentum * velocity[i][j] - learning_rate * gradients[i][j];
                weights[i][j] += velocity[i][j];
            }
        }
    }

    void step(std::vector<double>& values,
              std::vector<double>& gradients) override {
        const void* key = static_cast<const void*>(&values);
        auto& velocity = velocity_by_vector[key];

        if (velocity.size() != values.size()) {
            velocity.assign(values.size(), 0.0);
        }

        for (size_t i = 0; i < values.size(); ++i) {
            velocity[i] = momentum * velocity[i] - learning_rate * gradients[i];
            values[i] += velocity[i];
        }
    }
};

} // namespace nn
