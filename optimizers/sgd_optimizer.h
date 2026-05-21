#pragma once

#include "optimizer_base.h"
#include <vector>
#include <cmath>

namespace nn {

// Stochastic Gradient Descent optimizer
class SGDOptimizer : public Optimizer {
private:
    double momentum;
    std::vector<std::vector<double>> velocity;
    bool initialized;
    
public:
    SGDOptimizer(double lr = 0.01, double mom = 0.9) 
        : Optimizer(lr), momentum(mom), initialized(false) {}
    
    void step(std::vector<std::vector<double>>& weights,
              std::vector<std::vector<double>>& gradients) override {
        // Re-initialize if dimensions don't match
        if (!initialized || velocity.size() != weights.size() || 
            (weights.size() > 0 && velocity[0].size() != weights[0].size())) {
            velocity.resize(weights.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                velocity[i].resize(weights[i].size(), 0.0);
            }
            initialized = true;
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                velocity[i][j] = momentum * velocity[i][j] - learning_rate * gradients[i][j];
                weights[i][j] += velocity[i][j];
            }
        }
    }
};

} // namespace nn
