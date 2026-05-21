#pragma once

#include "../src/nn_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <functional>

namespace nn {

// Base optimizer class
class Optimizer {
protected:
    double learning_rate;
    
public:
    Optimizer(double lr = 0.01) : learning_rate(lr) {}
    virtual ~Optimizer() = default;
    
    double get_learning_rate() const {
        return learning_rate;
    }
    
    virtual void step(std::vector<std::vector<double>>& weights,
                      std::vector<std::vector<double>>& gradients) = 0;
    
    virtual void set_learning_rate(double lr) {
        learning_rate = lr;
    }
};

} // namespace nn
