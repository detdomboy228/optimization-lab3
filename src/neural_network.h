#pragma once

#include "nn_utils.h"
#include "../optimizers/optimizer_base.h"
#include "../optimizers/sgd_optimizer.h"
#include "../optimizers/adam_optimizer.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iomanip>

namespace nn {

class NeuralNetwork {
private:
    std::vector<size_t> architecture;  // Layer sizes including input and output
    std::vector<std::vector<std::vector<double>>> weights;  // W[l][i][j]
    std::vector<std::vector<double>> biases;  // b[l][j]
    
    // Cache for forward pass
    std::vector<std::vector<std::vector<double>>> activations;  // A[l][batch][j]
    std::vector<std::vector<std::vector<double>>> z_values;     // Z[l][batch][j]
    
    std::mt19937 rng;
    
public:
    NeuralNetwork(const std::vector<size_t>& layers, unsigned int seed = 42)
        : architecture(layers), rng(seed) {
        initialize_weights();
    }
    
    void initialize_weights() {
        weights.clear();
        biases.clear();
        
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t l = 0; l < architecture.size() - 1; ++l) {
            size_t in_size = architecture[l];
            size_t out_size = architecture[l + 1];
            
            // Xavier initialization
            double scale = std::sqrt(2.0 / (in_size + out_size));
            
            std::vector<std::vector<double>> W(out_size, std::vector<double>(in_size));
            for (size_t i = 0; i < out_size; ++i) {
                for (size_t j = 0; j < in_size; ++j) {
                    W[i][j] = dist(rng) * scale;
                }
            }
            weights.push_back(W);
            
            std::vector<double> b(out_size, 0.0);
            biases.push_back(b);
        }
    }
    
    // Forward pass for a batch of inputs
    std::vector<double> forward(const std::vector<double>& x) {
        size_t batch_size = 1;
        size_t input_size = architecture[0];
        
        // Initialize activations cache
        activations.clear();
        z_values.clear();
        
        // Input layer
        std::vector<std::vector<double>> A0(batch_size, std::vector<double>(input_size));
        A0[0] = x;
        activations.push_back(A0);
        
        // Hidden and output layers
        for (size_t l = 0; l < weights.size(); ++l) {
            size_t out_size = architecture[l + 1];
            std::vector<std::vector<double>> Z(batch_size, std::vector<double>(out_size, 0.0));
            
            // Z = A * W^T + b
            for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                for (size_t i = 0; i < out_size; ++i) {
                    Z[b_idx][i] = biases[l][i];
                    for (size_t j = 0; j < architecture[l]; ++j) {
                        Z[b_idx][i] += activations[l][b_idx][j] * weights[l][i][j];
                    }
                }
            }
            z_values.push_back(Z);
            
            // Apply activation
            std::vector<std::vector<double>> A = Z;
            if (l < weights.size() - 1) {
                // Hidden layers: ReLU
                for (auto& row : A) {
                    for (auto& val : row) {
                        val = relu(val);
                    }
                }
            } else {
                // Output layer: Sigmoid for binary classification
                for (auto& row : A) {
                    for (auto& val : row) {
                        val = sigmoid(val);
                    }
                }
            }
            activations.push_back(A);
        }
        
        return activations.back()[0];
    }
    
    // Compute gradients using backpropagation
    std::pair<std::vector<std::vector<std::vector<double>>>, 
              std::vector<std::vector<double>>> 
    compute_gradients(const std::vector<std::vector<double>>& X, 
                      const std::vector<double>& y) {
        size_t batch_size = X.size();
        size_t num_layers = weights.size();
        
        // Forward pass
        activations.clear();
        z_values.clear();
        
        // Input layer
        std::vector<std::vector<double>> A0 = X;
        activations.push_back(A0);
        
        for (size_t l = 0; l < num_layers; ++l) {
            size_t out_size = architecture[l + 1];
            std::vector<std::vector<double>> Z(batch_size, std::vector<double>(out_size, 0.0));
            
            for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                for (size_t i = 0; i < out_size; ++i) {
                    Z[b_idx][i] = biases[l][i];
                    for (size_t j = 0; j < architecture[l]; ++j) {
                        Z[b_idx][i] += activations[l][b_idx][j] * weights[l][i][j];
                    }
                }
            }
            z_values.push_back(Z);
            
            std::vector<std::vector<double>> A = Z;
            if (l < num_layers - 1) {
                // ReLU for hidden layers
                for (auto& row : A) {
                    for (size_t j = 0; j < row.size(); ++j) {
                        row[j] = relu(row[j]);
                    }
                }
            } else {
                // Sigmoid for output
                for (auto& row : A) {
                    for (size_t j = 0; j < row.size(); ++j) {
                        row[j] = sigmoid(row[j]);
                    }
                }
            }
            activations.push_back(A);
        }
        
        // Backward pass
        std::vector<std::vector<std::vector<double>>> dW(num_layers);
        std::vector<std::vector<double>> db(num_layers);
        
        // Output layer gradient (binary cross-entropy derivative)
        std::vector<std::vector<double>> delta(batch_size, std::vector<double>(architecture.back()));
        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            delta[b_idx][0] = activations.back()[b_idx][0] - y[b_idx];
        }
        
        for (int l = num_layers - 1; l >= 0; --l) {
            size_t in_size = architecture[l];
            size_t out_size = architecture[l + 1];
            
            dW[l].resize(out_size, std::vector<double>(in_size, 0.0));
            db[l].resize(out_size, 0.0);
            
            // Accumulate gradients over batch
            for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                for (size_t i = 0; i < out_size; ++i) {
                    db[l][i] += delta[b_idx][i];
                    for (size_t j = 0; j < in_size; ++j) {
                        dW[l][i][j] += delta[b_idx][i] * activations[l][b_idx][j];
                    }
                }
            }
            
            // Average over batch
            for (size_t i = 0; i < out_size; ++i) {
                db[l][i] /= batch_size;
                for (size_t j = 0; j < in_size; ++j) {
                    dW[l][i][j] /= batch_size;
                }
            }
            
            // Propagate error to previous layer (if not input layer)
            if (l > 0) {
                std::vector<std::vector<double>> delta_prev(batch_size, std::vector<double>(in_size));
                for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                    for (size_t j = 0; j < in_size; ++j) {
                        delta_prev[b_idx][j] = 0.0;
                        for (size_t i = 0; i < out_size; ++i) {
                            delta_prev[b_idx][j] += delta[b_idx][i] * weights[l][i][j];
                        }
                        // ReLU derivative
                        if (z_values[l-1][b_idx][j] <= 0) {
                            delta_prev[b_idx][j] = 0.0;
                        }
                    }
                }
                delta = delta_prev;
            }
        }
        
        return {dW, db};
    }
    
    // Training step
    template<typename OptimizerType>
    double train_step(const std::vector<std::vector<double>>& X,
                      const std::vector<double>& y,
                      OptimizerType& optimizer) {
        auto [dW, db] = compute_gradients(X, y);
        
        // Update weights
        for (size_t l = 0; l < weights.size(); ++l) {
            optimizer.step(weights[l], dW[l]);
        }
        
        // Update biases directly using gradient descent
        for (size_t l = 0; l < biases.size(); ++l) {
            for (size_t i = 0; i < biases[l].size(); ++i) {
                biases[l][i] -= optimizer.get_learning_rate() * db[l][i];
            }
        }
        
        // Compute loss
        std::vector<double> predictions;
        for (const auto& x : X) {
            predictions.push_back(forward(x)[0]);
        }
        return binary_cross_entropy(y, predictions);
    }
    
    // Predict class probabilities
    std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) {
        std::vector<double> probs;
        for (const auto& x : X) {
            probs.push_back(forward(x)[0]);
        }
        return probs;
    }
    
    // Predict classes
    std::vector<int> predict(const std::vector<std::vector<double>>& X, double threshold = 0.5) {
        std::vector<double> probs = predict_proba(X);
        std::vector<int> classes;
        for (double p : probs) {
            classes.push_back(p >= threshold ? 1 : 0);
        }
        return classes;
    }
    
    // Get architecture
    const std::vector<size_t>& get_architecture() const {
        return architecture;
    }
};

} // namespace nn
