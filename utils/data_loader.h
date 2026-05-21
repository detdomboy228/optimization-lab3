#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

struct Dataset {
    std::vector<std::vector<double>> X;  // Features
    std::vector<double> y;                // Labels
    size_t n_features;
    size_t n_samples;
};

class DataLoader {
public:
    static Dataset load_csv(const std::string& filepath) {
        Dataset data;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        std::string line;
        bool header = true;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            if (header) {
                header = false;
                continue;
            }
            
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    // Skip non-numeric values
                }
            }
            
            if (row.size() >= 2) {
                // Last column is target
                data.y.push_back(row.back());
                row.pop_back();
                data.X.push_back(row);
            }
        }
        
        data.n_samples = data.X.size();
        if (data.n_samples > 0) {
            data.n_features = data.X[0].size();
        } else {
            data.n_features = 0;
        }
        
        return data;
    }
    
    // Split dataset into train and test sets
    static std::pair<Dataset, Dataset> train_test_split(const Dataset& data, 
                                                         double test_ratio = 0.2,
                                                         unsigned int seed = 42) {
        std::mt19937 rng(seed);
        std::vector<size_t> indices(data.n_samples);
        
        for (size_t i = 0; i < data.n_samples; ++i) {
            indices[i] = i;
        }
        
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t test_size = static_cast<size_t>(data.n_samples * test_ratio);
        size_t train_size = data.n_samples - test_size;
        
        Dataset train_data, test_data;
        train_data.n_features = data.n_features;
        test_data.n_features = data.n_features;
        
        for (size_t i = 0; i < train_size; ++i) {
            size_t idx = indices[i];
            train_data.X.push_back(data.X[idx]);
            train_data.y.push_back(data.y[idx]);
        }
        
        for (size_t i = train_size; i < data.n_samples; ++i) {
            size_t idx = indices[i];
            test_data.X.push_back(data.X[idx]);
            test_data.y.push_back(data.y[idx]);
        }
        
        train_data.n_samples = train_data.X.size();
        test_data.n_samples = test_data.X.size();
        
        return {train_data, test_data};
    }
    
    // Normalize features (standardization)
    static void normalize(Dataset& data) {
        if (data.n_samples == 0 || data.n_features == 0) return;
        
        std::vector<double> mean(data.n_features, 0.0);
        std::vector<double> std(data.n_features, 0.0);
        
        // Compute mean
        for (const auto& row : data.X) {
            for (size_t j = 0; j < data.n_features; ++j) {
                mean[j] += row[j];
            }
        }
        for (size_t j = 0; j < data.n_features; ++j) {
            mean[j] /= data.n_samples;
        }
        
        // Compute std
        for (const auto& row : data.X) {
            for (size_t j = 0; j < data.n_features; ++j) {
                std[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
            }
        }
        for (size_t j = 0; j < data.n_features; ++j) {
            std[j] = std::sqrt(std[j] / data.n_samples);
            if (std[j] < 1e-8) std[j] = 1.0;  // Avoid division by zero
        }
        
        // Normalize
        for (auto& row : data.X) {
            for (size_t j = 0; j < data.n_features; ++j) {
                row[j] = (row[j] - mean[j]) / std[j];
            }
        }
    }
    
    // Apply normalization parameters from training set to test set
    static void apply_normalization(Dataset& data, 
                                    const std::vector<double>& mean,
                                    const std::vector<double>& std_dev) {
        for (auto& row : data.X) {
            for (size_t j = 0; j < data.n_features && j < mean.size(); ++j) {
                double s = (j < std_dev.size()) ? std_dev[j] : 1.0;
                if (s < 1e-8) s = 1.0;
                row[j] = (row[j] - mean[j]) / s;
            }
        }
    }
    
    // Compute normalization parameters
    static std::pair<std::vector<double>, std::vector<double>> 
    compute_normalization_params(const Dataset& data) {
        std::vector<double> mean(data.n_features, 0.0);
        std::vector<double> std(data.n_features, 0.0);
        
        if (data.n_samples == 0) return {mean, std};
        
        // Compute mean
        for (const auto& row : data.X) {
            for (size_t j = 0; j < data.n_features; ++j) {
                mean[j] += row[j];
            }
        }
        for (size_t j = 0; j < data.n_features; ++j) {
            mean[j] /= data.n_samples;
        }
        
        // Compute std
        for (const auto& row : data.X) {
            for (size_t j = 0; j < data.n_features; ++j) {
                std[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
            }
        }
        for (size_t j = 0; j < data.n_features; ++j) {
            std[j] = std::sqrt(std[j] / data.n_samples);
            if (std[j] < 1e-8) std[j] = 1.0;
        }
        
        return {mean, std};
    }
};
