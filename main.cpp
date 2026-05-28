#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <map>

#include "src/neural_network.h"
#include "optimizers/sgd_optimizer.h"
#include "optimizers/adam_optimizer.h"
#include "utils/data_loader.h"
#include "utils/metrics.h"

using namespace nn;

// Training function template
template<typename OptimizerType>
struct TrainingResult {
    double train_f1;
    double test_f1;
    double train_accuracy;
    double test_accuracy;
    std::vector<double> loss_history;
    double final_loss;
    int epochs_trained;
    ClassificationMetrics train_metrics;
    ClassificationMetrics test_metrics;
};

template<typename OptimizerType>
TrainingResult<OptimizerType> train_model(NeuralNetwork& model,
                                           const Dataset& train_data,
                                           const Dataset& test_data,
                                           OptimizerType& optimizer,
                                           int num_classes,
                                           int epochs = 500,
                                           int batch_size = 32,
                                           bool verbose = true) {
    TrainingResult<OptimizerType> result;
    result.epochs_trained = epochs;
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, train_data.n_samples - 1);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int batches = 0;
        
        // Mini-batch training
        for (size_t i = 0; i < train_data.n_samples; i += batch_size) {
            size_t current_batch_size = std::min(static_cast<size_t>(batch_size), 
                                                  train_data.n_samples - i);
            
            std::vector<std::vector<double>> X_batch;
            std::vector<double> y_batch;
            
            for (size_t j = 0; j < current_batch_size; ++j) {
                size_t idx = dist(rng);
                X_batch.push_back(train_data.X[idx]);
                y_batch.push_back(train_data.y[idx]);
            }
            
            double loss = model.train_step(X_batch, y_batch, optimizer);
            epoch_loss += loss;
            batches++;
        }
        
        result.final_loss = epoch_loss / batches;
        result.loss_history.push_back(result.final_loss);
        
        if (verbose && (epoch % 50 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << result.final_loss
                      << std::endl;
        }
    }
    
    // Evaluate on train and test sets
    std::vector<int> train_pred = model.predict(train_data.X);
    std::vector<int> train_true(train_data.y.begin(), train_data.y.end());
    result.train_metrics = Metrics::compute_metrics(train_true, train_pred, num_classes);
    result.train_f1 = result.train_metrics.macro_f1;
    result.train_accuracy = result.train_metrics.accuracy;
    
    std::vector<int> test_pred = model.predict(test_data.X);
    std::vector<int> test_true(test_data.y.begin(), test_data.y.end());
    result.test_metrics = Metrics::compute_metrics(test_true, test_pred, num_classes);
    result.test_f1 = result.test_metrics.macro_f1;
    result.test_accuracy = result.test_metrics.accuracy;
    
    return result;
}

template<typename OptimizerType>
void save_loss_history(const std::string& filepath,
                       const TrainingResult<OptimizerType>& result) {
    std::ofstream file(filepath);
    file << "epoch,loss\n";
    for (size_t i = 0; i < result.loss_history.size(); ++i) {
        file << (i + 1) << "," << result.loss_history[i] << "\n";
    }
}

void save_confusion_matrix(const std::string& filepath,
                           const ClassificationMetrics& metrics) {
    std::ofstream file(filepath);
    for (const auto& row : metrics.confusion_matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) file << ",";
            file << row[i];
        }
        file << "\n";
    }
}

void print_class_distribution(const std::string& name, const Dataset& data) {
    std::map<int, int> counts;
    for (double label : data.y) {
        counts[static_cast<int>(label)]++;
    }

    std::cout << name << " class distribution:";
    for (const auto& [label, count] : counts) {
        std::cout << " class " << label << "=" << count;
    }
    std::cout << std::endl;
}

void print_comparison(const std::string& dataset_name,
                      const TrainingResult<SGDOptimizer>& sgd_result,
                      const TrainingResult<AdamOptimizer>& adam_result) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results for " << dataset_name << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nSGD Optimizer:" << std::endl;
    std::cout << "  Train Accuracy: " << std::fixed << std::setprecision(4) << sgd_result.train_accuracy << std::endl;
    std::cout << "  Train F1: " << std::fixed << std::setprecision(4) << sgd_result.train_f1 << std::endl;
    std::cout << "  Test Accuracy:  " << std::fixed << std::setprecision(4) << sgd_result.test_accuracy << std::endl;
    std::cout << "  Test F1:  " << std::fixed << std::setprecision(4) << sgd_result.test_f1 << std::endl;
    std::cout << "  Final Loss: " << std::fixed << std::setprecision(6) << sgd_result.final_loss << std::endl;
    
    std::cout << "\nAdam Optimizer:" << std::endl;
    std::cout << "  Train Accuracy: " << std::fixed << std::setprecision(4) << adam_result.train_accuracy << std::endl;
    std::cout << "  Train F1: " << std::fixed << std::setprecision(4) << adam_result.train_f1 << std::endl;
    std::cout << "  Test Accuracy:  " << std::fixed << std::setprecision(4) << adam_result.test_accuracy << std::endl;
    std::cout << "  Test F1:  " << std::fixed << std::setprecision(4) << adam_result.test_f1 << std::endl;
    std::cout << "  Final Loss: " << std::fixed << std::setprecision(6) << adam_result.final_loss << std::endl;
    
    std::cout << "\nComparison:" << std::endl;
    std::cout << "  Best Test F1: " << std::fixed << std::setprecision(4) 
              << std::max(sgd_result.test_f1, adam_result.test_f1) << std::endl;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Neural Network Classification with SGD/Adam" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Load datasets
    std::cout << "\nLoading datasets..." << std::endl;
    Dataset d1 = DataLoader::load_csv("data/d1.csv");
    Dataset d2 = DataLoader::load_csv("data/d2.csv");
    DataLoader::relabel_to_zero_based(d1);
    DataLoader::relabel_to_zero_based(d2);
    int d1_classes = static_cast<int>(DataLoader::unique_labels(d1).size());
    int d2_classes = static_cast<int>(DataLoader::unique_labels(d2).size());
    
    std::cout << "Dataset D1: " << d1.n_samples << " samples, " 
              << d1.n_features << " features, " << d1_classes << " classes" << std::endl;
    std::cout << "Dataset D2: " << d2.n_samples << " samples, " 
              << d2.n_features << " features, " << d2_classes << " classes" << std::endl;
    
    // Stratified split keeps class proportions stable in train and test sets.
    auto [d1_train, d1_test] = DataLoader::stratified_train_test_split(d1, 0.2, 42);
    auto [d2_train, d2_test] = DataLoader::stratified_train_test_split(d2, 0.2, 42);
    
    std::cout << "\nD1 Train/Test split: " << d1_train.n_samples << "/" 
              << d1_test.n_samples << std::endl;
    print_class_distribution("D1 train", d1_train);
    print_class_distribution("D1 test ", d1_test);
    std::cout << "D2 Train/Test split: " << d2_train.n_samples << "/" 
              << d2_test.n_samples << std::endl;
    print_class_distribution("D2 train", d2_train);
    print_class_distribution("D2 test ", d2_test);
    
    // Compute normalization parameters from training data
    auto [d1_mean, d1_std] = DataLoader::compute_normalization_params(d1_train);
    auto [d2_mean, d2_std] = DataLoader::compute_normalization_params(d2_train);
    
    // Apply normalization
    DataLoader::apply_normalization(d1_train, d1_mean, d1_std);
    DataLoader::apply_normalization(d1_test, d1_mean, d1_std);
    DataLoader::apply_normalization(d2_train, d2_mean, d2_std);
    DataLoader::apply_normalization(d2_test, d2_mean, d2_std);
    
    // Define network architecture based on dataset
    // Output layer size equals the number of classes.
    
    int d1_hidden = 16;
    int d2_hidden = 32;
    
    std::cout << "\nNetwork Architecture for D1: " 
              << d1.n_features << " -> " << d1_hidden << " -> " << d1_classes << std::endl;
    std::cout << "Network Architecture for D2: " 
              << d2.n_features << " -> " << d2_hidden << " -> " << d2_classes << std::endl;
    
    // Training parameters
    int epochs = 500;
    int batch_size = 16;
    
    // =====================
    // Dataset D1 Experiments
    // =====================
    std::cout << "\n\n========== DATASET D1 ==========" << std::endl;
    
    // SGD on D1
    std::cout << "\n--- Training with SGD ---" << std::endl;
    std::filesystem::create_directories("results");

    NeuralNetwork model_d1_sgd({d1.n_features, static_cast<size_t>(d1_hidden), static_cast<size_t>(d1_classes)}, 42);
    SGDOptimizer sgd_opt_d1(0.01, 0.9);
    auto sgd_d1_result = train_model(model_d1_sgd, d1_train, d1_test, sgd_opt_d1, 
                                      d1_classes, epochs, batch_size, true);
    std::cout << "Done" << std::endl;
    
    // Adam on D1
    std::cout << "--- Training with Adam ---" << std::endl;
    NeuralNetwork model_d1_adam({d1.n_features, static_cast<size_t>(d1_hidden), static_cast<size_t>(d1_classes)}, 42);
    AdamOptimizer adam_opt_d1(0.001, 0.9, 0.999, 1e-8);
    auto adam_d1_result = train_model(model_d1_adam, d1_train, d1_test, adam_opt_d1,
                                       d1_classes, epochs, batch_size, true);
    std::cout << "Done" << std::endl;
    
    print_comparison("D1", sgd_d1_result, adam_d1_result);
    save_loss_history("results/d1_sgd_loss.csv", sgd_d1_result);
    save_loss_history("results/d1_adam_loss.csv", adam_d1_result);
    save_confusion_matrix("results/d1_sgd_confusion.csv", sgd_d1_result.test_metrics);
    save_confusion_matrix("results/d1_adam_confusion.csv", adam_d1_result.test_metrics);
    
    // =====================
    // Dataset D2 Experiments
    // =====================
    std::cout << "\n\n========== DATASET D2 ==========" << std::endl;
    
    // SGD on D2
    std::cout << "\n--- Training with SGD ---" << std::endl;
    NeuralNetwork model_d2_sgd({d2.n_features, static_cast<size_t>(d2_hidden), static_cast<size_t>(d2_classes)}, 42);
    SGDOptimizer sgd_opt_d2(0.01, 0.9);
    auto sgd_d2_result = train_model(model_d2_sgd, d2_train, d2_test, sgd_opt_d2,
                                      d2_classes, epochs, batch_size, true);
    std::cout << "Done" << std::endl;
    
    // Adam on D2
    std::cout << "--- Training with Adam ---" << std::endl;
    NeuralNetwork model_d2_adam({d2.n_features, static_cast<size_t>(d2_hidden), static_cast<size_t>(d2_classes)}, 42);
    AdamOptimizer adam_opt_d2(0.001, 0.9, 0.999, 1e-8);
    auto adam_d2_result = train_model(model_d2_adam, d2_train, d2_test, adam_opt_d2,
                                       d2_classes, epochs, batch_size, true);
    std::cout << "Done" << std::endl;
    
    print_comparison("D2", sgd_d2_result, adam_d2_result);
    save_loss_history("results/d2_sgd_loss.csv", sgd_d2_result);
    save_loss_history("results/d2_adam_loss.csv", adam_d2_result);
    save_confusion_matrix("results/d2_sgd_confusion.csv", sgd_d2_result.test_metrics);
    save_confusion_matrix("results/d2_adam_confusion.csv", adam_d2_result.test_metrics);

    std::ofstream summary("results/summary.csv");
    summary << "dataset,optimizer,train_accuracy,train_macro_f1,test_accuracy,test_macro_f1,final_loss\n";
    summary << "D1,SGD," << sgd_d1_result.train_accuracy << "," << sgd_d1_result.train_f1 << ","
            << sgd_d1_result.test_accuracy << "," << sgd_d1_result.test_f1 << "," << sgd_d1_result.final_loss << "\n";
    summary << "D1,Adam," << adam_d1_result.train_accuracy << "," << adam_d1_result.train_f1 << ","
            << adam_d1_result.test_accuracy << "," << adam_d1_result.test_f1 << "," << adam_d1_result.final_loss << "\n";
    summary << "D2,SGD," << sgd_d2_result.train_accuracy << "," << sgd_d2_result.train_f1 << ","
            << sgd_d2_result.test_accuracy << "," << sgd_d2_result.test_f1 << "," << sgd_d2_result.final_loss << "\n";
    summary << "D2,Adam," << adam_d2_result.train_accuracy << "," << adam_d2_result.train_f1 << ","
            << adam_d2_result.test_accuracy << "," << adam_d2_result.test_f1 << "," << adam_d2_result.final_loss << "\n";
    
    // =====================
    // Summary
    // =====================
    std::cout << "\n\n============================================" << std::endl;
    std::cout << "FINAL SUMMARY" << std::endl;
    std::cout << "============================================" << std::endl;
    
    double best_d1_f1 = std::max(sgd_d1_result.test_f1, adam_d1_result.test_f1);
    double best_d2_f1 = std::max(sgd_d2_result.test_f1, adam_d2_result.test_f1);
    double best_d1_accuracy = std::max(sgd_d1_result.test_accuracy, adam_d1_result.test_accuracy);
    double best_d2_accuracy = std::max(sgd_d2_result.test_accuracy, adam_d2_result.test_accuracy);
    
    std::cout << "\nBest Accuracy on D1 (test): " << std::fixed << std::setprecision(4) << best_d1_accuracy << std::endl;
    std::cout << "\nBest F1 on D1 (test): " << std::fixed << std::setprecision(4) << best_d1_f1 << std::endl;
    std::cout << "Best Accuracy on D2 (test): " << std::fixed << std::setprecision(4) << best_d2_accuracy << std::endl;
    std::cout << "Best F1 on D2 (test): " << std::fixed << std::setprecision(4) << best_d2_f1 << std::endl;
    
    // Note: D3 will be evaluated during defense
    std::cout << "\nNote: D3 evaluation will be performed during lab defense." << std::endl;
    
    // Estimated score (without D3)
    std::cout << "\nPartial score estimate (0.3*D1 + 0.3*D2, without D3): " 
              << std::fixed << std::setprecision(4) 
              << (0.3 * best_d1_f1 + 0.3 * best_d2_f1) << std::endl;
    
    std::cout << "\nTo achieve final score >= 0.55, need good performance on D3 as well." << std::endl;
    
    return 0;
}
