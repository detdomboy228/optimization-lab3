#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

struct ClassificationMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    int tp, tn, fp, fn;  // True positives, true negatives, false positives, false negatives
};

class Metrics {
public:
    static ClassificationMetrics compute_metrics(const std::vector<int>& y_true, 
                                                  const std::vector<int>& y_pred) {
        ClassificationMetrics metrics;
        metrics.tp = metrics.tn = metrics.fp = metrics.fn = 0;
        
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == 1 && y_pred[i] == 1) metrics.tp++;
            else if (y_true[i] == 0 && y_pred[i] == 0) metrics.tn++;
            else if (y_true[i] == 0 && y_pred[i] == 1) metrics.fp++;
            else if (y_true[i] == 1 && y_pred[i] == 0) metrics.fn++;
        }
        
        int total = metrics.tp + metrics.tn + metrics.fp + metrics.fn;
        metrics.accuracy = (total > 0) ? static_cast<double>(metrics.tp + metrics.tn) / total : 0.0;
        
        // Precision: TP / (TP + FP)
        if (metrics.tp + metrics.fp > 0) {
            metrics.precision = static_cast<double>(metrics.tp) / (metrics.tp + metrics.fp);
        } else {
            metrics.precision = 0.0;
        }
        
        // Recall: TP / (TP + FN)
        if (metrics.tp + metrics.fn > 0) {
            metrics.recall = static_cast<double>(metrics.tp) / (metrics.tp + metrics.fn);
        } else {
            metrics.recall = 0.0;
        }
        
        // F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        if (metrics.precision + metrics.recall > 0) {
            metrics.f1_score = 2.0 * metrics.precision * metrics.recall / 
                               (metrics.precision + metrics.recall);
        } else {
            metrics.f1_score = 0.0;
        }
        
        return metrics;
    }
    
    static void print_metrics(const ClassificationMetrics& m, const std::string& prefix = "") {
        std::cout << prefix << "Classification Metrics:" << std::endl;
        std::cout << prefix << "  Accuracy:  " << std::fixed << std::setprecision(4) << m.accuracy << std::endl;
        std::cout << prefix << "  Precision: " << std::fixed << std::setprecision(4) << m.precision << std::endl;
        std::cout << prefix << "  Recall:    " << std::fixed << std::setprecision(4) << m.recall << std::endl;
        std::cout << prefix << "  F1 Score:  " << std::fixed << std::setprecision(4) << m.f1_score << std::endl;
        std::cout << prefix << "  Confusion Matrix:" << std::endl;
        std::cout << prefix << "    TP=" << m.tp << " TN=" << m.tn 
                  << " FP=" << m.fp << " FN=" << m.fn << std::endl;
    }
};
