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
    double macro_precision;
    double macro_recall;
    double macro_f1;
    int tp, tn, fp, fn;  // True positives, true negatives, false positives, false negatives
    std::vector<std::vector<int>> confusion_matrix;
};

class Metrics {
public:
    static ClassificationMetrics compute_metrics(const std::vector<int>& y_true,
                                                  const std::vector<int>& y_pred,
                                                  int num_classes = 0) {
        ClassificationMetrics metrics;
        metrics.tp = metrics.tn = metrics.fp = metrics.fn = 0;

        if (num_classes <= 0) {
            for (int label : y_true) num_classes = std::max(num_classes, label + 1);
            for (int label : y_pred) num_classes = std::max(num_classes, label + 1);
        }
        metrics.confusion_matrix.assign(num_classes, std::vector<int>(num_classes, 0));

        int correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) correct++;
            if (y_true[i] >= 0 && y_true[i] < num_classes &&
                y_pred[i] >= 0 && y_pred[i] < num_classes) {
                metrics.confusion_matrix[y_true[i]][y_pred[i]]++;
            }

            if (y_true[i] == 1 && y_pred[i] == 1) metrics.tp++;
            else if (y_true[i] == 0 && y_pred[i] == 0) metrics.tn++;
            else if (y_true[i] == 0 && y_pred[i] == 1) metrics.fp++;
            else if (y_true[i] == 1 && y_pred[i] == 0) metrics.fn++;
        }

        int total = static_cast<int>(y_true.size());
        metrics.accuracy = (total > 0) ? static_cast<double>(correct) / total : 0.0;

        metrics.macro_precision = 0.0;
        metrics.macro_recall = 0.0;
        metrics.macro_f1 = 0.0;

        for (int c = 0; c < num_classes; ++c) {
            int class_tp = metrics.confusion_matrix[c][c];
            int class_fp = 0;
            int class_fn = 0;

            for (int k = 0; k < num_classes; ++k) {
                if (k != c) {
                    class_fp += metrics.confusion_matrix[k][c];
                    class_fn += metrics.confusion_matrix[c][k];
                }
            }

            double class_precision = (class_tp + class_fp > 0)
                ? static_cast<double>(class_tp) / (class_tp + class_fp)
                : 0.0;
            double class_recall = (class_tp + class_fn > 0)
                ? static_cast<double>(class_tp) / (class_tp + class_fn)
                : 0.0;
            double class_f1 = (class_precision + class_recall > 0.0)
                ? 2.0 * class_precision * class_recall / (class_precision + class_recall)
                : 0.0;

            metrics.macro_precision += class_precision;
            metrics.macro_recall += class_recall;
            metrics.macro_f1 += class_f1;
        }

        if (num_classes > 0) {
            metrics.macro_precision /= num_classes;
            metrics.macro_recall /= num_classes;
            metrics.macro_f1 /= num_classes;
        }

        metrics.precision = metrics.macro_precision;
        metrics.recall = metrics.macro_recall;
        metrics.f1_score = metrics.macro_f1;

        return metrics;
    }
    
    static void print_metrics(const ClassificationMetrics& m, const std::string& prefix = "") {
        std::cout << prefix << "Classification Metrics:" << std::endl;
        std::cout << prefix << "  Accuracy:  " << std::fixed << std::setprecision(4) << m.accuracy << std::endl;
        std::cout << prefix << "  Precision: " << std::fixed << std::setprecision(4) << m.precision << std::endl;
        std::cout << prefix << "  Recall:    " << std::fixed << std::setprecision(4) << m.recall << std::endl;
        std::cout << prefix << "  Macro F1:  " << std::fixed << std::setprecision(4) << m.macro_f1 << std::endl;
        std::cout << prefix << "  Confusion Matrix:" << std::endl;
        for (const auto& row : m.confusion_matrix) {
            std::cout << prefix << "    ";
            for (int value : row) {
                std::cout << std::setw(4) << value;
            }
            std::cout << std::endl;
        }
    }
};
