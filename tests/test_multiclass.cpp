#include <cassert>
#include <cmath>
#include <map>
#include <vector>

#include "src/neural_network.h"
#include "src/nn_utils.h"
#include "utils/data_loader.h"
#include "utils/metrics.h"

using namespace nn;

namespace {

bool close(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

} // namespace

int main() {
    auto probs = softmax({1.0, 2.0, 3.0});
    assert(probs.size() == 3);
    assert(close(probs[0] + probs[1] + probs[2], 1.0));
    assert(probs[2] > probs[1]);
    assert(probs[1] > probs[0]);

    std::vector<double> labels = {0, 2};
    std::vector<std::vector<double>> pred = {
        {0.8, 0.1, 0.1},
        {0.1, 0.2, 0.7},
    };
    assert(multiclass_cross_entropy(labels, pred) < 0.3);

    auto metrics = Metrics::compute_metrics({0, 1, 2, 2}, {0, 2, 2, 1}, 3);
    assert(close(metrics.accuracy, 0.5));
    assert(metrics.confusion_matrix.size() == 3);
    assert(metrics.confusion_matrix[1][2] == 1);
    assert(metrics.confusion_matrix[2][1] == 1);
    assert(metrics.macro_f1 > 0.49 && metrics.macro_f1 < 0.51);

    NeuralNetwork model({2, 4, 3}, 42);
    auto batch_probs = model.predict_proba({{0.0, 1.0}, {1.0, 0.0}});
    assert(batch_probs.size() == 2);
    assert(batch_probs[0].size() == 3);
    assert(close(batch_probs[0][0] + batch_probs[0][1] + batch_probs[0][2], 1.0));

    auto classes = model.predict({{0.0, 1.0}, {1.0, 0.0}});
    assert(classes.size() == 2);
    assert(classes[0] >= 0 && classes[0] < 3);

    Dataset dataset;
    dataset.n_features = 2;
    for (int label = 0; label < 3; ++label) {
        for (int i = 0; i < 10; ++i) {
            dataset.X.push_back({static_cast<double>(label), static_cast<double>(i)});
            dataset.y.push_back(label);
        }
    }
    dataset.n_samples = dataset.X.size();

    auto [train_data, test_data] = DataLoader::stratified_train_test_split(dataset, 0.2, 42);
    assert(train_data.n_samples == 24);
    assert(test_data.n_samples == 6);

    std::map<int, int> test_counts;
    for (double label : test_data.y) {
        test_counts[static_cast<int>(label)]++;
    }
    assert(test_counts[0] == 2);
    assert(test_counts[1] == 2);
    assert(test_counts[2] == 2);

    return 0;
}
