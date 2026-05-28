CXX = g++
CXXFLAGS = -std=c++17 -O2 -I.
TARGET = neural_net
SOURCES = main.cpp
TEST_TARGET = tests/test_multiclass
HEADERS = src/neural_network.h src/nn_utils.h utils/data_loader.h utils/metrics.h optimizers/optimizer_base.h optimizers/sgd_optimizer.h optimizers/adam_optimizer.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

debug: CXXFLAGS += -g -fsanitize=address
debug: clean $(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): tests/test_multiclass.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) tests/test_multiclass.cpp

clean:
	rm -f $(TARGET) $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

visualize: run
	python3 visualize_results.py

.PHONY: all clean debug run test visualize
