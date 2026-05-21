CXX = g++
CXXFLAGS = -std=c++17 -O2 -I.
TARGET = neural_net
SOURCES = main.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

debug: CXXFLAGS += -g -fsanitize=address
debug: clean $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean debug run
