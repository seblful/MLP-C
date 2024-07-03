#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include "MLP.h"

// ==== CONSTANTS ====
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define IMAGE_SIZE 784
#define LABEL_SIZE 10

// ==== DATA ====
const char trainImageFilename[] = "data/train-images.idx3-ubyte";
const char trainLabelFilename[] = "data/train-labels.idx1-ubyte";
const char testImageFilename[] = "data/t10k-images.idx3-ubyte";
const char testLabelFilename[] = "data/t10k-labels.idx1-ubyte";

int main()
{
    // Load training and testing data
    DataItem *trainData = createDataItem(trainImageFilename, trainLabelFilename, TRAIN_SIZE, IMAGE_SIZE, LABEL_SIZE);
    DataItem *testData = createDataItem(testImageFilename, testLabelFilename, TEST_SIZE, IMAGE_SIZE, LABEL_SIZE);

    // Initialize network
    int hidden_size = 24; // Hidden layer size
    double learning_rate = 0.000001;
    int epochs = 10;

    MLP *network = initialize_network(IMAGE_SIZE, hidden_size, LABEL_SIZE);

    // Train the network
    train(network, trainData, TRAIN_SIZE, epochs, learning_rate);

    // Evaluate the network
    double accuracy = evaluate(network, testData, TEST_SIZE);
    printf("Test Accuracy: %.2f%%\n", accuracy * 100);

    // Free memory
    freeDataItem(trainData, TRAIN_SIZE);
    freeDataItem(testData, TEST_SIZE);
    free_network(network);

    return 0;
}
