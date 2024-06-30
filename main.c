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

    // Initialize MLP
    int input_size = IMAGE_SIZE;
    int hidden_size = 24; // Example hidden layer size
    int output_size = LABEL_SIZE;
    MLP *mlp = initialize_mlp(input_size, hidden_size, output_size);

    // Train MLP
    int epochs = 10;
    double learning_rate = 0.01;
    train(mlp, trainData, TRAIN_SIZE, epochs, learning_rate);

    // Evaluate MLP
    double accuracy = evaluate(mlp, testData, TEST_SIZE);
    printf("Accuracy: %f\n", accuracy);

    // Free resources
    freeDataItem(trainData, TRAIN_SIZE);
    freeDataItem(testData, TEST_SIZE);
    free_mlp(mlp);

    return 0;
}
