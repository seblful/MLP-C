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

    printMnistItem(trainData, 0, IMAGE_SIZE);

    // Initialize MLP
    int inputSize = IMAGE_SIZE;
    int hiddenSize = 100; // Example hidden layer size, you can change it
    int outputSize = LABEL_SIZE;
    MLP *mlp = initializeMLP(inputSize, hiddenSize, outputSize);

    // Training parameters
    int epochs = 0;
    double learningRate = 0.01;

    // Train the MLP
    trainMLP(mlp, trainData, TRAIN_SIZE, epochs, learningRate);

    // Evaluate the MLP
    double accuracy = evaluateMLP(mlp, testData, TEST_SIZE);
    printf("Evaluation Accuracy: %f%%\n", accuracy * 100);

    // Free allocated memory
    freeMLP(mlp);
    freeDataItem(trainData, TRAIN_SIZE);
    freeDataItem(testData, TEST_SIZE);

    return 0;
};