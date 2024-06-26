#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

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

// Functions declaration

int main()
{
    DataItem *trainData, *testData;

    // Load train data and test data
    trainData = createDataItem(trainImageFilename, trainLabelFilename, TRAIN_SIZE, IMAGE_SIZE, LABEL_SIZE);
    testData = createDataItem(testImageFilename, testLabelFilename, TEST_SIZE, IMAGE_SIZE, LABEL_SIZE);

    printMnistItem(trainData, 0, IMAGE_SIZE);

    return 0;
};