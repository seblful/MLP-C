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

// // Structs
// typedef struct
// {
//     double input[INPUT_SIZE];
//     double output[OUTPUT_SIZE];
// } DataItem;

// Functions declaration

int main()
{
    DataItem *trainData, *testData;

    trainData = createDataItem(trainImageFilename, trainLabelFilename, TRAIN_SIZE, IMAGE_SIZE, LABEL_SIZE);

    // uint8_t **trainImages, **testImages;
    // uint8_t *trainLabels, *testLabels;

    // // Load train and test data
    // trainImages = read_mnist_images(trainImageFilename);
    // trainLabels = read_mnist_labels(trainLabelFilename);

    // testImages = read_mnist_images(testImageFilename);
    // testLabels = read_mnist_labels(testLabelFilename);

    // // Print MNIST image by index
    // printMnist(trainImages, trainLabels, 0);

    return 0;
};