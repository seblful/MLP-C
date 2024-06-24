#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"

// ==== CONSTANTS ====
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// ==== DATA ====
const char trainImageFilename[] = "data/train-images.idx3-ubyte";
const char trainLabelFilename[] = "data/train-labels.idx1-ubyte";
const char testImageFilename[] = "data/t10k-images.idx3-ubyte";
const char testLabelFilename[] = "data/t10k-labels.idx1-ubyte";

const int IMAGE_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;

// Structs
typedef struct
{
    double input[INPUT_SIZE];
    double output[OUTPUT_SIZE];
} DataItem;

// Functions declaration

int main()
{
    // DataItem *trainData = (DataItem *)malloc(TRAIN_SIZE * sizeof(DataItem));
    // DataItem *testData = (DataItem *)malloc(TEST_SIZE * sizeof(DataItem));

    uint8_t **trainImages, **testImages;
    uint8_t *trainLabels, *test_labels;

    // Load train and test data

    trainImages = read_mnist_images(trainImageFilename);
    trainLabels = read_mnist_labels(trainLabelFilename);

    // printf("%d ", sizeof(trainImages));
    // printf("%d ", sizeof(trainImages[0]));
    // printf("%d\n", sizeof(trainImages[0][0]));

    for (int i = 0; i < 255; i++)
    {
        printf("%d", trainImages[0][i]);
    };

    return 0;
};