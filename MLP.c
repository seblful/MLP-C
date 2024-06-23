#include <stdio.h>
#include "mnist.h"

// ==== CONSTANTS ====
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// ==== DATA ====
const char trainImageFilename[] = "mnist_data/train-images.idx3-ubyte";
const char trainLabelFilename[] = "mnist_data/train-labels.idx1-ubyte";
const char testImageFilename[] = "mnist_data/t10k-images.idx3-ubyte";
const char testLabelFilename[] = "mnist_data/t10k-labels.idx1-ubyte";

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

    uint8_t **images;

    // Load train and test data

    images = read_mnist_images(trainImageFilename);

    return 0;
};