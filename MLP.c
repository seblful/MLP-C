#include <stdio.h>

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
void readFile();

int main()
{
    // Create train and test arrays with data items
    DataItem *trainData = (DataItem *)malloc(sizeof(TRAIN_SIZE));
    DataItem *testData = (DataItem *)malloc(sizeof(TEST_SIZE));

    // Load train and test data
    readFile(trainData, trainImageFilename, trainLabelFilename, TRAIN_SIZE);
    readFile(testData, testImageFilename, testLabelFilename, TEST_SIZE);

    return 0;
};

void readFile(DataItem *arr, const char image_filename[], const char label_filename[], int n)
{
    FILE *images = fopen(image_filename, "rb");
    FILE *labels = fopen(label_filename, "rb");
    unsigned char c;
    for (int i = 0; i < IMAGE_HEADER_SIZE; i++)
    {
        fgetc(images);
    };
    for (int i = 0; i < LABEL_HEADER_SIZE; i++)
        fgetc(labels);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            c = fgetc(images);
            arr[i].input[j] = (double)c / 255;
        }
        c = fgetc(labels);
        arr[i].output[c] = 1.0;
    }
}
