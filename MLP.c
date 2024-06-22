#include <stdio.h>

// ==== CONSTANTS ====
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// ==== DATA ====
const char train_image_filename[] = "mnist_data/train-images.idx3-ubyte";
const char train_label_filename[] = "mnist_data/train-labels.idx1-ubyte";
const char test_image_filename[] = "mnist_data/t10k-images.idx3-ubyte";
const char test_label_filename[] = "mnist_data/t10k-labels.idx1-ubyte";

const int IMAGE_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;

// Functions declaration
void read_file();

// Structs
typedef struct
{
    double input[INPUT_SIZE];
    double output[OUTPUT_SIZE];
} DataItem;

DataItem train_data[TRAIN_SIZE];
DataItem test_data[TEST_SIZE];

int main()
{
    DataItem a;
    a = train_data[0];
    return 0;
};

void read_file(DataItem *arr, const char image_filename[], const char label_filename[], int n)
{
    FILE *images = fopen(image_filename, "rb");
    FILE *labels = fopen(label_filename, "rb");
    unsigned char c;
    for (int i = 0; i < IMAGE_HEADER_SIZE; i++)
        fgetc(images);
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
