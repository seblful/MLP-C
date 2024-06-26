#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "mnist.h"

struct DataItem
{
    // double **data;
    double **data;
    uint8_t *label;
};

DataItem *createDataItem(const char *imageFileame,
                         const char *labelFileame,
                         int setSize,
                         int imageSize,
                         int labelSize)
{
    uint8_t **images, *labels;
    double temp;

    // Allocate memory for DataItem
    DataItem *item = (DataItem *)malloc(setSize * sizeof(DataItem));

    item->data = (double **)malloc(setSize * sizeof(double *));
    for (int i = 0; i < setSize; i++)
    {
        item->data[i] = (double *)malloc(imageSize * sizeof(double));
    };

    item->label = (uint8_t *)malloc(setSize * sizeof(uint8_t));

    // Load images and labels in uint8
    images = read_mnist_images(imageFileame);
    labels = read_mnist_labels(labelFileame);

    // Set images and labels data in item in proper format
    // Iterate in range of set size
    for (int i = 0; i < setSize; i++)
    {

        // Convert images to double format, standardize them and set to data
        for (int j = 0; j < imageSize; j++)
        {
            temp = ((double)images[i][j]) / 255.0;
            item->data[i][j] = temp;
        };
        item->label[i] = labels[i];
    };

    return item;
};

// Function to read 4 bytes from a file and convert to a 32-bit integer
uint32_t read_uint32(FILE *file)
{
    uint8_t buffer[4];
    fread(buffer, 1, 4, file);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
};

bool isValidFile(FILE *file, const char *filename)
{
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file %s\n", filename);
        return false;
    };
    return true;
};

// Function to read magic number and validate it
bool isValidMagicNumber(FILE *file, const char *filename, uint32_t validMagicNumber)
{
    // Read the magic number and validate it
    uint32_t magicNumber = read_uint32(file);
    if (magicNumber != validMagicNumber)
    {
        fprintf(stderr, "Invalid magic number in %s\n", filename);
        fclose(file);
        return false;
    }
    return true;
};

// Function to read MNIST images
uint8_t **read_mnist_images(const char *filename)
{
    int number_of_images;
    int rows;
    int cols;

    // Open file and validate it
    FILE *file = fopen(filename, "rb");
    if (isValidFile(file, filename) == false)
    {
        fclose(file);
        return NULL;
    };

    // Read the magic number and validate it
    if (isValidMagicNumber(file, filename, 2051) == false)
    {
        return NULL;
    };

    // Read the number of images, rows, and columns
    number_of_images = read_uint32(file);
    rows = read_uint32(file);
    cols = read_uint32(file);

    // Allocate memory for the images
    uint8_t **images = (uint8_t **)malloc(number_of_images * sizeof(uint8_t *));
    for (int i = 0; i < number_of_images; i++)
    {
        images[i] = (uint8_t *)malloc(rows * cols * sizeof(uint8_t));
        fread(images[i], 1, rows * cols, file);
    }

    fclose(file);
    return images;
};

// Function to read MNIST labels
uint8_t *read_mnist_labels(const char *filename)
{
    int number_of_labels;

    FILE *file = fopen(filename, "rb");
    if (isValidFile(file, filename) == false)
    {
        fclose(file);
        return NULL;
    };

    // Read the magic number and validate it
    if (isValidMagicNumber(file, filename, 2049) == false)
    {
        return NULL;
    };

    // Read the number of labels
    number_of_labels = read_uint32(file);

    // Allocate memory for the labels
    uint8_t *labels = (uint8_t *)malloc(number_of_labels * sizeof(uint8_t));
    fread(labels, 1, number_of_labels, file);

    fclose(file);
    return labels;
};

void printMnistImages(uint8_t **imagesArray, uint8_t *labelsArray, int imageIndex, int imageSize)
{
    int side = (int)sqrt((double)imageSize);

    // Print label
    printf("Label of image is %d.\n", labelsArray[imageIndex]);

    // Print array with alignment
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {
            printf("%3d ", imagesArray[imageIndex][i * side + j]);
        }
        printf("\n");
    };
};

void printMnistItem(DataItem *item, int imageIndex, int imageSize)
{
    int side = (int)sqrt((double)imageSize);

    // Print label
    printf("Label of image is %d.\n", item->label[imageIndex]);

    // Print array with alignment
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {
            printf("%3.0f ", item->data[imageIndex][i * side + j]);
        }
        printf("\n");
    };
};