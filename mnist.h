#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Structs prototypes
typedef struct
{
    double *image;
    double *label;
} DataItem;

// Function prototypes
uint32_t read_uint32(FILE *file);

bool isValidFile(FILE *file, const char *filename);
bool isValidMagicNumber(FILE *file, const char *filename, uint32_t validMagicNumber);

uint8_t **read_mnist_images(const char *filename);
uint8_t *read_mnist_labels(const char *filename);

void printMnistImages(uint8_t **imagesArray, uint8_t *labelsArray, int imageIndex, int imageSize);
void printMnistItem(DataItem *items, int imageIndex, int imageSize);

DataItem *createDataItem(const char *imageFileame,
                         const char *labelFileame,
                         int setSize,
                         int imageSize,
                         int labelSize);

void freeDataItem(DataItem *item, int setSize);

#endif // MNIST_H