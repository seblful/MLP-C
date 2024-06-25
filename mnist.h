#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Function prototypes
uint32_t read_uint32(FILE *file);

bool isValidFile(FILE *file, const char *filename);
bool isValidMagicNumber(FILE *file, const char *filename, uint32_t validMagicNumber);

uint8_t **read_mnist_images(const char *filename);
uint8_t *read_mnist_labels(const char *filename);

void printMnist(uint8_t **imagesArray, uint8_t *labelsArray, int imageIndex);

#endif // MNIST_H