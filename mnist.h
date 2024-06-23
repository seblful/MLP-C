#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>

// Function prototypes
uint32_t read_uint32(FILE *file);
uint8_t **read_mnist_images(const char *filename);
uint8_t *read_mnist_labels(const char *filename);

#endif // MNIST_H