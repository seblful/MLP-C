#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "mnist.h"

// Function to read 4 bytes from a file and convert to a 32-bit integer
uint32_t read_uint32(FILE *file)
{
    uint8_t buffer[4];
    fread(buffer, 1, 4, file);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to read MNIST images
uint8_t **read_mnist_images(const char *filename)
{
    int number_of_images;
    int rows;
    int cols;

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file %s\n", filename);
        return NULL;
    }

    // Read the magic number and validate it
    uint32_t magic_number = read_uint32(file);
    if (magic_number != 2051)
    {
        fprintf(stderr, "Invalid magic number in %s\n", filename);
        fclose(file);
        return NULL;
    }

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
}

// Function to read MNIST labels
uint8_t *read_mnist_labels(const char *filename)
{
    int number_of_labels;

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file %s\n", filename);
        return NULL;
    }

    // Read the magic number and validate it
    uint32_t magic_number = read_uint32(file);
    if (magic_number != 2049)
    {
        fprintf(stderr, "Invalid magic number in %s\n", filename);
        fclose(file);
        return NULL;
    }

    // Read the number of labels
    number_of_labels = read_uint32(file);

    // Allocate memory for the labels
    uint8_t *labels = (uint8_t *)malloc(number_of_labels * sizeof(uint8_t));
    fread(labels, 1, number_of_labels, file);

    fclose(file);
    return labels;
}