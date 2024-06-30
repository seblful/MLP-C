#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mnist.h"

// Structure to represent the MLP
typedef struct
{
    int input_size;
    int hidden_size;
    int output_size;
    double *input;
    double *hidden;
    double *output;
    double *hidden_bias;
    double *output_bias;
    double **input_hidden_weights;
    double **hidden_output_weights;
    double *hidden_delta;
    double *output_delta;
} MLP;

// Function prototypes
MLP *initialize_network(int input_size, int hidden_size, int output_size);
void forward_propagation(MLP *network, double *input);
void backpropagation(MLP *network, double *input, double *target, double learning_rate);
void update_weights(MLP *network, double learning_rate);
double calculate_error(double *output, double *target, int size);
void train(MLP *network, DataItem *train_data, int train_size, int epochs, double learning_rate);
double evaluate(MLP *network, DataItem *test_data, int test_size);
void free_network(MLP *network);

#endif // MLP_H
