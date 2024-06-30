#ifndef MLP_H
#define MLP_H

#include "mnist.h"

// Struct prototypes
typedef struct
{
    int input_size;
    int hidden_size;
    int output_size;
    double *input;
    double *hidden;
    double *output;
    double *weights_input_hidden;
    double *weights_hidden_output;
    double *bias_hidden;
    double *bias_output;
} MLP;

// Function prototypes
MLP *initialize_mlp(int input_size, int hidden_size, int output_size);
void free_mlp(MLP *mlp);
double sigmoid(double x);
double sigmoid_derivative(double x);
void forward_propagation(MLP *mlp, double *input);
void backpropagation(MLP *mlp, double *target, double learning_rate);
double compute_loss(double *output, double *target, int size);
void update_weights(MLP *mlp, double learning_rate);
void train(MLP *mlp, DataItem *train_data, int train_size, int epochs, double learning_rate);
double evaluate(MLP *mlp, DataItem *test_data, int test_size);

#endif // MLP_H
