#ifndef MLP_H
#define MLP_H

#include "mnist.h"

typedef struct
{
    int inputSize;
    int hiddenSize;
    int outputSize;

    double *weight1;
    double *weight2;
    double *bias1;
    double *bias2;

    double hiddenLayer[100]; // Assuming hidden layer size will not exceed 100
    double outputLayer[10];  // Assuming output layer size is 10 (for MNIST)
} MLP;

// Function prototypes
MLP *initializeMLP(int inputSize, int hiddenSize, int outputSize);
void forward(MLP *mlp, double *input);
void backward(MLP *mlp, double *input, double *target, double learningRate);
void trainMLP(MLP *mlp, DataItem *trainData, int trainSize, int epochs, double learningRate);
double evaluateMLP(MLP *mlp, DataItem *testData, int testSize);
void freeMLP(MLP *mlp);

#endif // MLP_H