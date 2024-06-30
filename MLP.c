#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mnist.h"
#include "MLP.h"

// Function to initialize MLP
MLP *initializeMLP(int inputSize, int hiddenSize, int outputSize)
{
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->inputSize = inputSize;
    mlp->hiddenSize = hiddenSize;
    mlp->outputSize = outputSize;

    // Allocate memory for weights and biases
    mlp->weight1 = (double *)malloc(inputSize * hiddenSize * sizeof(double));
    mlp->weight2 = (double *)malloc(hiddenSize * outputSize * sizeof(double));
    mlp->bias1 = (double *)malloc(hiddenSize * sizeof(double));
    mlp->bias2 = (double *)malloc(outputSize * sizeof(double));

    // Initialize weights and biases with small random values
    for (int i = 0; i < inputSize * hiddenSize; i++)
    {
        mlp->weight1[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < hiddenSize * outputSize; i++)
    {
        mlp->weight2[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < hiddenSize; i++)
    {
        mlp->bias1[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < outputSize; i++)
    {
        mlp->bias2[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    return mlp;
}

// Activation function (Sigmoid)
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Forward propagation
void forward(MLP *mlp, double *input)
{
    // Hidden layer
    for (int i = 0; i < mlp->hiddenSize; i++)
    {
        mlp->hiddenLayer[i] = 0;
        for (int j = 0; j < mlp->inputSize; j++)
        {
            mlp->hiddenLayer[i] += input[j] * mlp->weight1[j * mlp->hiddenSize + i];
        }
        mlp->hiddenLayer[i] += mlp->bias1[i];
        mlp->hiddenLayer[i] = sigmoid(mlp->hiddenLayer[i]);
    }

    // Output layer
    for (int i = 0; i < mlp->outputSize; i++)
    {
        mlp->outputLayer[i] = 0;
        for (int j = 0; j < mlp->hiddenSize; j++)
        {
            mlp->outputLayer[i] += mlp->hiddenLayer[j] * mlp->weight2[j * mlp->outputSize + i];
        }
        mlp->outputLayer[i] += mlp->bias2[i];
        mlp->outputLayer[i] = sigmoid(mlp->outputLayer[i]);
    }
}

// Backward propagation
void backward(MLP *mlp, double *input, double *target, double learningRate)
{
    double *outputError = (double *)malloc(mlp->outputSize * sizeof(double));
    double *hiddenError = (double *)malloc(mlp->hiddenSize * sizeof(double));

    // Calculate output error
    for (int i = 0; i < mlp->outputSize; i++)
    {
        outputError[i] = (target[i] - mlp->outputLayer[i]) * sigmoidDerivative(mlp->outputLayer[i]);
    }

    // Calculate hidden layer error
    for (int i = 0; i < mlp->hiddenSize; i++)
    {
        hiddenError[i] = 0;
        for (int j = 0; j < mlp->outputSize; j++)
        {
            hiddenError[i] += outputError[j] * mlp->weight2[i * mlp->outputSize + j];
        }
        hiddenError[i] *= sigmoidDerivative(mlp->hiddenLayer[i]);
    }

    // Update weights and biases for the second layer
    for (int i = 0; i < mlp->outputSize; i++)
    {
        for (int j = 0; j < mlp->hiddenSize; j++)
        {
            mlp->weight2[j * mlp->outputSize + i] += learningRate * outputError[i] * mlp->hiddenLayer[j];
        }
        mlp->bias2[i] += learningRate * outputError[i];
    }

    // Update weights and biases for the first layer
    for (int i = 0; i < mlp->hiddenSize; i++)
    {
        for (int j = 0; j < mlp->inputSize; j++)
        {
            mlp->weight1[j * mlp->hiddenSize + i] += learningRate * hiddenError[i] * input[j];
        }
        mlp->bias1[i] += learningRate * hiddenError[i];
    }

    // Free allocated memory for errors
    free(outputError);
    free(hiddenError);
}

// Train MLP
void trainMLP(MLP *mlp, DataItem *trainData, int trainSize, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalError = 0;
        for (int i = 0; i < trainSize; i++)
        {
            forward(mlp, trainData[i].image);
            backward(mlp, trainData[i].image, trainData[i].label, learningRate);

            // Calculate error
            for (int j = 0; j < mlp->outputSize; j++)
            {
                double error = trainData[i].label[j] - mlp->outputLayer[j];
                totalError += error * error;
            }
        }
        printf("Epoch %d, Error: %f\n", epoch + 1, totalError / trainSize);
    }
}

// Evaluate MLP
double evaluateMLP(MLP *mlp, DataItem *testData, int testSize)
{
    int correct = 0;
    for (int i = 0; i < testSize; i++)
    {
        forward(mlp, testData[i].image);
        int predicted = 0, actual = 0;
        for (int j = 0; j < mlp->outputSize; j++)
        {
            if (mlp->outputLayer[j] > mlp->outputLayer[predicted])
            {
                predicted = j;
            }
            if (testData[i].label[j] == 1)
            {
                actual = j;
            }
        }
        if (predicted == actual)
        {
            correct++;
        }
    }
    return (double)correct / testSize;
}

// Free MLP memory
void freeMLP(MLP *mlp)
{
    free(mlp->weight1);
    free(mlp->weight2);
    free(mlp->bias1);
    free(mlp->bias2);
    free(mlp);
}