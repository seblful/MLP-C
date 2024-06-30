#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MLP.h"

#define RANDOM_WEIGHT() ((double)rand() / RAND_MAX - 0.5)

MLP *initialize_mlp(int input_size, int hidden_size, int output_size)
{
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    mlp->input = (double *)malloc(input_size * sizeof(double));
    mlp->hidden = (double *)malloc(hidden_size * sizeof(double));
    mlp->output = (double *)malloc(output_size * sizeof(double));

    mlp->weights_input_hidden = (double *)malloc(input_size * hidden_size * sizeof(double));
    mlp->weights_hidden_output = (double *)malloc(hidden_size * output_size * sizeof(double));
    mlp->bias_hidden = (double *)malloc(hidden_size * sizeof(double));
    mlp->bias_output = (double *)malloc(output_size * sizeof(double));

    for (int i = 0; i < input_size * hidden_size; i++)
    {
        mlp->weights_input_hidden[i] = RANDOM_WEIGHT();
    }
    for (int i = 0; i < hidden_size * output_size; i++)
    {
        mlp->weights_hidden_output[i] = RANDOM_WEIGHT();
    }
    for (int i = 0; i < hidden_size; i++)
    {
        mlp->bias_hidden[i] = RANDOM_WEIGHT();
    }
    for (int i = 0; i < output_size; i++)
    {
        mlp->bias_output[i] = RANDOM_WEIGHT();
    }

    return mlp;
}

void free_mlp(MLP *mlp)
{
    free(mlp->input);
    free(mlp->hidden);
    free(mlp->output);
    free(mlp->weights_input_hidden);
    free(mlp->weights_hidden_output);
    free(mlp->bias_hidden);
    free(mlp->bias_output);
    free(mlp);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

void forward_propagation(MLP *mlp, double *input)
{
    for (int i = 0; i < mlp->input_size; i++)
    {
        mlp->input[i] = input[i];
    }

    for (int j = 0; j < mlp->hidden_size; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < mlp->input_size; i++)
        {
            sum += mlp->input[i] * mlp->weights_input_hidden[i * mlp->hidden_size + j];
        }
        sum += mlp->bias_hidden[j];
        mlp->hidden[j] = sigmoid(sum);
    }

    for (int k = 0; k < mlp->output_size; k++)
    {
        double sum = 0.0;
        for (int j = 0; j < mlp->hidden_size; j++)
        {
            sum += mlp->hidden[j] * mlp->weights_hidden_output[j * mlp->output_size + k];
        }
        sum += mlp->bias_output[k];
        mlp->output[k] = sigmoid(sum);
    }
}

void backpropagation(MLP *mlp, double *target, double learning_rate)
{
    double *output_errors = (double *)malloc(mlp->output_size * sizeof(double));
    double *hidden_errors = (double *)malloc(mlp->hidden_size * sizeof(double));

    for (int k = 0; k < mlp->output_size; k++)
    {
        output_errors[k] = target[k] - mlp->output[k];
    }

    for (int j = 0; j < mlp->hidden_size; j++)
    {
        double error = 0.0;
        for (int k = 0; k < mlp->output_size; k++)
        {
            error += output_errors[k] * mlp->weights_hidden_output[j * mlp->output_size + k];
        }
        hidden_errors[j] = error * sigmoid_derivative(mlp->hidden[j]);
    }

    for (int j = 0; j < mlp->hidden_size; j++)
    {
        for (int k = 0; k < mlp->output_size; k++)
        {
            mlp->weights_hidden_output[j * mlp->output_size + k] += learning_rate * output_errors[k] * mlp->hidden[j];
        }
    }

    for (int k = 0; k < mlp->output_size; k++)
    {
        mlp->bias_output[k] += learning_rate * output_errors[k];
    }

    for (int i = 0; i < mlp->input_size; i++)
    {
        for (int j = 0; j < mlp->hidden_size; j++)
        {
            mlp->weights_input_hidden[i * mlp->hidden_size + j] += learning_rate * hidden_errors[j] * mlp->input[i];
        }
    }

    for (int j = 0; j < mlp->hidden_size; j++)
    {
        mlp->bias_hidden[j] += learning_rate * hidden_errors[j];
    }

    free(output_errors);
    free(hidden_errors);
}

double compute_loss(double *output, double *target, int size)
{
    double loss = 0.0;
    for (int i = 0; i < size; i++)
    {
        loss += 0.5 * (target[i] - output[i]) * (target[i] - output[i]);
    }
    return loss;
}

void train(MLP *mlp, DataItem *train_data, int train_size, int epochs, double learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;
        for (int i = 0; i < train_size; i++)
        {
            forward_propagation(mlp, train_data[i].image);
            total_loss += compute_loss(mlp->output, train_data[i].label, mlp->output_size);
            backpropagation(mlp, train_data[i].label, learning_rate);
        }
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / train_size);
    }
}

double evaluate(MLP *mlp, DataItem *test_data, int test_size)
{
    int correct_predictions = 0;
    for (int i = 0; i < test_size; i++)
    {
        forward_propagation(mlp, test_data[i].image);

        int predicted_label = 0;
        int actual_label = 0;

        for (int j = 0; j < mlp->output_size; j++)
        {
            if (mlp->output[j] > mlp->output[predicted_label])
            {
                predicted_label = j;
            }
            if (test_data[i].label[j] > 0)
            {
                actual_label = j;
            }
        }

        if (predicted_label == actual_label)
        {
            correct_predictions++;
        }
    }
    return (double)correct_predictions / test_size;
}
