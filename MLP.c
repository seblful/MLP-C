#include "MLP.h"

double tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanh_deriv(double x)
{
    double t = tanh(x);
    return 1.0 - (t * t);
}

void softmax(double *output, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    for (int i = 0; i < size; i++)
    {
        output[i] /= sum;
    }
}

MLP *initialize_network(int input_size, int hidden_size, int output_size)
{
    MLP *network = (MLP *)malloc(sizeof(MLP));
    network->input_size = input_size;
    network->hidden_size = hidden_size;
    network->output_size = output_size;

    network->hidden = (double *)calloc(hidden_size, sizeof(double));
    network->output = (double *)calloc(output_size, sizeof(double));

    network->hidden_bias = (double *)calloc(hidden_size, sizeof(double));
    network->output_bias = (double *)calloc(output_size, sizeof(double));

    network->input_hidden_weights = (double **)malloc(input_size * sizeof(double *));
    for (int i = 0; i < input_size; i++)
    {
        network->input_hidden_weights[i] = (double *)calloc(hidden_size, sizeof(double));
    }

    network->output_hidden_weights = (double **)malloc(hidden_size * sizeof(double *));
    for (int i = 0; i < hidden_size; i++)
    {
        network->output_hidden_weights[i] = (double *)calloc(output_size, sizeof(double));
    }

    network->hidden_delta = (double *)calloc(hidden_size, sizeof(double));
    network->output_delta = (double *)calloc(output_size, sizeof(double));

    // Initialize weights and biases with small random values
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < hidden_size; j++)
        {
            network->input_hidden_weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    for (int i = 0; i < hidden_size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            network->output_hidden_weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    for (int i = 0; i < hidden_size; i++)
    {
        network->hidden_bias[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < output_size; i++)
    {
        network->output_bias[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    return network;
}

void forward_propagation(MLP *network, double *input)
{

    for (int i = 0; i < network->hidden_size; i++)
    {
        network->hidden[i] = 0;
        for (int j = 0; j < network->input_size; j++)
        {
            network->hidden[i] += input[j] * network->input_hidden_weights[j][i];
        }
        network->hidden[i] += network->hidden_bias[i];
        network->hidden[i] = tanh(network->hidden[i]);
    }

    for (int i = 0; i < network->output_size; i++)
    {
        network->output[i] = 0;
        for (int j = 0; j < network->hidden_size; j++)
        {
            network->output[i] += network->hidden[j] * network->output_hidden_weights[j][i];
        }
        network->output[i] += network->output_bias[i];
    }

    softmax(network->output, network->output_size);
}

void backpropagation(MLP *network, double *input, double *target, double learning_rate)
{

    // Calculate output layer delta
    for (int i = 0; i < network->output_size; i++)
    {
        network->output_delta[i] = (network->output[i] - target[i]);
    }

    // Calculate hidden layer delta
    for (int i = 0; i < network->hidden_size; i++)
    {
        network->hidden_delta[i] = 0;

        for (int j = 0; j < network->output_size; j++)
        {
            network->hidden_delta[i] += network->output_delta[j] * network->output_hidden_weights[i][j];
        }
        network->hidden_delta[i] *= tanh_deriv(network->hidden[i]);
    }

    // Update weights and biases
    update_weights(network, input, learning_rate);
}

void update_weights(MLP *network, double *input, double learning_rate)
{
    // Update hidden-output weights and biases
    for (int i = 0; i < network->hidden_size; i++)
    {
        for (int j = 0; j < network->output_size; j++)
        {
            network->output_hidden_weights[i][j] -= learning_rate * network->output_delta[j] * network->hidden[i];
        }
    }

    for (int i = 0; i < network->output_size; i++)
    {
        network->output_bias[i] -= learning_rate * network->output_delta[i];
    }

    // Update input-hidden weights and biases
    for (int i = 0; i < network->input_size; i++)
    {
        for (int j = 0; j < network->hidden_size; j++)
        {
            network->input_hidden_weights[i][j] -= learning_rate * network->hidden_delta[j] * input[i];
        }
    }

    for (int i = 0; i < network->hidden_size; i++)
    {
        network->hidden_bias[i] -= learning_rate * network->hidden_delta[i];
    }
}

double calculate_error(double *output, double *target, int size)
{
    double error = 0;
    for (int i = 0; i < size; i++)
    {
        error += 0.5 * (target[i] - output[i]) * (target[i] - output[i]);
    }
    return error;
}

void train(MLP *network, DataItem *train_data, int train_size, int epochs, double learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_error = 0;
        for (int i = 0; i < train_size; i++)
        {
            forward_propagation(network, train_data[i].image);
            backpropagation(network, train_data[i].image, train_data[i].label, learning_rate);
            total_error += calculate_error(network->output, train_data[i].label, network->output_size);
        }
        printf("Epoch %d, Error: %f\n", epoch, total_error);
    }
}

double evaluate(MLP *network, DataItem *test_data, int test_size)
{
    int correct_predictions = 0;
    for (int i = 0; i < test_size; i++)
    {
        forward_propagation(network, test_data[i].image);
        int predicted_label = 0;
        int actual_label = 0;
        for (int j = 0; j < network->output_size; j++)
        {
            if (network->output[j] > network->output[predicted_label])
            {
                predicted_label = j;
            }
            if (test_data[i].label[j] == 1.0)
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

void free_network(MLP *network)
{
    free(network->hidden);
    free(network->output);
    free(network->hidden_bias);
    free(network->output_bias);

    for (int i = 0; i < network->input_size; i++)
    {
        free(network->input_hidden_weights[i]);
    }
    free(network->input_hidden_weights);

    for (int i = 0; i < network->hidden_size; i++)
    {
        free(network->output_hidden_weights[i]);
    }
    free(network->output_hidden_weights);

    free(network->hidden_delta);
    free(network->output_delta);
    free(network);
}
