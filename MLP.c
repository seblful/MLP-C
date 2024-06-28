#include <stdio.h>
#include <stdlib.h>

#include "MLP.h"

double *relu() {};

double *maximum() {};

double *sigmoid() {};

double **matmul() {};

void printMatrix(size_t nRows, size_t nCols, double *matrix)
{

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            printf("%f", *matrix[i][j]);
        }
        printf("\n");
    }
};