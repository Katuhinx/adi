#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

#include "adi.h"

double second();

DATA_TYPE **X;
DATA_TYPE **A;
DATA_TYPE **B;

static void init_array()
{
    int i, j;
    
    X = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));
    A = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));
    B = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));

    for (int i = 0; i < N; i++)
    {
        X[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        A[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        B[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / N;
            B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / N;
        }
}

static void print_row(int row)
{
    row--;

    printf("Row %d:\n", row);

    for (int i = 0; i < N; i++)
        printf("%f \n", X[row][i]);
}

static void kernel_adi(DATA_TYPE **X, DATA_TYPE **A, DATA_TYPE **B)
{
    int t, i1, i2;

    for (t = 0; t < TSTEPS; t++)
    {
        // Горизонтальные обновления
        for (i1 = 0; i1 < N; i1++)
            for (i2 = 1; i2 < N; i2++)
            {
                X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
            }

        // Обновление X[i1][n-1] для локальных строк
        for (i1 = 0; i1 < N; i1++)
            X[i1][N - 1] = X[i1][N - 1] / B[i1][N - 1];

        // Обратная подстановка для локальных строк
        for (i1 = 0; i1 < N; i1++)
            for (i2 = 0; i2 < N - 2; i2++)
                X[i1][N - i2 - 2] = (X[i1][N - 2 - i2] - X[i1][N - 2 - i2 - 1] * A[i1][N - i2 - 3]) / B[i1][N - 3 - i2];

        // Вертикальные обновления
        for (i1 = 1; i1 < N; i1++)
            for (i2 = 0; i2 < N; i2++) {
                X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
            }

        // Обновление последней строки
        for (i2 = 0; i2 < N; i2++)
             X[N-1][i2] = X[N-1][i2] / B[N-1][i2];

        // Обратная подстановка для вертикальных обновлений
        for (i1 = 0; i1 < N-2; i1++)
            for (i2 = 0; i2 < N; i2++)
                X[N-2-i1][i2] = (X[N-2-i1][i2] - X[N-i1-3][i2] * A[N-3-i1][i2]) / B[N-2-i1][i2];
    }
}

int main(int argc, char **argv)
{

    double time0, time1;

    init_array();

    time0 = second();
    kernel_adi(X, A, B);
    time1 = second();

    printf("\nn=%d\n", N);
    printf("\n\n\ntime=%f\n", time1 - time0);
    // Добавлен вывод общего времени выполнения
    printf("Total execution time: %f seconds\n", time1 - time0);

    //print_row(31);
    return 0;
}

double second()
{
    struct timeval tm;
    double t;

    static int base_sec = 0, base_usec = 0;

    gettimeofday(&tm, NULL);

    if (base_sec == 0 && base_usec == 0)
    {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    }
    else
    {
        t = (double)(tm.tv_sec - base_sec) + (double)(tm.tv_usec - base_usec) / 1.0e6;
    }

    return t;
}
