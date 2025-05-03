#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "adi.h"
#include "reportlib/reportlib.h"

DATA_TYPE **X;
DATA_TYPE **A;
DATA_TYPE **B;
int threads = 1;

static void init_arrays()
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
    {
        for (j = 0; j < N; j++)
        {
            X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / N;
            B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / N;
        }
    }
}

// Функция освобождения памяти массивов X, A и B
static void free_arrays()
{
    for (int i = 0; i < N; i++)
    {
        free(X[i]);
        free(A[i]);
        free(B[i]);
    }

    free(X);
    free(A);
    free(B);
}

static void print_row(int row)
{
    row--;

    #pragma omp target update from(X[row][0 : N])
    printf("Row %d:\n", row);

    for (int i = 0; i < N; i++)
        printf("%f \n", X[row][i]);
}

static void kernel_adi()
{
    int t, i1, i2;
    
    #pragma omp target enter data map(to : A[0:N][ 0: N],  X[0:N][ 0: N], B[0:N][0 : N]) 

    for (t = 0; t < TSTEPS; t++)
    {
        // Горизонтальные обновления
        #pragma omp target teams distribute parallel for simd num_threads(threads)//num_teams(0) thread_limit(num_threads)
        for (i1 = 0; i1 < N; i1++)
        {
            for (i2 = 1; i2 < N; i2++)
            {
                X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
            }

            X[i1][N - 1] /= B[i1][N - 1];
        }

        // Обратная подстановка для локальных строк
        #pragma omp target teams distribute parallel for simd num_threads(threads) //num_teams(0) thread_limit(num_threads)
        for (i1 = 0; i1 < N; i1++)
            for (i2 = 0; i2 < N - 2; i2++)
                X[i1][N - i2 - 2] = (X[i1][N - 2 - i2] - X[i1][N - 2 - i2 - 1] * A[i1][N - i2 - 3]) / B[i1][N - 3 - i2];

        // Вертикальные обновления
        #pragma omp target teams //num_teams(0) thread_limit(num_threads)
        for (i1 = 1; i1 < N; i1++)
        {
            #pragma omp distribute parallel for simd 
            for (i2 = 0; i2 < N; i2++)
            {
                X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
            }
        }

        // Обновление последней строки
        #pragma omp target teams distribute parallel for simd num_threads(threads) //num_teams(0) thread_limit(num_threads)
        for (i2 = 0; i2 < N; i2++)
            X[N - 1][i2] = X[N - 1][i2] / B[N - 1][i2];

        // Обратная подстановка для вертикальных обновлений
        #pragma omp target teams//num_teams(0) thread_limit(num_threads)
        for (i1 = 0; i1 < N-2; i1++)
        {
            #pragma omp distribute parallel for simd 
            for (i2 = 0; i2 < N; i2++)
                X[N - 2 - i1][i2] = (X[N - 2 - i1][i2] - X[N - i1 - 3][i2] * A[N - 3 - i1][i2]) / B[N - 2 - i1][i2];
        }
    }
}

int main(int argc, char **argv)
{
    const char args_string[256];
    double time0, time1;

    if (argc > 1)
    {
        threads = atoi(argv[1]);
    } 
    else
    {
        threads = omp_get_max_threads();
    }

    init_arrays();

    sprintf(args_string, "%d", threads);

    time0 = omp_get_wtime();
    kernel_adi();
    time1 =  omp_get_wtime();

    printf("\nN: %d", N);
    printf("\nNumber of theards: %d", threads);
    printf("\nTotal execution time: %f seconds\n", time1 - time0);

    // print_row(31);

//    report_result("adi_omp", args_string, time1 - time0);

    free_arrays();
}
