#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "adi.h"
//#include "reportlib/reportlib.h"

double second();

DATA_TYPE **X;
DATA_TYPE **A;
DATA_TYPE **B;

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

    #pragma acc enter data create(X[0 : N][0 : N], A[0 : N][0 : N], B[0 : N][0 : N])
    #pragma acc parallel loop collapse(2) present(X, A, B)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
        {
            X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / N;
            B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / N;
        }
    }
}

static void free_arrays()
{
    #pragma acc exit data delete (X[0 : N][0 : N], A[0 : N][0 : N], B[0 : N][0 : N])
}

static void print_row(int row)
{
    row--;

    #pragma acc update host(X[row][0 : N])
    printf("Row %d:\n", row);

    for (int i = 0; i < N; i++)
        printf("%f \n", X[row][i]);
}

static void kernel_adi()
{
    int t, i1, i2;
    
    #pragma acc data present(A, B, X)
    for (t = 0; t < TSTEPS; t++)
    {
        // Горизонтальные обновления
        #pragma acc parallel //program runtime 62.28%  
        {
            #pragma acc loop gang vector 
            for (i1 = 0; i1 < N; i1++)
            {  
                #pragma acc loop seq
                for (i2 = 1; i2 < N; i2++)
                {
                    int idx = i2 - 1;

                    X[i1][i2] = X[i1][i2] - X[i1][idx] * A[i1][i2] / B[i1][idx];
                    B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][idx];
                } 
            }
        }

        #pragma acc parallel loop gang vector
        for (i1 = 0; i1 < N; i1++)
        {
            X[i1][N - 1] /= B[i1][N - 1];
        }
        
        // Обратная подстановка для локальных строк
        #pragma acc parallel //program runtime 28.80%
        {
            #pragma acc loop gang vector
            for (i1 = 0; i1 < N; i1++)
            {
                #pragma acc loop seq
                for (i2 = 0; i2 < N - 2; i2++)
                {
                    int idx = N - i2 - 2;
                    X[i1][idx] = (X[i1][idx] - X[i1][idx - 1] * A[i1][idx - 1]) / B[i1][idx - 1];
                }
            }
        }

        // Вертикальные обновления
        #pragma acc parallel 
        {
            #pragma acc loop seq
            for (i1 = 1; i1 < N; i1++) {
                int idx = i1 - 1;

                #pragma acc loop gang vector
                for (i2 = 0; i2 < N; i2++)
                {
                    X[i1][i2] = X[i1][i2] - X[idx][i2] * A[i1][i2] / B[idx][i2];
                    B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[idx][i2];
                }
            }
        }    
        
        #pragma acc parallel loop gang vector
        for (i2 = 0; i2 < N; i2++)
        {
            X[N - 1][i2] /= B[N - 1][i2];
        }

        // Обратная подстановка для вертикальных обновлений
        #pragma acc parallel 
        {
            #pragma acc loop seq
            for (i1 = 0; i1 < N - 2; i1++) {
                int idx = N - i1 - 2;

                #pragma acc loop gang vector
                for (i2 = 0; i2 < N; i2++) {
                    X[idx][i2] = (X[idx][i2] - X[idx - 1][i2] * A[idx - 1][i2]) / B[idx][i2];
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    double time0, time1;

    init_arrays();

    time0 = second();
    kernel_adi();
    time1 = second();

    printf("\nN: %d", N);
    printf("\nTotal execution time: %f seconds\n", time1 - time0);

    //report_result("adi_oacc", "", time1 - time0);

    // print_row(32);
    
    free_arrays();    

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
