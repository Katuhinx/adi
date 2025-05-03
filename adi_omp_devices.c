#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>
#include "adi.h"
//#include "reportlib/reportlib.h"

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

    for (int i = 0; i < N; i++) {
        X[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        A[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        B[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / N;
            B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / N;
        }
    }
}

static void free_arrays()
{
    for (int i = 0; i < N; i++) {
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
    int num_devices = omp_get_num_devices();
    int chunk = N / num_devices;

    // Загрузка данных на устройства
    for (int dev = 0; dev < num_devices; dev++) {
        int start = dev * chunk;
        int count = (dev == num_devices - 1) ? (N - start) : chunk;
        #pragma omp target enter data map(to: A[start:count][0:N], B[start:count][0:N], X[start:count][0:N]) device(dev)
    }

    // for (t = 0; t < TSTEPS; t++) {
    //     // Горизонтальные обновления
    //     #pragma omp parallel for num_threads(num_devices)
    //     for (int dev = 0; dev < num_devices; dev++) {
    //         int start = dev * chunk;
    //         int end = (dev == num_devices - 1) ? N : start + chunk;
    //         int count = end - start;
    //         #pragma omp target teams distribute parallel for simd device(dev) nowait
    //         for (i1 = start; i1 < end; i1++) {
    //             for (i2 = 1; i2 < N; i2++) {
    //                 X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
    //                 B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
    //             }
    //             X[i1][N - 1] /= B[i1][N - 1];
    //         }
    //     }

        // // Обратная подстановка для локальных строк
        // #pragma omp parallel for num_threads(num_devices)
        // for (int dev = 0; dev < num_devices; dev++) {
        //     int start = dev * chunk;
        //     int end = (dev == num_devices - 1) ? N : start + chunk;
        //     int count = end - start;
        //     #pragma omp target teams distribute parallel for simd device(dev) nowait
        //     for (i1 = start; i1 < end; i1++) {
        //         for (i2 = 0; i2 < N - 2; i2++) {
        //             X[i1][N - i2 - 2] = (X[i1][N - 2 - i2] - X[i1][N - 2 - i2 - 1] * A[i1][N - i2 - 3]) / B[i1][N - 3 - i2];
        //         }
        //     }
        // }

        // // Синхронизация граничных строк между устройствами
        // for (int dev = 1; dev < num_devices; dev++) {
        //     int prev = dev - 1;
        //     int src_row = dev * chunk - 1;
        //     #pragma omp target update from(X[src_row][0:N], B[src_row][0:N]) device(prev)
        //     #pragma omp target update to(X[src_row][0:N], B[src_row][0:N]) device(dev)
        // }

        // // Вертикальные обновления (пропускаем первую строку каждого блока)
        // #pragma omp parallel for num_threads(num_devices)
        // for (int dev = 0; dev < num_devices; dev++) {
        //     int start = dev * chunk;
        //     int end = (dev == num_devices - 1) ? N : start + chunk;
        //     int count = end - start;

        //     int local_start = (start == 0) ? start + 1 : start;

        //     #pragma omp target teams device(dev) nowait
        //     for (i1 = local_start; i1 < end; i1++) {
        //         #pragma omp distribute parallel for simd
        //         for (i2 = 0; i2 < N; i2++) {
        //             X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
        //             B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
        //         }
        //     }
        // }

        // // Обновление последней строки
        // #pragma omp target teams distribute parallel for simd device(num_devices - 1)
        // for (i2 = 0; i2 < N; i2++)
        //     X[N - 1][i2] = X[N - 1][i2] / B[N - 1][i2];

        // // Обратная подстановка для вертикальных обновлений
        // #pragma omp parallel for num_threads(num_devices)
        // for (int dev = 0; dev < num_devices; dev++) {
        //     int start = dev * chunk;
        //     int end = (dev == num_devices - 1) ? N : start + chunk;
        //     int count = end - start;

        //     #pragma omp target teams device(dev) nowait
        //     for (i1 = end - 2; i1 >= start + 1; i1--) {
        //         #pragma omp distribute parallel for simd
        //         for (i2 = 0; i2 < N; i2++) {
        //             X[i1][i2] = (X[i1][i2] - X[i1 - 1][i2] * A[i1 - 1][i2]) / B[i1][i2];
        //         }
        //     }
        // }
    }

    // Обновление данных на хост и освобождение
    for (int dev = 0; dev < num_devices; dev++) {
        int start = dev * chunk;
        int count = (dev == num_devices - 1) ? (N - start) : chunk;
        #pragma omp target update from(X[start:count][0:N]) device(dev)
        #pragma omp target exit data map(delete: A[start:count][0:N], B[start:count][0:N], X[start:count][0:N]) device(dev)
    }
}

int main(int argc, char **argv)
{
    const char args_string[256];
    double time0, time1;

    if (argc > 1) {
        threads = atoi(argv[1]);
    } else {
        threads = omp_get_max_threads();
    }

    init_arrays();

    sprintf(args_string, "%d", threads);

    int num_devices = omp_get_num_devices();
    printf("Detected %d GPU devices\n", num_devices);

    time0 = omp_get_wtime();
    kernel_adi();
    time1 = omp_get_wtime();

    printf("\nN: %d", N);
    printf("\nNumber of threads: %d", threads);
    printf("\nTotal execution time: %f seconds\n", time1 - time0);

    free_arrays();
    return 0;
}


// #include <stdio.h>
// #include <stdlib.h>
// #include <stdbool.h>
// #include <sys/time.h>
// #include <omp.h>
// #include "adi.h"

// DATA_TYPE **X;
// DATA_TYPE **A;
// DATA_TYPE **B;
// int num_threads = 1;
// // int num_devices = 1;
// // int device_rows = N;

// static void init_arrays()
// {
//     int i, j;

//     X = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));
//     A = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));
//     B = (DATA_TYPE **)malloc(N * sizeof(DATA_TYPE *));

//     for (int i = 0; i < N; i++)
//     {
//         X[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
//         A[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
//         B[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
//     }

//     for (i = 0; i < N; i++)
//     {
//         for (j = 0; j < N; j++)
//         {
//             X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
//             A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / N;
//             B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / N;
//         }
//     }

//     // #pragma omp parallel num_threads(num_devices)
//     // {
//     //     int device_id = omp_get_thread_num();
//     //     int device_start = device_id * device_rows;
//     //     int device_length = (device_id == num_devices - 1) ? N - device_start : device_rows;
        
//     //     #pragma omp target enter data map(to: A[device_start : device_length][ : N], X[device_start : device_length][ : N], B[device_start : device_length][ : N]) device(device_id) 
//     //     {}
//     // }   
// }

// // Функция освобождения памяти массивов X, A и B
// static void free_arrays()
// {
//     for (int i = 0; i < N; i++)
//     {
//         free(X[i]);
//         free(A[i]);
//         free(B[i]);
//     }

//     free(X);
//     free(A);
//     free(B);
// }

// static void print_row(int row)
// {
//     row--;

//     // #pragma omp target update from(X[row][0 : N]) device(1)
//     printf("Row %d:\n", row);

//     for (int i = 0; i < N; i++)
//         printf("%f \n", X[row][i]);
// }

// static void kernel_adi()
// {
//     int t, i1, i2;

//     #pragma omp parallel num_threads(num_devices)
//     {
//         // #pragma omp target enter data map(to : A[0 : N][ : N],  X[0 : N][ : N], B[0 : N][ : N])

//         int device_id = omp_get_thread_num();
//         int device_start = device_id * device_rows;
//         int device_length = (device_id == num_devices - 1) ? N - device_start : device_rows;

//         #pragma omp target device(dev)
//         for (t = 0; t < TSTEPS; t++)
//         {
//             // Горизонтальные обновления
            
//             #pragma omp target teams distribute parallel for simd num_threads(num_threads) device(device_id)
//             for (i1 = device_start; i1 < device_start + device_length; i1++)
//             {
//                 for (i2 = 1; i2 < N; i2++)
//                 {
//                     // X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
//                     // B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];

//                     if (device_id == 1 && i1 == 31) {
//                         printf("%f\n", X[i1][i2]);
//                     }
//                 }

//                 // X[i1][N - 1] /= B[i1][N - 1];
//             }

//             // // Обратная подстановка для локальных строк
//             // #pragma omp target teams distribute parallel for simd num_teams(0) thread_limit(num_threads)
//             // for (i1 = 0; i1 < N; i1++)
//             //     for (i2 = 0; i2 < N - 2; i2++)
//             //         X[i1][N - i2 - 2] = (X[i1][N - 2 - i2] - X[i1][N - 2 - i2 - 1] * A[i1][N - i2 - 3]) / B[i1][N - 3 - i2];

//             // // Вертикальные обновления
//             // #pragma omp target teams num_teams(0) thread_limit(num_threads)
//             // for (i1 = 1; i1 < N; i1++)
//             // {
//             //     #pragma omp distribute parallel for simd //num_teams(0) thread_limit(num_threads)
//             //     for (i2 = 0; i2 < N; i2++)
//             //     {
//             //         X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
//             //         B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
//             //     }
//             // }

//             // // Обновление последней строки
//             // #pragma omp target teams distribute parallel for simd num_teams(0) thread_limit(num_threads)//num_teams(0) thread_limit(num_threads)
//             // for (i2 = 0; i2 < N; i2++)
//             //     X[N - 1][i2] = X[N - 1][i2] / B[N - 1][i2];

//             // // Обратная подстановка для вертикальных обновлений
//             // #pragma omp target teams num_teams(0) thread_limit(num_threads)
//             // for (i1 = 0; i1 < N - 2; i1++)
//             // {
//             //     #pragma omp distribute parallel for simd  //num_teams(0) thread_limit(num_threads)
//             //     for (i2 = 0; i2 < N; i2++)
//             //         X[N - 2 - i1][i2] = (X[N - 2 - i1][i2] - X[N - i1 - 3][i2] * A[N - 3 - i1][i2]) / B[N - 2 - i1][i2];
//             // }
//         }

//         #pragma omp target update from(X[device_start : device_length][ : N]) device(device_id)
//     }
// }

// int main(int argc, char **argv)
// {
//     num_devices = omp_get_num_devices();
//     double time0, time1;

//     if (argc > 1)
//     {
//         num_threads = atoi(argv[1]);
//     } 
//     else
//     {
//         num_threads = omp_get_max_threads();
//     }

//     device_rows = N / num_devices;

//     init_arrays();

//     time0 = omp_get_wtime();
//     kernel_adi();
//     time1 = omp_get_wtime();

//     printf("\nN: %d", N);
//     printf("\nNumber of theards: %d", num_threads);
//     printf("\nNumber of devices: %d", num_devices);
//     printf("\nTotal execution time: %f seconds\n", time1 - time0);

//     // print_row(31);

//     free_arrays();
// }
