#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <openacc.h>
#include "adi.h"
#include "reportlib/reportlib.h"

DATA_TYPE **X;
DATA_TYPE **A;
DATA_TYPE **B;
int rank, size;
int local_rows = N;
int num_quanta = 1;
int quantum_size = N; 

static int calculate_optimal_quanta()
{
    const int cache_line_size = 64;             // Размер кэш-линии в байтах
    const int element_size = sizeof(DATA_TYPE); // Размер одного элемента в байтах

    int elements_per_cache_line = cache_line_size / element_size; // Рассчитаем количество элементов, которое помещается в одну кэш-линию
    quantum_size = elements_per_cache_line > 0 ? elements_per_cache_line : 1; // Убедимся, что размер кванта не меньше одного элемента
    int optimal_quanta = local_rows/ quantum_size; // Рассчитаем количество квантов
    return optimal_quanta > 0 ? optimal_quanta : 1; // Убедимся, что количество квантов не меньше 1
}

// Функция инициализации массивов X, A и B
static void init_arrays()
{
    int i, j;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    X = (DATA_TYPE **)malloc((local_rows + 2) * sizeof(DATA_TYPE *));
    A = (DATA_TYPE **)malloc((local_rows + 2) * sizeof(DATA_TYPE *));
    B = (DATA_TYPE **)malloc((local_rows + 2) * sizeof(DATA_TYPE *));

    for (int i = 0; i < local_rows + 2; i++)
    {
        X[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        A[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
        B[i] = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
    }

    // #pragma acc parallel loop collapse(2) present(X[0:local_rows+2][0:N], A[0:local_rows+2][0:N], B[0:local_rows+2][0:N])
    for (i = 0; i < local_rows; i++)
    {
        for (j = 0; j < N; j++)
        {
            X[i + 1][j] = ((DATA_TYPE)(rank * local_rows + i) * (j + 1) + 1) / N;
            A[i + 1][j] = ((DATA_TYPE)(rank * local_rows + i) * (j + 2) + 2) / N;
            B[i + 1][j] = ((DATA_TYPE)(rank * local_rows + i) * (j + 3) + 3) / N;
        }
    }
    
    // Передача указателей верхнего уровня на устройство
    #pragma acc enter data create(X[0 : local_rows + 2], A[0 : local_rows + 2], B[0 : local_rows + 2])
    // Передача данных для каждого подмассива
    for (int i = 0; i < local_rows + 2; i++)
    {
        #pragma acc enter data create(X[i][0 : N], A[i][0 : N], B[i][0 : N])
        #pragma acc update device(X[i][0 : N], A[i][0 : N], B[i][0 : N])
    }
}

// Функция освобождения памяти массивов X, A и B
static void free_arrays()
{
    for (int i = 0; i < local_rows + 2; i++)
    {
        #pragma acc exit data delete (X[i][0 : N], A[i][0 : N], B[i][0 : N])
    }

    #pragma acc exit data delete (X[0 : local_rows + 2], A[0 : local_rows + 2], B[0 : local_rows + 2])

    for (int i = 0; i < local_rows + 2; i++)
    {
        free(X[i]);
        free(A[i]);
        free(B[i]);
    }

    free(X);
    free(A);
    free(B);
}

// Функция вывода результирующих значений из строки row массива X
static void print_row(int row)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    row--;

    int valid_rank = row / local_rows;
    row = row % local_rows;

    #pragma acc update host(X[row + 1][0 : N])
    if (rank == valid_rank)
    {
        printf("Rank %d, row %d:\n", valid_rank, row);

        for (int i = 0; i < N; i++)
            printf("%f \n", X[row + 1][i]);
    }
}

static void kernel_adi()
{
    int t, i1, i2, q;
    MPI_Status statuses[2];
    MPI_Request requests[2];

    for (t = 0; t < TSTEPS; t++)
    {
        // Горизонтальные обновления
        #pragma acc parallel loop gang vector present(X[0 : local_rows + 2][0 : N], A[0 : local_rows + 2][0 : N], B[0 : local_rows + 2][0 : N])
        for (i1 = 1; i1 < local_rows + 1; i1++)
        {
            #pragma acc loop seq
            for (i2 = 1; i2 < N; i2++)
            {
                X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
            }
            
            X[i1][N - 1] = X[i1][N - 1] / B[i1][N - 1];
        }

        // Обратная подстановка для локальных строк
        #pragma acc parallel loop gang vector present(X[0 : local_rows + 2][0 : N], A[0 : local_rows + 2][0 : N], B[0 : local_rows + 2][0 : N])
        for (int i1 = 1; i1 < local_rows + 1; i1++)
        {
            #pragma acc loop seq
            for (i2 = 0; i2 < N - 2; i2++)
            {
                int idx = N - i2 - 2;
                X[i1][idx] = (X[i1][idx] - X[i1][idx - 1] * A[i1][idx - 1]) / B[i1][idx - 1];
            }
        }

        // Вертикальные обновления с конвейерной обработкой
        for (q = 0; q < num_quanta; q++)
        {
            int start = q * quantum_size;   // с какого элемента начинается данный квант
            int end = start + quantum_size; // на каком элементе заканчивается данный квант

            end = (end > N) ? N : end; // ограничение квантов
            int current_quantum_size = end - start;

            // Получение данных от предыдущего процесса
            if (rank > 0)
            {
                MPI_Irecv(&X[0][start], current_quantum_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(&B[0][start], current_quantum_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[1]);
                MPI_Waitall(2, requests, statuses);
                #pragma acc update device(X[0][start : end - start], B[0][start : end - start])
            }

            // Обработка текущего кванта
            #pragma acc parallel present(X[0 : local_rows + 2][0 : N], A[0 : local_rows + 2][0 : N], B[0 : local_rows + 2][0 : N])
            {
                #pragma acc loop seq
                for (i1 = (rank == 0 ? 2 : 1); i1 < local_rows + 1; i1++)
                {
                    #pragma acc loop gang vector
                    for (i2 = start; i2 < end; i2++)
                    {
                        X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
                        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
                    }
                }
            }

            // Отправка данных следующему процессу
            if (rank < size - 1)
            {
                #pragma acc update host(X[local_rows][start : end - start], B[local_rows][start : end - start])
                MPI_Isend(&X[local_rows][start], current_quantum_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[0]);
                MPI_Isend(&B[local_rows][start], current_quantum_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[1]);
                MPI_Waitall(2, requests, statuses);
            }
        }

        // Обновление последней строки
        if (rank == size - 1)
        {
            #pragma acc parallel loop gang vector present(X[0 : local_rows + 2][0 : N], A[0 : local_rows + 2][0 : N], B[0 : local_rows + 2][0 : N])
            for (i2 = 0; i2 < N; i2++)
            {
                X[local_rows][i2] = X[local_rows][i2] / B[local_rows][i2];
            }
        }

        // Обратная подстановка для вертикальных обновлений с обменом данными
        if (rank > 0)
        {
            MPI_Irecv(X[0], N, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(A[0], N, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, statuses);
            #pragma acc update device(X[0][0 : N], A[0][0 : N])
        }

        if (rank < size - 1)
        {
            #pragma acc update host(X[local_rows][0 : N], A[local_rows][0 : N])
            MPI_Isend(X[local_rows], N, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(A[local_rows], N, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, statuses);
        }

        #pragma acc parallel present(X[0 : local_rows + 2][0 : N], A[0 : local_rows + 2][0 : N], B[0 : local_rows + 2][0 : N])
        {
            #pragma acc loop seq
            for (i1 = local_rows - (rank == (size - 1) ? 1 : 0); i1 >= (rank == 0 ? 2 : 1); i1--)
            {
                #pragma acc loop gang vector
                for (i2 = 0; i2 < N; i2++)
                {
                    X[i1][i2] = (X[i1][i2] - X[i1 - 1][i2] * A[i1 - 1][i2]) / B[i1][i2];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{ 
    const char args_string[256];
    double time0, time1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    acc_get_num_devices(acc_device_nvidia);
    acc_set_device_num(rank % acc_get_num_devices(acc_device_nvidia), acc_device_nvidia);


    if (argc > 1)
    {
        num_quanta = atoi(argv[1]);
    }
    else
    {        
        if (size > 1)  
        {
            num_quanta = calculate_optimal_quanta();
        }
    }
    
    num_quanta = num_quanta < 1 ? 1 : num_quanta;
    num_quanta = num_quanta > N ? N : num_quanta;

    sprintf(args_string, "%d %d", size, num_quanta);

    local_rows = (N / size) + (rank < (N % size) ? 1 : 0); // количество элементов на 1 процессе
    quantum_size = N / num_quanta;                         // количество элементов на 1 кванте
    num_quanta = N / quantum_size;                         // количество квантов пересчитаное с учетом размера кванта
    
    init_arrays();
    
    // Замер времени выполнения
    time0 = MPI_Wtime();
    kernel_adi(TSTEPS, N, X, A, B, rank, size, local_rows);
    time1 = MPI_Wtime();
    
    if (rank == 0)
    {
        printf("\nN: %d", N);
        printf("\nQuantum size: %d", quantum_size);
        printf("\nNumber of quanta: %d", num_quanta);
        printf("\nTotal execution time: %f seconds\n", time1 - time0);
        //report_result("adi_mpi_acc", args_string, time1 - time0);
    }

    
    
    //print_row(32);
    free_arrays();
    MPI_Finalize();
    
    return 0;
}
