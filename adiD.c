#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#include "adi.h"

int num_stages = 1;

#pragma dvm array distribute[block][block]
DATA_TYPE A[N][N];

#pragma dvm array align([i][j] with A[i][j]) shadow[1 : 0][1 : 0]
DATA_TYPE X[N][N];

#pragma dvm array align([i][j] with A[i][j]) shadow[1 : 0][1 : 0]
DATA_TYPE B[N][N];

static void init_array()
{
  int i, j;

#pragma dvm parallel([i][j] on A[i][j]) 
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

  const char header[] = "Row %d:\n";
  char header_buf[32];
  int header_len = snprintf(header_buf, sizeof(header_buf), header, row);
  fwrite(header_buf, 1, header_len, stdout);

#pragma dvm get_actual(X[0 : N][0 : N])

  for (int i = 0; i < N; i++)
  {
#pragma dvm remote_access(X[row][i])
    {
      char num_buf[32];
      int num_len = snprintf(num_buf, sizeof(num_buf), "%f\n", X[row][i]);
      fwrite(num_buf, 1, num_len, stdout);
    }
  }
}

static void kernel_adi()
{
    int t, i1, i2;
    #pragma dvm actual(X, A, B)
    
    {
        #pragma dvm region
        {
            #pragma dvm parallel([i1][i2] on A[i1][i2]) shadow_renew(A)
            for (i1 = 0; i1 < N; i1++)
                for (i2 = 0; i2 < N; i2++)
                {}
        }
    }

    for (t = 0; t < TSTEPS; t++)
    {
            #pragma dvm region
            {
                #pragma dvm parallel([i1][i2] on X[i1][i2]) across(X[0 : 0][1 : 0], B[0 : 0][1 : 0])
                for (i1 = 0; i1 < N; i1++) 
                {
                    for (i2 = 1; i2 < N; i2++)
                    {
                        X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
                        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
                    }
                }
            }

            #pragma dvm region
            {
                #pragma dvm parallel([i1] on X[i1][N - 1])
                for (i1 = 0; i1 < N; i1++)
                    X[i1][N - 1] /= B[i1][N - 1];
            }

            #pragma dvm region
            {
                #pragma dvm parallel([i1][i2] on X[i1][i2]) shadow_renew(B) across(X[0 : 0][1 : 0]) //stage(num_stages)
                for (i1 = 0; i1 < N; i1++)
                    for (i2 = N - 2; i2 > 0; i2--)
                        X[i1][i2] = (X[i1][i2] - X[i1][i2 - 1] * A[i1][i2 - 1]) / B[i1][i2 - 1];
            }

            #pragma dvm region
            {
                #pragma dvm parallel([i1][i2] on X[i1][i2]) across(X[1 : 0][0 : 0], B[1 : 0][0 : 0]) //stage(num_stages)
                for (i1 = 1; i1 < N; i1++)
                    for (i2 = 0; i2 < N; i2++)
                    {
                        X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
                        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
                    }
            }

            #pragma dvm region
            {
                #pragma dvm parallel([i2] on X[N - 1][i2])
                    for (i2 = 0; i2 < N; i2++)
                        X[N - 1][i2] = X[N - 1][i2] / B[N - 1][i2];
            }

            #pragma dvm region
            {
                #pragma dvm parallel([i1][i2] on X[i1][i2]) across(X[1 : 0][0 : 0]) //stage(num_stages)
                for (i1 = N - 2; i1 > 0; i1--)
                    for (i2 = 0; i2 < N; i2++)
                        X[i1][i2] = (X[i1][i2] - X[i1 - 1][i2] * A[i1 - 1][i2]) / B[i1][i2];
            }
    }
}

int main(int argc, char **argv)
{
  const char args_string[256];
  double time0, time1;

  if (argc > 1)
  {
    num_stages = atoi(argv[1]);
  }
  
  init_array();

  time0 = dvmh_wtime();
  kernel_adi();
  time1 = dvmh_wtime();

  printf("\nN: %d", N);
  printf("\nStages: %d", num_stages);
  printf("\nTotal execution time: %f seconds\n", time1 - time0);

  return 0;
}