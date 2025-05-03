#ifndef ADI_H
# define ADI_H

/* Default to STANDARD_DATASET. */

# if !defined(USE_SCALAR)//используются ли скалярные значения или массивы в программе
#  define EXTRALARGE_DATASET
#endif

# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)// размер массивов
#  define MINI_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(TSTEPS) && ! defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define TSTEPS 2//количество шагов времени, используемых в методе
#   define N 32//размер массивов
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 10
#   define N 512
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define TSTEPS 50
#   define N 1024
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 50
#   define N 2048
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 10
#   define N 20000
#  endif
# endif /* !N */

# ifdef USE_SCALAR
# define _PB_N n//тип, который определяет размер массива n
# define _PB_TSTEPS tsteps//тип, который определяет количество шагов времени
#else
# define _PB_N N
# define _PB_TSTEPS TSTEPS
# endif

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif


#endif /* !ADI */


