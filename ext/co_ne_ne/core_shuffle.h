// ext/co_ne_ne/core_shuffle.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CORE_QSORT_H
#define CORE_QSORT_H

#include "core_mt.h"

void shuffle_ints( int n, int *array );

void quicksort_ints_by_floats( int *sortable, float *sort_by, int lowest, int highest );

#endif
