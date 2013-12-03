// ext/co_ne_ne/core_qsort.c

#include "core_qsort.h"

void shuffle_ints( int n, int *array ) {
  int i;
  float *scramble = ALLOC_N( float, n );
  for ( i = 0; i < n; i++ ) {
    array[i] = i;
    scramble[i] = genrand_real1();
  }
  quicksort_ints_by_floats( array, scramble, 0, n-1 );
  xfree( scramble );
  return;
}

// Sorts sortable array in situ using sort_by array, assuming sortable is list of ids 0...n
void quicksort_ints_by_floats( int *sortable, double *sort_by, int lowest, int highest ) {
  int pivot, j, i;
  int temp_id;

  if ( lowest < highest ) {
    pivot = lowest;
    i = lowest;
    j = highest;

    // Calculate pivot
    while ( i < j ) {
      while ( sort_by[sortable[i]] <= sort_by[sortable[pivot]] && i < highest )
        i++;
      while ( sort_by[sortable[j]] > sort_by[sortable[pivot]] )
        j--;
      if( i < j ) {
        temp_id = sortable[i];
        sortable[i] = sortable[j];
        sortable[j] = temp_id;
      }
    }

    temp_id = sortable[pivot];
    sortable[pivot] = sortable[j];
    sortable[j] = temp_id;

    // Recurse
    quicksort_ints_by_floats( sortable, sort_by, lowest, j-1 );
    quicksort_ints_by_floats( sortable, sort_by, j+1, highest );
  }
}