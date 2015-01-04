// ext/ru_ne_ne/core_shuffle.c

#include "core_shuffle.h"

// A Fisher-Yates shuffle
void shuffle_ints( int n, int *array ) {
  int i, tmp, r;
  for ( i = n-1; i >= 0; i-- ) {
    // This will be slightly biased for large n, but it is not a
    // noticeable issue for dynamic solutions
    r = genrand_int31() % ( i + 1 );
    tmp = array[r];
    array[r] = array[i];
    array[i] = tmp;
  }
  return;
}
