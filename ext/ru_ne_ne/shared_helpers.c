// ext/ru_ne_ne/shared_helpers.c

#include "shared_helpers.h"

// Hash lookup helper
VALUE ValAtSymbol(VALUE hash, const char* key) {
    return rb_hash_lookup(hash, ID2SYM( rb_intern(key) ) );
}
