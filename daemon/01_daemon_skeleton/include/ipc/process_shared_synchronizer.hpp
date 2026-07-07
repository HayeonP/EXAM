#pragma once

#include <pthread.h>

namespace exam {

class ProcessSharedSynchronizer {
public:
    static void check(int rc, const char* what);
    static void init_mutex(
        pthread_mutex_t* mutex,
        bool process_shared,
        bool* priority_inheritance_enabled = nullptr);
    static void init_cond(pthread_cond_t* cond);
};

} // namespace exam
