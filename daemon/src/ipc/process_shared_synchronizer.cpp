#include "ipc/process_shared_synchronizer.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace exam {
namespace {

std::string error_message(int rc) {
    return std::strerror(rc);
}

} // namespace

void ProcessSharedSynchronizer::check(int rc, const char* what) {
    if (rc != 0) {
        throw std::runtime_error(std::string(what) + ": " + error_message(rc));
    }
}

void ProcessSharedSynchronizer::init_mutex(
    pthread_mutex_t* mutex,
    bool process_shared,
    bool* priority_inheritance_enabled) {
    pthread_mutexattr_t attr;
    check(pthread_mutexattr_init(&attr), "pthread_mutexattr_init");

    if (process_shared) {
        check(
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED),
            "pthread_mutexattr_setpshared");
    }

    const int protocol_rc = pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    if (priority_inheritance_enabled != nullptr) {
        *priority_inheritance_enabled = (protocol_rc == 0);
    }
    if (protocol_rc != 0) {
        std::cerr << "PTHREAD_PRIO_INHERIT was not enabled: "
                  << error_message(protocol_rc) << '\n';
    }

    try {
        check(pthread_mutex_init(mutex, &attr), "pthread_mutex_init");
    } catch (...) {
        pthread_mutexattr_destroy(&attr);
        throw;
    }

    check(pthread_mutexattr_destroy(&attr), "pthread_mutexattr_destroy");
}

void ProcessSharedSynchronizer::init_cond(pthread_cond_t* cond) {
    pthread_condattr_t attr;
    check(pthread_condattr_init(&attr), "pthread_condattr_init");
    check(
        pthread_condattr_setpshared(&attr, PTHREAD_PROCESS_SHARED),
        "pthread_condattr_setpshared");

    try {
        check(pthread_cond_init(cond, &attr), "pthread_cond_init");
    } catch (...) {
        pthread_condattr_destroy(&attr);
        throw;
    }

    check(pthread_condattr_destroy(&attr), "pthread_condattr_destroy");
}

} // namespace exam
