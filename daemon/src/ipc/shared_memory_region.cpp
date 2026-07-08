#include "ipc/shared_memory_region.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

namespace exam {
namespace {

std::string posix_error(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

} // namespace

SharedMemoryRegion SharedMemoryRegion::create(const std::string& name, std::size_t size) {
    shm_unlink(name.c_str());

    const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd == -1) {
        throw std::runtime_error(posix_error("shm_open create"));
    }

    if (ftruncate(fd, static_cast<off_t>(size)) == -1) {
        const int saved_errno = errno;
        close(fd);
        shm_unlink(name.c_str());
        errno = saved_errno;
        throw std::runtime_error(posix_error("ftruncate"));
    }

    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        const int saved_errno = errno;
        close(fd);
        shm_unlink(name.c_str());
        errno = saved_errno;
        throw std::runtime_error(posix_error("mmap create"));
    }

    return SharedMemoryRegion(name, fd, data, size);
}

SharedMemoryRegion SharedMemoryRegion::open(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd == -1) {
        throw std::runtime_error(posix_error("shm_open open"));
    }

    const off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size == static_cast<off_t>(-1)) {
        const int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        throw std::runtime_error(posix_error("lseek open"));
    }

    const std::size_t size = static_cast<std::size_t>(file_size);
    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        const int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        throw std::runtime_error(posix_error("mmap open"));
    }

    return SharedMemoryRegion(name, fd, data, size);
}

void SharedMemoryRegion::unlink(const std::string& name) {
    if (!name.empty() && shm_unlink(name.c_str()) == -1 && errno != ENOENT) {
        throw std::runtime_error(posix_error("shm_unlink"));
    }
}

void SharedMemoryRegion::unlink_noexcept(const char* name) {
    if (name != nullptr && name[0] != '\0') {
        shm_unlink(name);
    }
}

SharedMemoryRegion::~SharedMemoryRegion() {
    close_region();
}

SharedMemoryRegion::SharedMemoryRegion(SharedMemoryRegion&& other) noexcept
    : name_(std::move(other.name_)),
      fd_(other.fd_),
      data_(other.data_),
      size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

SharedMemoryRegion& SharedMemoryRegion::operator=(SharedMemoryRegion&& other) noexcept {
    if (this != &other) {
        close_region();
        name_ = std::move(other.name_);
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool SharedMemoryRegion::valid() const {
    return data_ != nullptr;
}

const std::string& SharedMemoryRegion::name() const {
    return name_;
}

std::size_t SharedMemoryRegion::size() const {
    return size_;
}

void* SharedMemoryRegion::data() const {
    return data_;
}

void SharedMemoryRegion::clear() {
    if (!valid()) {
        throw std::runtime_error("shared memory region is not open");
    }
    std::memset(data_, 0, size_);
}

void SharedMemoryRegion::write_bytes(const void* source, std::size_t source_size) {
    if (!valid()) {
        throw std::runtime_error("shared memory region is not open");
    }
    if (source == nullptr && source_size != 0) {
        throw std::runtime_error("source buffer is null");
    }
    if (source_size > size_) {
        throw std::runtime_error("source buffer is larger than shared memory region");
    }

    clear();
    std::memcpy(data_, source, source_size);
}

void SharedMemoryRegion::read_bytes(void* destination, std::size_t destination_size) const {
    if (!valid()) {
        throw std::runtime_error("shared memory region is not open");
    }
    if (destination == nullptr && destination_size != 0) {
        throw std::runtime_error("destination buffer is null");
    }
    if (destination_size > size_) {
        throw std::runtime_error("destination buffer is larger than shared memory region");
    }

    std::memcpy(destination, data_, destination_size);
}

SharedMemoryRegion::SharedMemoryRegion(std::string name, int fd, void* data, std::size_t size)
    : name_(std::move(name)),
      fd_(fd),
      data_(data),
      size_(size) {}

void SharedMemoryRegion::close_region() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

} // namespace exam
