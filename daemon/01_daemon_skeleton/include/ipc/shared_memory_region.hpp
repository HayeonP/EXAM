#pragma once

#include <cstddef>
#include <string>

namespace exam {

class SharedMemoryRegion {
public:
    SharedMemoryRegion() = default;
    static SharedMemoryRegion create(const std::string& name, std::size_t size);
    static SharedMemoryRegion open(const std::string& name);
    static void unlink(const std::string& name);
    static void unlink_noexcept(const char* name);

    ~SharedMemoryRegion();

    SharedMemoryRegion(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;

    SharedMemoryRegion(SharedMemoryRegion&& other) noexcept;
    SharedMemoryRegion& operator=(SharedMemoryRegion&& other) noexcept;

    bool valid() const;
    const std::string& name() const;
    std::size_t size() const;
    void* data() const;
    void clear();
    void write_bytes(const void* source, std::size_t source_size);
    void read_bytes(void* destination, std::size_t destination_size) const;

private:
    SharedMemoryRegion(std::string name, int fd, void* data, std::size_t size);

    void close_region();

    std::string name_;
    int fd_{-1};
    void* data_{nullptr};
    std::size_t size_{0};
};

} // namespace exam
