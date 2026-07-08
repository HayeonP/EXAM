#include "ipc/shared_memory_region.hpp"

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

constexpr const char* SAMPLE_SHM_NAME = "/exam_shared_memory_region_sample";
constexpr std::size_t MESSAGE_SIZE = 64;

class SampleState {
public:
    std::uint32_t version;
    std::uint64_t event_count;
    char message[MESSAGE_SIZE];
};

} // namespace

int main() {
    try {
        // PROVIDER
        // 0. init
        exam::SharedMemoryRegion::unlink_noexcept(SAMPLE_SHM_NAME);

        // 1. create
        exam::SharedMemoryRegion owner_region =
            exam::SharedMemoryRegion::create(SAMPLE_SHM_NAME, sizeof(SampleState));

        // 2. write
        SampleState local_state{1, 3, "hello from shared memory"};
        owner_region.write_bytes(&local_state, sizeof(local_state));

        SampleState writer_snapshot{};
        owner_region.read_bytes(&writer_snapshot, sizeof(writer_snapshot));
        std::cout << "writer: version=" << writer_snapshot.version
                  << ", event_count=" << writer_snapshot.event_count
                  << ", message=\"" << writer_snapshot.message << "\"\n";

        // CONSUMER
        // 3. open
        exam::SharedMemoryRegion reader_region =
            exam::SharedMemoryRegion::open(SAMPLE_SHM_NAME);

        // 4. read_bytes
        SampleState reader_snapshot{};
        reader_region.read_bytes(&reader_snapshot, sizeof(reader_snapshot));

        std::cout << "reader: version=" << reader_snapshot.version
                  << ", event_count=" << reader_snapshot.event_count
                  << ", message=\"" << reader_snapshot.message << "\"\n";

        // Test
        reader_snapshot.event_count += 1;
        reader_region.write_bytes(&reader_snapshot, sizeof(reader_snapshot));
        owner_region.read_bytes(&writer_snapshot, sizeof(writer_snapshot));
        std::cout << "writer after reader update: event_count="
                  << writer_snapshot.event_count << '\n';

        // 6. unlink는 이름을 제거하는 동작이다.
        //    이미 mmap된 영역은 SharedMemoryRegion 객체가 소멸될 때 자동으로 unmap/close된다.
        exam::SharedMemoryRegion::unlink(SAMPLE_SHM_NAME);
        return 0;
    } catch (const std::exception& e) {
        exam::SharedMemoryRegion::unlink_noexcept(SAMPLE_SHM_NAME);
        std::cerr << "shared_memory_region_sample: " << e.what() << '\n';
        return 1;
    }
}
