#include "BufferManager.h"

namespace mvk {
namespace {
// Adapted from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 for 64-bits.
uint64_t NextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}
} // namespace

UniqueBuffers::UniqueBuffers(const BufferManager &manager, vk::DeviceSize size, vk::BufferUsageFlags usage)
    : Manager(manager),
      Usage(usage),
      HostBuffer(*Manager, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc),
      DeviceBuffer(*Manager, size, MemoryUsage::GpuOnly, usage) {}

UniqueBuffers::UniqueBuffers(const BufferManager &manager, std::span<const std::byte> data, vk::BufferUsageFlags usage)
    : UniqueBuffers(manager, data.size(), usage) {
    Size = data.size();
    HostBuffer.Write(data);
    Manager.TransferCb.copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{0, 0, data.size()});
}

UniqueBuffers::UniqueBuffers(UniqueBuffers &&other)
    : Manager(other.Manager),
      Usage(other.Usage),
      HostBuffer(std::move(other.HostBuffer)),
      DeviceBuffer(std::move(other.DeviceBuffer)),
      Size(other.Size) {}

UniqueBuffers::~UniqueBuffers() {
    Manager.MarkStale(std::move(HostBuffer));
    Manager.MarkStale(std::move(DeviceBuffer));
}

UniqueBuffers &UniqueBuffers::operator=(UniqueBuffers &&other) {
    if (this != &other) {
        Manager.MarkStale(std::move(HostBuffer));
        Manager.MarkStale(std::move(DeviceBuffer));
        HostBuffer = std::move(other.HostBuffer);
        DeviceBuffer = std::move(other.DeviceBuffer);
        Size = other.Size;
        Usage = other.Usage;
    }
    return *this;
}

void UniqueBuffers::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return;

    const auto required_size = offset + data.size();
    EnsureAllocated(required_size);
    Size = std::max(Size, required_size);
    HostBuffer.Write(data, offset);
    Manager.TransferCb.copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, data.size()});
}

void UniqueBuffers::EnsureAllocated(vk::DeviceSize required_size) {
    if (required_size == 0) return;
    if (required_size <= DeviceBuffer.GetAllocatedSize()) return;

    // Create a new buffer with enough space.
    const auto new_size = NextPowerOfTwo(required_size);
    UniqueBuffers new_buffer(Manager, new_size, Usage);
    if (Size > 0) {
        // Copy the old buffer into the new buffer (host and device).
        new_buffer.HostBuffer.Write(HostBuffer.GetData());
        new_buffer.Size = Size;
        Manager.TransferCb.copyBuffer(*DeviceBuffer, *new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, Size});
    }
    *this = std::move(new_buffer);
}

void UniqueBuffers::Insert(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || Size + data.size() > DeviceBuffer.GetAllocatedSize()) return;

    if (offset < Size) {
        HostBuffer.Move(offset, offset + data.size(), Size - offset);
    }
    HostBuffer.Write(data, offset);
    Size += data.size();
    Manager.TransferCb.copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, Size - offset});
}
void UniqueBuffers::Erase(vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > Size) return;

    if (const auto move_size = Size - (offset + size); move_size > 0) {
        HostBuffer.Move(offset + size, offset, move_size);
        Manager.TransferCb.copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, move_size});
    }
    Size -= size;
}
} // namespace mvk
