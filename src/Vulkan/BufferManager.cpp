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

vk::Buffer BufferManager::CreateStaging(std::span<const std::byte> data) const {
    auto staging_buffer = Allocator.Allocate(data.size(), MemoryUsage::CpuOnly);
    Allocator.Write(staging_buffer, data);
    return staging_buffer;
}

Buffer BufferManager::Create(std::span<const std::byte> data, vk::BufferUsageFlags usage) const {
    auto buffer = Allocate(data.size(), usage);
    buffer.Size = data.size();
    Allocator.Write(buffer.HostBuffer, data);
    TransferCb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{0, 0, data.size()});
    return buffer;
}
void BufferManager::EnsureAllocated(Buffer &buffer, vk::DeviceSize required_size) const {
    if (required_size == 0) return;
    if (required_size <= GetAllocatedSize(buffer)) return;

    // Create a new buffer with enough space.
    auto new_buffer = Allocate(NextPowerOfTwo(required_size), buffer.Usage);
    if (buffer.Size > 0) {
        // Copy the old buffer into the new buffer (host and device).
        Allocator.Write(new_buffer.HostBuffer, Allocator.GetData(buffer.HostBuffer));
        new_buffer.Size = buffer.Size;
        TransferCb.copyBuffer(buffer.DeviceBuffer, new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, buffer.Size});
    }
    Invalidate(buffer);
    buffer = std::move(new_buffer);
}
void BufferManager::Update(Buffer &buffer, std::span<const std::byte> data, vk::DeviceSize offset) const {
    if (data.empty()) return;

    const auto required_size = offset + data.size();
    EnsureAllocated(buffer, required_size);
    buffer.Size = std::max(buffer.Size, required_size);
    Allocator.Write(buffer.HostBuffer, data, offset);
    TransferCb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, data.size()});
}
void BufferManager::Insert(Buffer &buffer, std::span<const std::byte> data, vk::DeviceSize offset) const {
    if (data.empty() || buffer.Size + data.size() > GetAllocatedSize(buffer)) return;

    if (offset < buffer.Size) {
        Allocator.Move(buffer.HostBuffer, offset, offset + data.size(), buffer.Size - offset);
    }
    Allocator.Write(buffer.HostBuffer, data, offset);
    buffer.Size += data.size();
    TransferCb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, buffer.Size - offset});
}
void BufferManager::Erase(Buffer &buffer, vk::DeviceSize offset, vk::DeviceSize size) const {
    if (size == 0 || offset + size > buffer.Size) return;

    if (const auto move_size = buffer.Size - (offset + size); move_size > 0) {
        Allocator.Move(buffer.HostBuffer, offset + size, offset, move_size);
        TransferCb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, move_size});
    }
    buffer.Size -= size;
}
} // namespace mvk
