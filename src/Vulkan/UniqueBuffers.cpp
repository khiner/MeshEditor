#include "UniqueBuffers.h"

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

UniqueBuffers::UniqueBuffers(const BufferContext &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage)
    : Ctx(ctx),
      Usage(usage),
      HostBuffer(*Ctx.Allocator, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc),
      DeviceBuffer(*Ctx.Allocator, size, MemoryUsage::GpuOnly, usage | vk::BufferUsageFlagBits::eTransferDst) {}

UniqueBuffers::UniqueBuffers(const BufferContext &ctx, std::span<const std::byte> data, vk::BufferUsageFlags usage)
    : UniqueBuffers(ctx, data.size(), usage) {
    UsedSize = data.size();
    HostBuffer.Write(data);
    Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{0, 0, data.size()});
}

UniqueBuffers::UniqueBuffers(UniqueBuffers &&other)
    : Ctx(other.Ctx),
      UsedSize(other.UsedSize),
      Usage(other.Usage),
      HostBuffer(std::move(other.HostBuffer)),
      DeviceBuffer(std::move(other.DeviceBuffer)) {}

UniqueBuffers::~UniqueBuffers() {
    Retire();
}

void UniqueBuffers::Retire() {
    Ctx.Reclaimer.Retire(std::move(HostBuffer));
    Ctx.Reclaimer.Retire(std::move(DeviceBuffer));
}

UniqueBuffers &UniqueBuffers::operator=(UniqueBuffers &&other) {
    if (this != &other) {
        UsedSize = other.UsedSize;
        Usage = other.Usage;
        Retire();
        HostBuffer = std::move(other.HostBuffer);
        DeviceBuffer = std::move(other.DeviceBuffer);
    }
    return *this;
}

void UniqueBuffers::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return;

    const auto required_size = offset + data.size();
    Reserve(required_size);
    UsedSize = std::max(UsedSize, required_size);
    HostBuffer.Write(data, offset);
    Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, data.size()});
}

void UniqueBuffers::Reserve(vk::DeviceSize required_size) {
    if (required_size <= DeviceBuffer.GetAllocatedSize()) return;

    // Create a new buffer with enough space.
    UniqueBuffers new_buffer(Ctx, NextPowerOfTwo(required_size), Usage);
    if (UsedSize > 0) {
        // Copy the old buffer into the new buffer (host and device).
        new_buffer.HostBuffer.Write(HostBuffer.GetData());
        new_buffer.UsedSize = UsedSize;
        Ctx.TransferCb->copyBuffer(*DeviceBuffer, *new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, UsedSize});
    }
    *this = std::move(new_buffer);
}

void UniqueBuffers::Insert(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || UsedSize + data.size() > DeviceBuffer.GetAllocatedSize()) return;

    if (offset < UsedSize) {
        HostBuffer.Move(offset, offset + data.size(), UsedSize - offset);
    }
    HostBuffer.Write(data, offset);
    UsedSize += data.size();
    Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, UsedSize - offset});
}

void UniqueBuffers::Erase(vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > UsedSize) return;

    if (const auto move_size = UsedSize - (offset + size); move_size > 0) {
        HostBuffer.Move(offset + size, offset, move_size);
        Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, move_size});
    }
    UsedSize -= size;
}
} // namespace mvk
