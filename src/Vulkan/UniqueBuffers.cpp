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
      // Device buffers need to be copy destinations for staging uploads and copy sources for reallocations.
      DeviceBuffer(
          *Ctx.Allocator,
          size,
          MemoryUsage::GpuOnly,
          usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst
      ) {}

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
      DeviceBuffer(std::move(other.DeviceBuffer)),
      PendingWriteSize(other.PendingWriteSize),
      WritePending(other.WritePending) {}

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
        PendingWriteSize = other.PendingWriteSize;
        WritePending = other.WritePending;
        Usage = other.Usage;
        Retire();
        HostBuffer = std::move(other.HostBuffer);
        DeviceBuffer = std::move(other.DeviceBuffer);
    }
    return *this;
}

void UniqueBuffers::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return;

    PendingWriteSize = 0;
    WritePending = false;

    const auto required_size = offset + data.size();
    Reserve(required_size);
    UsedSize = std::max(UsedSize, required_size);
    HostBuffer.Write(data, offset);
    Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{offset, offset, data.size()});
}

vk::DeviceSize UniqueBuffers::Append(std::span<const std::byte> data) {
    if (data.empty()) return UsedSize;
    if (!WritePending) {
        PendingWriteSize = 0;
        UsedSize = 0;
        WritePending = true;
    }

    const auto offset = UsedSize;
    const auto required_size = UsedSize + data.size();
    ReserveForWrite(required_size);
    HostBuffer.Write(data, offset);
    UsedSize = required_size;
    PendingWriteSize = UsedSize;
    return offset;
}

void UniqueBuffers::EndWrite() {
    if (!WritePending) return;
    if (PendingWriteSize > 0) {
        Ctx.TransferCb->copyBuffer(*HostBuffer, *DeviceBuffer, vk::BufferCopy{0, 0, PendingWriteSize});
    }
    PendingWriteSize = 0;
    WritePending = false;
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

void UniqueBuffers::ReserveForWrite(vk::DeviceSize required_size) {
    if (required_size <= DeviceBuffer.GetAllocatedSize()) return;

    const bool was_write_pending = WritePending;
    const auto pending_size = PendingWriteSize;
    UniqueBuffers new_buffer(Ctx, NextPowerOfTwo(required_size), Usage);
    if (UsedSize > 0) {
        const auto host_data = HostBuffer.GetMappedData().subspan(0, UsedSize);
        new_buffer.HostBuffer.Write(std::span<const std::byte>(host_data.data(), host_data.size()));
        new_buffer.UsedSize = UsedSize;
    }
    *this = std::move(new_buffer);
    WritePending = was_write_pending;
    PendingWriteSize = pending_size;
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
