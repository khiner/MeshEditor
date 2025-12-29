#include "UniqueBuffers.h"

#include <cassert>

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

bool ReserveStaging(UniqueBuffers &buffers, vk::DeviceSize required_size);
bool ReserveDirect(UniqueBuffers &buffers, vk::DeviceSize required_size);

bool UpdateStaging(UniqueBuffers &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return false;
    const auto required_size = offset + data.size();
    const bool reallocated = ReserveStaging(buffers, required_size);
    buffers.UsedSize = std::max(buffers.UsedSize, required_size);
    buffers.HostBuffer->Write(data, offset);
    buffers.Ctx.TransferCb->copyBuffer(**buffers.HostBuffer, *buffers.DeviceBuffer, vk::BufferCopy{offset, offset, data.size()});
    return reallocated;
}

bool UpdateDirect(UniqueBuffers &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return false;
    const auto required_size = offset + data.size();
    const bool reallocated = ReserveDirect(buffers, required_size);
    buffers.UsedSize = std::max(buffers.UsedSize, required_size);
    assert(!buffers.HostBuffer && "Direct buffer unexpectedly requires staging.");
    buffers.DeviceBuffer.Write(data, offset);
    return reallocated;
}

bool ReserveStaging(UniqueBuffers &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer.GetAllocatedSize()) return false;

    UniqueBuffers new_buffer(buffers.Ctx, NextPowerOfTwo(required_size), buffers.Usage);
    assert(new_buffer.HostBuffer && "Staging buffer unexpectedly missing.");
    if (buffers.UsedSize > 0) {
        new_buffer.HostBuffer->Write(buffers.HostBuffer->GetData());
        new_buffer.UsedSize = buffers.UsedSize;
        buffers.Ctx.TransferCb->copyBuffer(*buffers.DeviceBuffer, *new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, buffers.UsedSize});
    }
    buffers = std::move(new_buffer);
    return true;
}

bool ReserveDirect(UniqueBuffers &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer.GetAllocatedSize()) return false;

    UniqueBuffers new_buffer(buffers.Ctx, NextPowerOfTwo(required_size), buffers.Usage);
    assert(!new_buffer.HostBuffer && "Direct buffer lost host-visible mapping.");
    if (buffers.UsedSize > 0) {
        new_buffer.DeviceBuffer.Write(buffers.DeviceBuffer.GetData());
        new_buffer.UsedSize = buffers.UsedSize;
    }
    buffers = std::move(new_buffer);
    return true;
}

void InsertStaging(UniqueBuffers &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || buffers.UsedSize + data.size() > buffers.DeviceBuffer.GetAllocatedSize()) return;
    if (offset < buffers.UsedSize) {
        buffers.HostBuffer->Move(offset, offset + data.size(), buffers.UsedSize - offset);
    }
    buffers.HostBuffer->Write(data, offset);
    buffers.UsedSize += data.size();
    buffers.Ctx.TransferCb->copyBuffer(**buffers.HostBuffer, *buffers.DeviceBuffer, vk::BufferCopy{offset, offset, buffers.UsedSize - offset});
}

void InsertDirect(UniqueBuffers &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || buffers.UsedSize + data.size() > buffers.DeviceBuffer.GetAllocatedSize()) return;
    if (offset < buffers.UsedSize) {
        buffers.DeviceBuffer.Move(offset, offset + data.size(), buffers.UsedSize - offset);
    }
    buffers.DeviceBuffer.Write(data, offset);
    buffers.UsedSize += data.size();
}

void EraseStaging(UniqueBuffers &buffers, vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > buffers.UsedSize) return;
    if (const auto move_size = buffers.UsedSize - (offset + size); move_size > 0) {
        buffers.HostBuffer->Move(offset + size, offset, move_size);
        buffers.Ctx.TransferCb->copyBuffer(**buffers.HostBuffer, *buffers.DeviceBuffer, vk::BufferCopy{offset, offset, move_size});
    }
    buffers.UsedSize -= size;
}

void EraseDirect(UniqueBuffers &buffers, vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > buffers.UsedSize) return;
    if (const auto move_size = buffers.UsedSize - (offset + size); move_size > 0) {
        buffers.DeviceBuffer.Move(offset + size, offset, move_size);
    }
    buffers.UsedSize -= size;
}
} // namespace

UniqueBuffers::UniqueBuffers(const BufferContext &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage)
    : Ctx(ctx),
      Usage(usage),
      DeviceBuffer(*Ctx.Allocator, size, MemoryUsage::GpuOnly, usage | vk::BufferUsageFlagBits::eTransferDst) {
    if (!DeviceBuffer.IsMapped()) {
        HostBuffer.emplace(*Ctx.Allocator, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
    }
    ImplOps = HostBuffer ? Impl{UpdateStaging, ReserveStaging, InsertStaging, EraseStaging} : Impl{UpdateDirect, ReserveDirect, InsertDirect, EraseDirect};
}

UniqueBuffers::UniqueBuffers(const BufferContext &ctx, std::span<const std::byte> data, vk::BufferUsageFlags usage)
    : UniqueBuffers(ctx, data.size(), usage) {
    Update(data);
}

UniqueBuffers::UniqueBuffers(UniqueBuffers &&other)
    : Ctx(other.Ctx),
      UsedSize(other.UsedSize),
      Usage(other.Usage),
      HostBuffer(std::move(other.HostBuffer)),
      DeviceBuffer(std::move(other.DeviceBuffer)),
      ImplOps(other.ImplOps) {}

UniqueBuffers::~UniqueBuffers() {
    Retire();
}

void UniqueBuffers::Retire() {
    if (HostBuffer) {
        Ctx.Reclaimer.Retire(std::move(*HostBuffer));
        HostBuffer.reset();
    }
    Ctx.Reclaimer.Retire(std::move(DeviceBuffer));
}

UniqueBuffers &UniqueBuffers::operator=(UniqueBuffers &&other) {
    if (this != &other) {
        UsedSize = other.UsedSize;
        Usage = other.Usage;
        Retire();
        HostBuffer = std::move(other.HostBuffer);
        DeviceBuffer = std::move(other.DeviceBuffer);
        ImplOps = other.ImplOps;
    }
    return *this;
}

bool UniqueBuffers::Update(std::span<const std::byte> data, vk::DeviceSize offset) { return ImplOps.Update(*this, data, offset); }
bool UniqueBuffers::Reserve(vk::DeviceSize required_size) { return ImplOps.Reserve(*this, required_size); }
void UniqueBuffers::Insert(std::span<const std::byte> data, vk::DeviceSize offset) { ImplOps.Insert(*this, data, offset); }
void UniqueBuffers::Erase(vk::DeviceSize offset, vk::DeviceSize size) { ImplOps.Erase(*this, offset, size); }
} // namespace mvk
