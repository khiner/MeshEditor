#include "UniqueBuffers.h"
#include "../Bindless.h"

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

// Reallocate underlying storage without touching the slot.
bool ReserveStaging(UniqueBuffers &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer.GetAllocatedSize()) return false;

    const auto new_size = NextPowerOfTwo(required_size);
    UniqueBuffer new_device(*buffers.Ctx.Allocator, new_size, MemoryUsage::GpuOnly, buffers.Usage | vk::BufferUsageFlagBits::eTransferDst);
    UniqueBuffer new_host(*buffers.Ctx.Allocator, new_size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);

    if (buffers.UsedSize > 0) {
        new_host.Write(buffers.HostBuffer->GetData());
        buffers.Ctx.TransferCb->copyBuffer(*buffers.DeviceBuffer, *new_device, vk::BufferCopy{0, 0, buffers.UsedSize});
    }

    buffers.Ctx.Reclaimer.Retire(std::move(*buffers.HostBuffer));
    buffers.Ctx.Reclaimer.Retire(std::move(buffers.DeviceBuffer));
    buffers.HostBuffer.emplace(std::move(new_host));
    buffers.DeviceBuffer = std::move(new_device);
    return true;
}

// Reallocate underlying storage without touching the slot.
bool ReserveDirect(UniqueBuffers &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer.GetAllocatedSize()) return false;

    const auto new_size = NextPowerOfTwo(required_size);
    UniqueBuffer new_device(*buffers.Ctx.Allocator, new_size, MemoryUsage::GpuOnly, buffers.Usage | vk::BufferUsageFlagBits::eTransferDst);

    if (buffers.UsedSize > 0) {
        new_device.Write(buffers.DeviceBuffer.GetData());
    }

    buffers.Ctx.Reclaimer.Retire(std::move(buffers.DeviceBuffer));
    buffers.DeviceBuffer = std::move(new_device);
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

UniqueBuffers::UniqueBuffers(BufferContext &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, SlotType slot_type)
    : Ctx(ctx),
      Type(slot_type),
      Slot(ctx.Slots.Allocate(slot_type)),
      Usage(usage),
      DeviceBuffer(*Ctx.Allocator, size, MemoryUsage::GpuOnly, usage | vk::BufferUsageFlagBits::eTransferDst) {
    if (!DeviceBuffer.IsMapped()) {
        HostBuffer.emplace(*Ctx.Allocator, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
    }
    ImplOps = HostBuffer ? Impl{UpdateStaging, ReserveStaging, InsertStaging, EraseStaging} : Impl{UpdateDirect, ReserveDirect, InsertDirect, EraseDirect};
    UpdateDescriptor();
}

UniqueBuffers::UniqueBuffers(BufferContext &ctx, std::span<const std::byte> data, vk::BufferUsageFlags usage, SlotType slot_type)
    : UniqueBuffers(ctx, data.size(), usage, slot_type) {
    Update(data);
}

UniqueBuffers::UniqueBuffers(UniqueBuffers &&other)
    : Ctx(other.Ctx),
      Type(other.Type),
      Slot(other.Slot),
      UsedSize(other.UsedSize),
      Usage(other.Usage),
      HostBuffer(std::move(other.HostBuffer)),
      DeviceBuffer(std::move(other.DeviceBuffer)),
      ImplOps(other.ImplOps) {
    other.Slot = InvalidSlot; // Prevent other from releasing the slot
}

UniqueBuffers::~UniqueBuffers() {
    Retire();
    if (Slot != InvalidSlot) {
        Ctx.Slots.Release(Type, Slot);
    }
}

void UniqueBuffers::Retire() {
    if (HostBuffer) {
        Ctx.Reclaimer.Retire(std::move(*HostBuffer));
        HostBuffer.reset();
    }
    Ctx.Reclaimer.Retire(std::move(DeviceBuffer));
}

void UniqueBuffers::UpdateDescriptor() {
    Ctx.Device.updateDescriptorSets(Ctx.Slots.MakeBufferWrite(Type, Slot, GetDescriptor()), {});
}

UniqueBuffers &UniqueBuffers::operator=(UniqueBuffers &&other) {
    if (this != &other) {
        // Release our current resources
        Retire();
        if (Slot != InvalidSlot) {
            Ctx.Slots.Release(Type, Slot);
        }
        // Take over other's resources
        Type = other.Type;
        Slot = other.Slot;
        UsedSize = other.UsedSize;
        Usage = other.Usage;
        HostBuffer = std::move(other.HostBuffer);
        DeviceBuffer = std::move(other.DeviceBuffer);
        ImplOps = other.ImplOps;
        other.Slot = InvalidSlot; // Prevent other from releasing the slot
    }
    return *this;
}

void UniqueBuffers::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (ImplOps.Update(*this, data, offset)) UpdateDescriptor();
}

void UniqueBuffers::Reserve(vk::DeviceSize required_size) {
    if (ImplOps.Reserve(*this, required_size)) UpdateDescriptor();
}

void UniqueBuffers::Insert(std::span<const std::byte> data, vk::DeviceSize offset) { ImplOps.Insert(*this, data, offset); }
void UniqueBuffers::Erase(vk::DeviceSize offset, vk::DeviceSize size) { ImplOps.Erase(*this, offset, size); }
} // namespace mvk
