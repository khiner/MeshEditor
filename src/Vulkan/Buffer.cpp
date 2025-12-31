#include "Buffer.h"
#include "../Bindless.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <algorithm>
#include <cassert>
#include <print>
#include <ranges>

using std::views::transform, std::ranges::to;

namespace mvk {
namespace {
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

VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    }
}

std::string FormatBytes(uint32_t bytes) {
    static constexpr std::array<std::string_view, 6> Suffixes{"B", "KB", "MB", "GB", "TB", "PB"};
    static constexpr float K = 1024;
    auto value = float(bytes);
    size_t pow;
    for (pow = 0; value >= K && pow + 1 < Suffixes.size(); ++pow) value /= K;
    return std::format("{:.2f} {}", value, Suffixes[pow]);
}

#ifndef RELEASE_BUILD
void LoggingVmaAllocate(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Allocating {} bytes of memory of type {}", FormatBytes(size), memory_type);
}
void LoggingVmaFree(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Freeing {} bytes of memory of type {}", FormatBytes(size), memory_type);
}
#endif

std::vector<VmaBudget> QueryHeapBudgets(VmaAllocator allocator, vk::PhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties memory_props;
    vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(pd), &memory_props);
    std::vector<VmaBudget> budgets(memory_props.memoryHeapCount);
    vmaGetHeapBudgets(allocator, budgets.data());
    return budgets;
}
} // namespace

struct Buffer::VmaBuffer {
    VmaBuffer(VmaAllocator vma, vk::DeviceSize size, MemoryUsage memory_usage, vk::BufferUsageFlags usage) : Vma(vma) {
        VmaAllocationCreateInfo aci{};
        aci.usage = ToVmaMemoryUsage(memory_usage);
        if (memory_usage == MemoryUsage::GpuOnly) {
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        vk::BufferCreateInfo bci{{}, size, usage | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
        if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
            aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        } else {
            bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
        }
        if (vmaCreateBuffer(Vma, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&Handle), &Allocation, &Info) != VK_SUCCESS) {
            throw std::runtime_error("vmaCreateBuffer failed");
        }
        vmaGetMemoryTypeProperties(Vma, Info.memoryType, &MemoryProps);
    }
    VmaBuffer(VmaAllocator vma, std::span<const std::byte> data, MemoryUsage mem, vk::BufferUsageFlags usage)
        : VmaBuffer(vma, data.size(), mem, usage) { Write(data); }
    ~VmaBuffer() { vmaDestroyBuffer(Vma, Handle, Allocation); }

    vk::Buffer Get() const { return Handle; }
    vk::DeviceSize GetAllocatedSize() const { return Info.size; }
    std::span<const std::byte> GetData() const { return {static_cast<const std::byte *>(Info.pMappedData), Info.size}; }
    std::span<std::byte> GetMappedData() const { return {static_cast<std::byte *>(Info.pMappedData), Info.size}; }
    bool IsMapped() const {
        return Info.pMappedData != nullptr &&
            (MemoryProps & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    void Write(std::span<const std::byte> src, vk::DeviceSize offset = 0) const {
        if (src.empty() || offset >= GetAllocatedSize()) return;
        std::copy(src.begin(), src.end(), GetMappedData().subspan(offset).data());
    }
    void Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const {
        const auto allocated_size = GetAllocatedSize();
        if (size == 0 || from + size > allocated_size || to + size > allocated_size) return;
        const auto mapped_data = GetMappedData();
        std::memmove(mapped_data.subspan(to).data(), mapped_data.subspan(from).data(), size_t(size));
    }

    VmaAllocator Vma;
    vk::Buffer Handle;
    VmaAllocation Allocation;
    VmaAllocationInfo Info;
    VkMemoryPropertyFlags MemoryProps{};
};

struct DeferredBufferReclaimer {
    void Retire(std::unique_ptr<Buffer::VmaBuffer> buffer) { Retired.emplace_back(std::move(buffer)); }
    void Reclaim() { Retired.clear(); }

private:
    std::vector<std::unique_ptr<Buffer::VmaBuffer>> Retired;
};

BufferContext::BufferContext(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, vk::CommandPool command_pool, DescriptorSlots &slots)
    : PhysicalDevice(pd), Device(d),
      TransferCb(std::move(d.allocateCommandBuffersUnique({command_pool, vk::CommandBufferLevel::ePrimary, 1}).front())),
      Reclaimer(std::make_unique<DeferredBufferReclaimer>()),
      Slots(slots) {
    VmaAllocatorCreateInfo create_info{};
    create_info.physicalDevice = PhysicalDevice;
    create_info.device = Device;
    create_info.instance = instance;
#ifndef RELEASE_BUILD
    VmaDeviceMemoryCallbacks memory_callbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    create_info.pDeviceMemoryCallbacks = &memory_callbacks;
#endif
    if (vmaCreateAllocator(&create_info, &Vma) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator.");
    }
    TransferCb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}
BufferContext::~BufferContext() {
    TransferCb->end();
    Reclaimer->Reclaim();
    vmaDestroyAllocator(Vma);
}

void BufferContext::ReclaimRetiredBuffers() { Reclaimer->Reclaim(); }
std::string BufferContext::DebugHeapUsage() const {
    const auto budgets = QueryHeapBudgets(Vma, PhysicalDevice);
    std::string result;
    for (uint32_t i = 0; i < budgets.size(); ++i) {
        const auto &b = budgets[i];
        result += std::format(
            "Heap {}/{}:\n\tAllocations:\n\t\tCount: {}\n\t\tBytes: {}\n\tBlocks:\n\t\tCount: {}\n\t\tBytes: {}\n\tTotal:\n\t\tUsed: {}\n\t\tBudget: {}\n",
            i + 1, budgets.size(), b.statistics.allocationCount, FormatBytes(b.statistics.allocationBytes),
            b.statistics.blockCount, FormatBytes(b.statistics.blockBytes), FormatBytes(b.usage), FormatBytes(b.budget)
        );
    }
    return result;
}

std::vector<vk::WriteDescriptorSet> BufferContext::GetPendingDescriptorUpdates() {
    return PendingDescriptorUpdates | transform([&](const auto &pending) { return Slots.MakeBufferWrite(pending.Type, pending.Slot, pending.Info); }) | to<std::vector>();
}

void BufferContext::CancelDescriptorUpdate(SlotType type, uint32_t slot) {
    PendingDescriptorUpdates.erase(PendingDescriptorUpdate{type, slot, {}});
}

namespace {
bool ReserveStaging(Buffer &buffers, vk::DeviceSize required_size);
bool ReserveDirect(Buffer &buffers, vk::DeviceSize required_size);

bool UpdateStaging(Buffer &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return false;
    const auto required_size = offset + data.size();
    const bool reallocated = ReserveStaging(buffers, required_size);
    buffers.UsedSize = std::max(buffers.UsedSize, required_size);
    buffers.HostBuffer->Write(data, offset);
    buffers.Ctx.TransferCb->copyBuffer(buffers.HostBuffer->Get(), buffers.DeviceBuffer->Get(), vk::BufferCopy{offset, offset, data.size()});
    return reallocated;
}

bool UpdateDirect(Buffer &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return false;
    const auto required_size = offset + data.size();
    const bool reallocated = ReserveDirect(buffers, required_size);
    buffers.UsedSize = std::max(buffers.UsedSize, required_size);
    assert(!buffers.HostBuffer && "Direct buffer unexpectedly requires staging.");
    buffers.DeviceBuffer->Write(data, offset);
    return reallocated;
}

bool ReserveStaging(Buffer &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer->GetAllocatedSize()) return false;
    const auto new_size = NextPowerOfTwo(required_size);
    auto new_host = std::make_unique<Buffer::VmaBuffer>(buffers.Ctx.Vma, new_size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
    auto new_device = std::make_unique<Buffer::VmaBuffer>(buffers.Ctx.Vma, new_size, MemoryUsage::GpuOnly, buffers.Usage | vk::BufferUsageFlagBits::eTransferDst);
    if (buffers.UsedSize > 0) {
        new_host->Write(buffers.HostBuffer->GetData());
        buffers.Ctx.TransferCb->copyBuffer(buffers.DeviceBuffer->Get(), new_device->Get(), vk::BufferCopy{0, 0, buffers.UsedSize});
    }
    buffers.Ctx.Reclaimer->Retire(std::move(buffers.HostBuffer));
    buffers.Ctx.Reclaimer->Retire(std::move(buffers.DeviceBuffer));
    buffers.HostBuffer = std::move(new_host);
    buffers.DeviceBuffer = std::move(new_device);
    return true;
}

bool ReserveDirect(Buffer &buffers, vk::DeviceSize required_size) {
    if (required_size <= buffers.DeviceBuffer->GetAllocatedSize()) return false;
    const auto new_size = NextPowerOfTwo(required_size);
    auto new_device = std::make_unique<Buffer::VmaBuffer>(buffers.Ctx.Vma, new_size, MemoryUsage::GpuOnly, buffers.Usage | vk::BufferUsageFlagBits::eTransferDst);
    if (buffers.UsedSize > 0) {
        new_device->Write(buffers.DeviceBuffer->GetData());
    }
    buffers.Ctx.Reclaimer->Retire(std::move(buffers.DeviceBuffer));
    buffers.DeviceBuffer = std::move(new_device);
    return true;
}

void InsertStaging(Buffer &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || buffers.UsedSize + data.size() > buffers.DeviceBuffer->GetAllocatedSize()) return;
    if (offset < buffers.UsedSize) buffers.HostBuffer->Move(offset, offset + data.size(), buffers.UsedSize - offset);
    buffers.HostBuffer->Write(data, offset);
    buffers.UsedSize += data.size();
    buffers.Ctx.TransferCb->copyBuffer(buffers.HostBuffer->Get(), buffers.DeviceBuffer->Get(), vk::BufferCopy{offset, offset, buffers.UsedSize - offset});
}

void InsertDirect(Buffer &buffers, std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || buffers.UsedSize + data.size() > buffers.DeviceBuffer->GetAllocatedSize()) return;
    if (offset < buffers.UsedSize) buffers.DeviceBuffer->Move(offset, offset + data.size(), buffers.UsedSize - offset);
    buffers.DeviceBuffer->Write(data, offset);
    buffers.UsedSize += data.size();
}

void EraseStaging(Buffer &buffers, vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > buffers.UsedSize) return;
    if (const auto move_size = buffers.UsedSize - (offset + size); move_size > 0) {
        buffers.HostBuffer->Move(offset + size, offset, move_size);
        buffers.Ctx.TransferCb->copyBuffer(buffers.HostBuffer->Get(), buffers.DeviceBuffer->Get(), vk::BufferCopy{offset, offset, move_size});
    }
    buffers.UsedSize -= size;
}

void EraseDirect(Buffer &buffers, vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > buffers.UsedSize) return;
    if (const auto move_size = buffers.UsedSize - (offset + size); move_size > 0) {
        buffers.DeviceBuffer->Move(offset + size, offset, move_size);
    }
    buffers.UsedSize -= size;
}
} // namespace

Buffer::Buffer(BufferContext &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, SlotType slot_type)
    : Ctx(ctx), Slot(ctx.Slots.Allocate(slot_type)), Usage(usage),
      DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, size, MemoryUsage::GpuOnly, usage | vk::BufferUsageFlagBits::eTransferDst)),
      Type(slot_type) {
    if (!DeviceBuffer->IsMapped()) {
        HostBuffer = std::make_unique<VmaBuffer>(Ctx.Vma, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
    }
    ImplOps = HostBuffer ? Impl{UpdateStaging, ReserveStaging, InsertStaging, EraseStaging} : Impl{UpdateDirect, ReserveDirect, InsertDirect, EraseDirect};
    UpdateDescriptor();
}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, vk::BufferUsageFlags usage, SlotType slot_type)
    : Buffer(ctx, data.size(), usage, slot_type) { Update(data); }

Buffer::Buffer(BufferContext &ctx, vk::DeviceSize size, MemoryUsage mem, vk::BufferUsageFlags usage)
    : Ctx(ctx), Usage(usage), DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, size, mem, usage)),
      ImplOps{UpdateDirect, ReserveDirect, InsertDirect, EraseDirect} {}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, MemoryUsage mem, vk::BufferUsageFlags usage)
    : Ctx(ctx), Usage(usage), DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, data, mem, usage)),
      ImplOps{UpdateDirect, ReserveDirect, InsertDirect, EraseDirect} {}

Buffer::Buffer(Buffer &&other)
    : Ctx(other.Ctx), Slot(other.Slot), UsedSize(other.UsedSize), Usage(other.Usage),
      DeviceBuffer(std::move(other.DeviceBuffer)), HostBuffer(std::move(other.HostBuffer)),
      Type(other.Type), ImplOps(other.ImplOps) {
    other.Slot = InvalidSlot;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        Retire();
        if (Slot != InvalidSlot) Ctx.Slots.Release(Type, Slot);
        Slot = other.Slot;
        UsedSize = other.UsedSize;
        Usage = other.Usage;
        DeviceBuffer = std::move(other.DeviceBuffer);
        HostBuffer = std::move(other.HostBuffer);
        Type = other.Type;
        ImplOps = other.ImplOps;
        other.Slot = InvalidSlot;
    }
    return *this;
}

Buffer::~Buffer() {
    Retire();
    if (Slot != InvalidSlot) Ctx.Slots.Release(Type, Slot);
}

void Buffer::Retire() {
    if (HostBuffer) { Ctx.Reclaimer->Retire(std::move(HostBuffer)); }
    if (DeviceBuffer) { Ctx.Reclaimer->Retire(std::move(DeviceBuffer)); }
    if (Slot != InvalidSlot) { Ctx.CancelDescriptorUpdate(Type, Slot); }
}

void Buffer::UpdateDescriptor() {
    if (Slot == InvalidSlot) return;
    Ctx.AddPendingDescriptorUpdate(Type, Slot, GetDescriptor());
}

bool Buffer::IsMapped() const { return DeviceBuffer && DeviceBuffer->IsMapped(); }
vk::Buffer Buffer::operator*() const { return DeviceBuffer->Get(); }
std::span<const std::byte> Buffer::GetData() const { return DeviceBuffer->GetData(); }
std::span<const std::byte> Buffer::GetMappedData() const {
    if (HostBuffer) return HostBuffer->GetData();
    return DeviceBuffer->GetData();
}
std::span<std::byte> Buffer::GetMappedData() {
    if (HostBuffer) return HostBuffer->GetMappedData();
    return DeviceBuffer->GetMappedData();
}
vk::DeviceSize Buffer::GetAllocatedSize() const { return DeviceBuffer->GetAllocatedSize(); }
void Buffer::Write(std::span<const std::byte> data, vk::DeviceSize offset) const { DeviceBuffer->Write(data, offset); }
void Buffer::Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const { DeviceBuffer->Move(from, to, size); }

void Buffer::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (ImplOps.Update && ImplOps.Update(*this, data, offset)) UpdateDescriptor();
}
void Buffer::Reserve(vk::DeviceSize required_size) {
    if (ImplOps.Reserve && ImplOps.Reserve(*this, required_size)) UpdateDescriptor();
}
void Buffer::Insert(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (ImplOps.Insert) ImplOps.Insert(*this, data, offset);
}
void Buffer::Erase(vk::DeviceSize offset, vk::DeviceSize size) {
    if (ImplOps.Erase) ImplOps.Erase(*this, offset, size);
}
} // namespace mvk
