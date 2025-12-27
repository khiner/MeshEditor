#include "UniqueBuffer.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <print>

namespace mvk {
VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    }
}

struct UniqueBuffer::Impl {
    Impl(VmaAllocator vma, vk::DeviceSize size, mvk::MemoryUsage memory_usage, vk::BufferUsageFlags usage) : Vma(vma) {
        VmaAllocationCreateInfo aci{};
        aci.usage = ToVmaMemoryUsage(memory_usage);
        if (memory_usage == mvk::MemoryUsage::GpuOnly) {
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        // All staging and device buffers act as transfer sources, since buffer updates include a device->device copy.
        vk::BufferCreateInfo bci{{}, size, usage | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
        if (memory_usage == mvk::MemoryUsage::CpuOnly || memory_usage == mvk::MemoryUsage::CpuToGpu) {
            aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        } else {
            bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
        }
        if (vmaCreateBuffer(Vma, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&Handle), &Allocation, &Info) != VK_SUCCESS) {
            throw std::runtime_error("vmaCreateBuffer failed");
        }
        vmaGetMemoryTypeProperties(Vma, Info.memoryType, &MemoryProps);
    }

    Impl(Impl &&other) noexcept
        : Vma(other.Vma),
          Handle(std::exchange(other.Handle, nullptr)),
          Allocation(std::exchange(other.Allocation, nullptr)),
          Info(other.Info) {}

    ~Impl() {
        vmaDestroyBuffer(Vma, Handle, Allocation);
    }

    VmaAllocator Vma;
    vk::Buffer Handle;
    VmaAllocation Allocation;
    VmaAllocationInfo Info;
    VkMemoryPropertyFlags MemoryProps{};
};

UniqueBuffer::UniqueBuffer(VmaAllocator vma, vk::DeviceSize size, mvk::MemoryUsage memory_usage, vk::BufferUsageFlags usage)
    : Imp(std::make_unique<Impl>(vma, size, memory_usage, usage)) {}

UniqueBuffer::UniqueBuffer(VmaAllocator vma, std::span<const std::byte> data, mvk::MemoryUsage memory_usage, vk::BufferUsageFlags usage)
    : Imp(std::make_unique<Impl>(vma, data.size(), memory_usage, usage)) {
    Write(data);
}
UniqueBuffer::UniqueBuffer(UniqueBuffer &&other) : Imp(std::move(other.Imp)) {}

UniqueBuffer::~UniqueBuffer() = default;

UniqueBuffer &UniqueBuffer::operator=(UniqueBuffer &&other) {
    if (this != &other) Imp = std::move(other.Imp);
    return *this;
}

vk::Buffer UniqueBuffer::Get() const { return Imp->Handle; }
std::span<const std::byte> UniqueBuffer::GetData() const { return {static_cast<const std::byte *>(Imp->Info.pMappedData), Imp->Info.size}; }
std::span<std::byte> UniqueBuffer::GetMappedData() const { return {static_cast<std::byte *>(Imp->Info.pMappedData), Imp->Info.size}; }
vk::DeviceSize UniqueBuffer::GetAllocatedSize() const { return Imp->Info.size; }
bool UniqueBuffer::IsMapped() const {
    return Imp->Info.pMappedData != nullptr &&
        (Imp->MemoryProps & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void UniqueBuffer::Write(std::span<const std::byte> src, vk::DeviceSize offset) const {
    if (src.empty() || offset >= GetAllocatedSize()) return;

    std::copy(src.begin(), src.end(), GetMappedData().subspan(offset).data());
}

void UniqueBuffer::Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const {
    const auto allocated_size = GetAllocatedSize();
    if (size == 0 || from + size > allocated_size || to + size > allocated_size) return;

    const auto mapped_data = GetMappedData();
    std::memmove(mapped_data.subspan(to).data(), mapped_data.subspan(from).data(), size_t(size));
}
} // namespace mvk

namespace {
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

namespace mvk {
UniqueVmaAllocator::UniqueVmaAllocator(vk::PhysicalDevice pd, vk::Device d, VkInstance instance) : PhysicalDevice(pd), Device(d) {
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
}

UniqueVmaAllocator::~UniqueVmaAllocator() {
    vmaDestroyAllocator(Vma);
}

VmaAllocator UniqueVmaAllocator::Get() const { return Vma; }

std::string UniqueVmaAllocator::DebugHeapUsage() const {
    const auto budgets = QueryHeapBudgets(Vma, PhysicalDevice);
    std::string result;
    for (uint32_t i = 0; i < budgets.size(); ++i) {
        const auto &b = budgets[i];
        result += std::format(
            "Heap {}/{}:\n"
            "\tAllocations:\n"
            "\t\tCount: {}\n"
            "\t\tBytes: {}\n"
            "\tBlocks:\n"
            "\t\tCount: {}\n"
            "\t\tBytes: {}\n"
            "\tTotal:\n"
            "\t\tUsed: {}\n"
            "\t\tBudget: {}\n",
            i + 1,
            budgets.size(),
            b.statistics.allocationCount,
            FormatBytes(b.statistics.allocationBytes),
            b.statistics.blockCount,
            FormatBytes(b.statistics.blockBytes),
            FormatBytes(b.usage),
            FormatBytes(b.budget)
        );
    }
    return result;
}

} // namespace mvk
