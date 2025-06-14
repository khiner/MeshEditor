#include "BufferAllocator.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <print>
#include <unordered_map>

namespace {
struct VmaBuffer {
    VmaBuffer(VmaAllocator vma, VmaAllocationCreateInfo aci, vk::BufferCreateInfo bci) : Vma(vma) {
        if (vmaCreateBuffer(Vma, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&Handle), &Allocation, &Info) != VK_SUCCESS) {
            throw std::runtime_error("vmaCreateBuffer failed");
        }
    }

    VmaBuffer(VmaBuffer &&other) noexcept
        : Vma(other.Vma),
          Handle(std::exchange(other.Handle, nullptr)),
          Allocation(std::exchange(other.Allocation, nullptr)),
          Info(other.Info) {}

    VmaBuffer &operator=(VmaBuffer &&) = delete;

    ~VmaBuffer() {
        vmaDestroyBuffer(Vma, Handle, Allocation);
    }

    vk::Buffer operator*() const { return Handle; }

    VmaAllocator Vma;
    vk::Buffer Handle;
    VmaAllocation Allocation;
    VmaAllocationInfo Info;
};

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

using BufferByHandle = std::unordered_map<VkBuffer, VmaBuffer>;
std::unordered_map<VmaAllocator, BufferByHandle> BuffersByAllocator;
const VmaAllocationInfo &GetAllocationInfo(VmaAllocator vma, vk::Buffer b) { return BuffersByAllocator.at(vma).at(static_cast<VkBuffer>(b)).Info; }
} // namespace

// vmaDestroyBuffer(Vma, Buffer, Allocation->Allocation);
namespace mvk {
BufferAllocator::BufferAllocator(vk::PhysicalDevice pd, vk::Device d, VkInstance instance) {
    VmaAllocatorCreateInfo create_info{};
    create_info.physicalDevice = pd;
    create_info.device = d;
    create_info.instance = instance;
#ifndef RELEASE_BUILD
    VmaDeviceMemoryCallbacks memory_callbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    create_info.pDeviceMemoryCallbacks = &memory_callbacks;
#endif

    if (vmaCreateAllocator(&create_info, &Vma) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator.");
    }
    BuffersByAllocator.emplace(Vma, BufferByHandle{});
}

BufferAllocator::~BufferAllocator() {
    BuffersByAllocator.erase(Vma);
    vmaDestroyAllocator(Vma);
}

std::span<std::byte> BufferAllocator::GetMappedData(vk::Buffer b) const {
    auto &info = GetAllocationInfo(Vma, b);
    return {static_cast<std::byte *>(info.pMappedData), info.size};
}
std::span<const std::byte> BufferAllocator::GetData(vk::Buffer b) const {
    const auto &info = GetAllocationInfo(Vma, b);
    return {static_cast<const std::byte *>(info.pMappedData), info.size};
}
vk::DeviceSize BufferAllocator::GetAllocatedSize(vk::Buffer b) const { return GetAllocationInfo(Vma, b).size; }

void BufferAllocator::Write(vk::Buffer b, std::span<const std::byte> src, vk::DeviceSize offset) const {
    if (src.empty() || offset >= GetAllocatedSize(b)) return;

    std::copy(src.begin(), src.end(), GetMappedData(b).subspan(offset).data());
}

void BufferAllocator::Move(vk::Buffer b, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const {
    const auto allocated_size = GetAllocatedSize(b);
    if (size == 0 || from + size > allocated_size || to + size > allocated_size) return;

    auto mapped_data = GetMappedData(b);
    std::memmove(mapped_data.subspan(to).data(), mapped_data.subspan(from).data(), size_t(size));
}

namespace {
VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    }
}
} // namespace

vk::Buffer BufferAllocator::Allocate(vk::DeviceSize size, MemoryUsage memory_usage, vk::BufferUsageFlags usage) const {
    VmaAllocationCreateInfo aci{};
    aci.usage = ToVmaMemoryUsage(memory_usage);
    // All staging and device buffers act as transfer sources, since buffer updates include a device->device copy.
    vk::BufferCreateInfo bci{{}, size, usage | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
    if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
        aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
    }
    VmaBuffer vma_buffer{Vma, aci, bci};
    auto vk_buffer = *vma_buffer;
    BuffersByAllocator.at(Vma).emplace(static_cast<VkBuffer>(vk_buffer), std::move(vma_buffer));
    return vk_buffer;
}

void BufferAllocator::Destroy(vk::Buffer b) const { BuffersByAllocator.at(Vma).erase(static_cast<VkBuffer>(b)); }

std::string BufferAllocator::DebugHeapUsage(vk::PhysicalDevice pd) const {
    const auto budgets = QueryHeapBudgets(Vma, pd);
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
