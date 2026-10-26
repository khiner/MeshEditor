#include "metal/SparseBuffer.h"
#include "Profile.h"
#include "Range.h"

#include "metal/MetalCpp.h"

#include <bit>
#include <format>
#include <mach/mach_vm.h>

namespace mtl {
namespace {
void CheckVm(kern_return_t result, const char *operation) {
    if (result != KERN_SUCCESS) throw std::runtime_error(std::format("{}: {}", operation, mach_error_string(result)));
}

void ForEachPhysicalRun(std::span<const PhysicalPageRef> pages, auto &&visit) {
    for (size_t first = 0; first < pages.size();) {
        auto end = first + 1;
        while (end < pages.size() && pages[end]->Heap() == pages[first]->Heap() &&
               pages[end]->Offset() == pages[first]->Offset() + (end - first) * PhysicalPageBytes) ++end;
        visit(first, end - first);
        first = end;
    }
}
} // namespace

SparseBuffer::SparseBuffer(const Context &ctx, uint64_t virtual_bytes) : Ctx(ctx), CpuVirtualBytes(std::bit_floor(uint64_t(ctx.Device->maxBufferLength()))) {
    const profile::CpuScope scope{"SparseBufferCreate"};
    // Virtual reservation allocates no geometry pages.
    // Physical storage grows on demand up to the device's per-resource address limit.
    mach_vm_address_t address{};
    CheckVm(mach_vm_allocate(mach_task_self(), &address, CpuVirtualBytes, VM_FLAGS_ANYWHERE), "Reserve CPU arena addresses");
    CpuAddress = address;
    try {
        ReserveAddresses(virtual_bytes);
    } catch (...) {
        mach_vm_deallocate(mach_task_self(), CpuAddress, CpuVirtualBytes);
        throw;
    }
}

SparseBuffer::SparseBuffer(const Context &ctx, uint64_t virtual_bytes, std::span<const uint32_t> indices, std::span<const PhysicalPageRef> pages)
    : Ctx(ctx), Gpu(ctx.ReserveSparseAddresses(std::bit_ceil(virtual_bytes))), VirtualBytes(Gpu->length()),
      Pages(pages.begin(), pages.end()), CloneIndices(indices.begin(), indices.end()) {
    ForEachIndexRun(indices, [&](size_t first, size_t count) { MapGpu(indices[first], pages.subspan(first, count)); });
}

SparseBuffer::~SparseBuffer() {
    if (CpuAddress) mach_vm_deallocate(mach_task_self(), CpuAddress, CpuVirtualBytes);
    if (!Gpu) return;
    std::vector<std::pair<uint64_t, uint64_t>> mapped;
    if (!CloneIndices.empty()) ForEachIndexRun(CloneIndices, [&](size_t first, size_t count) { mapped.emplace_back(CloneIndices[first], count); });
    else if (const auto count = std::min(uint64_t(Gpu->length()), ResidentBytes) / PhysicalPageBytes) mapped.emplace_back(0u, count);
    Ctx.RetireSparseAddresses(std::move(Gpu), std::move(mapped), std::move(Pages));
}

void SparseBuffer::ReserveAddresses(uint64_t bytes) {
    if (bytes <= VirtualBytes) return;
    if (bytes > CpuVirtualBytes) throw std::runtime_error("Metal buffer address space exhausted.");
    const auto size = Gpu ? CpuVirtualBytes : std::min(CpuVirtualBytes, std::max(std::bit_ceil(bytes), FirstGpuAddressBytes));
    auto next = Ctx.ReserveSparseAddresses(size);
    if (Gpu) {
        // Committed commands can still read the old address, and the mapping update that unmaps it waits for them to complete.
        // Later writes and history restores use the new address through the updated bindless slot.
        std::vector<std::pair<uint64_t, uint64_t>> mapped;
        if (ResidentBytes) mapped.emplace_back(0u, ResidentBytes / PhysicalPageBytes);
        Ctx.RetireSparseAddresses(std::move(Gpu), std::move(mapped), {});
    }
    Gpu = std::move(next);
    VirtualBytes = size;
    MapGpu(0u, Pages);
}

void SparseBuffer::MapCpu(uint64_t first, std::span<const PhysicalPageRef> pages) { // NOLINT(readability-make-member-function-const)
    ForEachPhysicalRun(pages, [&](size_t i, size_t count) {
        mach_vm_address_t destination = CpuAddress + (first + i) * PhysicalPageBytes;
        vm_prot_t current{}, maximum{};
        CheckVm(mach_vm_remap(mach_task_self(), &destination, count * PhysicalPageBytes, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(), reinterpret_cast<mach_vm_address_t>(pages[i]->Contents().data()), false, &current, &maximum, VM_INHERIT_NONE), "Map canonical CPU arena pages");
    });
}

void SparseBuffer::MapGpu(uint64_t first, std::span<const PhysicalPageRef> pages) {
    const auto capacity = Gpu->length() / PhysicalPageBytes;
    if (first >= capacity) return;
    pages = pages.first(std::min(uint64_t(pages.size()), capacity - first));
    ForEachPhysicalRun(pages, [&](size_t i, size_t count) {
        Ctx.MapBufferPages(Gpu.get(), pages[i]->Heap(), first + i, count, pages[i]->Offset() / PhysicalPageBytes);
    });
}

void SparseBuffer::Reserve(uint64_t bytes) {
    const profile::CpuScope scope{"SparseBufferReserve"};
    if (bytes <= ResidentBytes) return;
    if (!CloneIndices.empty()) throw std::logic_error("A page clone is read-only.");
    if (bytes > CpuVirtualBytes) throw std::runtime_error("Metal buffer address space exhausted.");
    const auto end = (bytes + PhysicalPageBytes - 1) & ~(PhysicalPageBytes - 1);
    ReserveAddresses(end);
    auto pages = Ctx.Pages().Allocate(uint32_t((end - ResidentBytes) / PhysicalPageBytes));
    const auto first = Pages.size();
    try {
        Pages.insert(Pages.end(), pages.begin(), pages.end());
        MapCpu(first, pages);
        MapGpu(first, pages);
    } catch (...) {
        Pages.resize(first);
        throw;
    }
    ResidentBytes = end;
}

PhysicalPageRef SparseBuffer::RetainPage(uint64_t index) const { return Pages.at(index); }

} // namespace mtl
