#pragma once

#include <deque>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Foundation/NSSharedPtr.hpp>
#include <string_view>

namespace NS {
class String;
}
namespace MTL {
class Allocation;
class Device;
class CommandQueue;
class ResidencySet;
class Buffer;
class CommandBuffer;
class Heap;
class SharedEvent;
} // namespace MTL

namespace MTL4 {
class CommandQueue;
}

namespace mtl {
struct PhysicalPagePool;
struct PhysicalPage;
NS::SharedPtr<NS::String> Str(std::string_view);

// Queue-wide residency includes the physical heaps behind sparse GPU buffers.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void AddResident(MTL::Allocation *) const;
    void RemoveResident(MTL::Allocation *) const;
    void CommitResidency() const;
    // Teardown/draining only. Normal publication and retirement are asynchronous.
    // Retired resources never add a wait to live-geometry submissions.
    bool DrainMappings() const;
    // Frame boundary: collect empty slabs only when native mappings are idle.
    void TrimPageCache(uint64_t keep_bytes = 32u << 20) const;
    // Each reservation is a new address object.
    // A retired address is released once its unmaps complete.
    // Mapping a reissued address while unmaps of other addresses are in flight loses some of its new pages.
    NS::SharedPtr<MTL::Buffer> ReserveSparseAddresses(uint64_t bytes) const;
    void RetireSparseAddresses(NS::SharedPtr<MTL::Buffer>) const;
    bool OwnsSparseAddresses(MTL::Buffer *) const;
    // Retire a range after submitted readers finish.
    // Each completed retirement unmaps in one background batch, and its addresses remain owned until the batch completes.
    void UnmapBufferPages(MTL::Buffer *, uint64_t first, uint64_t count) const;
    // Pages remain owned through submitted readers.
    // Their heaps remain owned through native unmapping.
    // Unused aliases do not prevent physical reuse.
    void RetainUnmappedPages(std::vector<std::shared_ptr<PhysicalPage>>) const;
    PhysicalPagePool &Pages() const;
    // Explicit ordering for commands that access distinct resource aliases.
    void OrderAfterGpuWork(MTL::CommandBuffer *) const;
    // Advances with every execution-order signal, so a recorder can tell whether GPU work was committed after its command buffer began.
    uint64_t ExecutionSignals() const { return ExecutionSerial; }
    // Arguments are physical pages. CommitResidency orders mappings after prior
    // Queue work and before subsequent consumers on Queue.
    void MapBufferPages(MTL::Buffer *, MTL::Heap *, uint64_t first, uint64_t count, uint64_t heap_first) const;

    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL::CommandQueue> Queue;
    NS::SharedPtr<MTL::ResidencySet> Residency;

private:
    // Recycle completed GPU ownership without publishing pending mappings.
    void CollectCompletedWork() const;
    void PublishRetirements() const;
    mutable std::shared_ptr<PhysicalPagePool> PagePool;
    mutable bool ResidencyDirty{false};
    mutable std::vector<NS::SharedPtr<MTL::Allocation>> PendingResidentRemovals;
    NS::SharedPtr<MTL4::CommandQueue> MappingQueue;
    // Separate event values identify mapping publication, GPU execution, and
    // retired-address cleanup on their respective timelines.
    NS::SharedPtr<MTL::SharedEvent> MappingEvent, ExecutionEvent, RetirementEvent;
    // An unmap has no heap.
    struct Mapping {
        NS::SharedPtr<MTL::Buffer> Destination;
        NS::SharedPtr<MTL::Heap> Heap;
        uint64_t First{}, Count{}, HeapFirst{};
    };
    struct MappingSubmission;
    struct MappingRetirement;
    struct PageRetirement;
    mutable std::vector<std::shared_ptr<MappingSubmission>> MappingSubmissions;
    mutable std::vector<Mapping> PendingMappings, PendingUnmaps;
    mutable std::deque<std::unique_ptr<MappingRetirement>> MappingRetirements;
    mutable std::deque<std::unique_ptr<PageRetirement>> PageRetirements;
    mutable std::vector<NS::SharedPtr<MTL::Buffer>> PendingSparseAddresses;
    mutable std::unordered_map<MTL::Buffer *, NS::SharedPtr<MTL::Buffer>> SparseAddresses;
    mutable std::vector<std::vector<std::shared_ptr<PhysicalPage>>> PendingPageRetirements;
    mutable uint64_t MappingSerial{}, ExecutionSerial{}, RetirementSerial{};
};
} // namespace mtl
