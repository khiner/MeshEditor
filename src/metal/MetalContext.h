#pragma once

#include "metal/MetalSchema.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Foundation/NSSharedPtr.hpp>

namespace NS {
class String;
} // namespace NS
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
} // namespace MTL4

namespace mtl {
struct PhysicalPagePool;
struct PhysicalPage;
NS::SharedPtr<NS::String> Str(std::string_view);
// Commits the command buffer and counts it in CommittedCommandBuffers.
void Commit(MTL::CommandBuffer *);
// The number of command buffers this process has committed.
uint64_t CommittedCommandBuffers();

// Queue-wide residency includes the physical heaps behind sparse GPU buffers.
// A page mapping update runs while no command buffer in the process is being submitted and completes before the call that published it returns.
// The driver races a mapping update against a submission in progress on another queue.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void AddResident(MTL::Allocation *) const;
    void RemoveResident(MTL::Allocation *) const;
    // Publishes pending mappings, retires the ranges whose readers finished, and commits residency changes.
    void CommitResidency() const;
    // Waits for every retirement's readers and returns whether they all completed without error.
    bool DrainMappings() const;
    // Frame boundary: release empty slabs once no retired alias can reach their heaps.
    void TrimPageCache(uint64_t keep_bytes = 32u << 20) const;
    // Each reservation is a new address object, released once its unmaps complete.
    NS::SharedPtr<MTL::Buffer> ReserveSparseAddresses(uint64_t bytes) const;
    void RetireSparseAddresses(NS::SharedPtr<MTL::Buffer>) const;
    bool OwnsSparseAddresses(MTL::Buffer *) const;
    // Unmaps the range once submitted readers finish.
    void UnmapBufferPages(MTL::Buffer *, uint64_t first, uint64_t count) const;
    // Pages remain owned through submitted readers, which keeps their heaps alive until the unmap.
    void RetainUnmappedPages(std::vector<std::shared_ptr<PhysicalPage>>) const;
    PhysicalPagePool &Pages() const;
    // Explicit ordering for commands that access distinct resource aliases.
    void OrderAfterGpuWork(MTL::CommandBuffer *) const;
    // Advances with every execution-order signal, so a recorder can tell whether GPU work was committed after its command buffer began.
    uint64_t ExecutionSignals() const { return ExecutionSerial; }
    // Arguments are physical pages.
    // CommitResidency maps them after prior Queue work completes.
    void MapBufferPages(MTL::Buffer *, MTL::Heap *, uint64_t first, uint64_t count, uint64_t heap_first) const;

    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL::CommandQueue> Queue;
    NS::SharedPtr<MTL::ResidencySet> Residency;

private:
    struct MappingCommand;
    struct Retirement;
    // Unmaps and releases each retirement whose readers completed.
    void RetireCompleted() const;
    // Runs the mapping commands and returns once they complete.
    void UpdateMappings(const std::vector<MappingCommand> &) const;
    mutable std::shared_ptr<PhysicalPagePool> PagePool;
    mutable bool ResidencyDirty{false};
    mutable std::vector<NS::SharedPtr<MTL::Allocation>> PendingResidentRemovals;
    NS::SharedPtr<MTL4::CommandQueue> MappingQueue;
    // Mapping completion and GPU execution order are signaled on separate timelines.
    NS::SharedPtr<MTL::SharedEvent> MappingEvent, ExecutionEvent;
    // An unmap has no heap.
    struct Mapping {
        NS::SharedPtr<MTL::Buffer> Destination;
        NS::SharedPtr<MTL::Heap> Heap;
        uint64_t First{}, Count{}, HeapFirst{};
    };
    mutable std::vector<Mapping> PendingMappings, PendingUnmaps;
    mutable std::deque<std::unique_ptr<Retirement>> Retirements;
    mutable std::vector<NS::SharedPtr<MTL::Buffer>> PendingSparseAddresses;
    mutable std::unordered_map<MTL::Buffer *, NS::SharedPtr<MTL::Buffer>> SparseAddresses;
    mutable std::vector<std::vector<std::shared_ptr<PhysicalPage>>> PendingPageRetirements;
    mutable uint64_t MappingSerial{}, ExecutionSerial{};
};
} // namespace mtl
