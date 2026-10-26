#pragma once

#include "metal/MetalSchema.h"

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
// A page mapping update or a residency set commit runs after every committed command buffer completes, and a mapping update completes before the call that published it returns.
// A retired address unmaps in the next mapping update, and the update after that releases it, so a command buffer that declared it has completed first.
// A retired address's pages return to the pool when its unmap has run, so no page is mapped in two addresses at once.
// A mapping update that overlaps a submission in progress on another queue panics the kernel, and a mapping queue wait on an event counts as an update in progress.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void AddResident(MTL::Allocation *) const;
    void RemoveResident(MTL::Allocation *) const;
    // Commits residency changes, and with pending maps runs one mapping update with them and every pending unmap.
    void CommitResidency() const;
    // Runs every pending map and unmap.
    void DrainMappings() const;
    // Frame boundary: runs the pending unmaps once the pages they hold exceed keep_bytes, and runs them all before it releases the empty slabs beyond keep_bytes.
    void TrimPageCache(uint64_t keep_bytes = 32u << 20) const;
    // Each reservation is a new address object, released by the mapping update after the one that unmaps it.
    NS::SharedPtr<MTL::Buffer> ReserveSparseAddresses(uint64_t bytes) const;
    // Unmaps the (first page, page count) runs of the addresses in the next mapping update, after which the pages return to the pool.
    // The mapping update after that releases the addresses.
    void RetireSparseAddresses(NS::SharedPtr<MTL::Buffer>, std::vector<std::pair<uint64_t, uint64_t>> mapped, std::vector<std::shared_ptr<PhysicalPage>> pages) const;
    bool OwnsSparseAddresses(MTL::Buffer *) const;
    PhysicalPagePool &Pages() const;
    // Explicit ordering for commands that access distinct resource aliases.
    void OrderAfterGpuWork(MTL::CommandBuffer *) const;
    // Advances with every execution-order signal, so a recorder can tell whether GPU work was committed after its command buffer began.
    uint64_t ExecutionSignals() const { return ExecutionSerial; }
    // Arguments are physical pages.
    // CommitResidency maps them after every committed command buffer completes.
    void MapBufferPages(MTL::Buffer *, MTL::Heap *, uint64_t first, uint64_t count, uint64_t heap_first) const;

    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL::CommandQueue> Queue;
    NS::SharedPtr<MTL::ResidencySet> Residency;

private:
    struct MappingCommand;
    // Runs the mapping commands and returns once they complete.
    void UpdateMappings(const std::vector<MappingCommand> &) const;
    // Runs one mapping update with every pending map and unmap, then returns the retired pages to the pool and holds the retired addresses until the next update.
    void UpdatePendingMappings() const;
    mutable std::shared_ptr<PhysicalPagePool> PagePool;
    mutable bool ResidencyDirty{false};
    mutable std::vector<NS::SharedPtr<MTL::Allocation>> PendingResidentRemovals;
    NS::SharedPtr<MTL4::CommandQueue> MappingQueue;
    // Mapping completion and GPU execution order are signaled on separate timelines.
    NS::SharedPtr<MTL::SharedEvent> MappingEvent, ExecutionEvent;
    struct Mapping {
        NS::SharedPtr<MTL::Buffer> Destination;
        NS::SharedPtr<MTL::Heap> Heap;
        uint64_t First{}, Count{}, HeapFirst{};
    };
    // Retired addresses with the page runs they map, and the pages behind those runs until the unmap runs.
    struct Retired {
        NS::SharedPtr<MTL::Buffer> Addresses;
        std::vector<std::pair<uint64_t, uint64_t>> Mapped;
        std::vector<std::shared_ptr<PhysicalPage>> Pages;
    };
    mutable std::vector<Mapping> PendingMappings;
    mutable std::vector<Retired> PendingRetired;
    // Unmapped addresses that a command buffer encoded before their unmap can still declare, released once every command buffer committed since has completed.
    mutable std::vector<NS::SharedPtr<MTL::Buffer>> UnmappedAddresses;
    mutable std::unordered_map<MTL::Buffer *, NS::SharedPtr<MTL::Buffer>> SparseAddresses;
    mutable uint64_t MappingSerial{}, ExecutionSerial{};
};
} // namespace mtl
