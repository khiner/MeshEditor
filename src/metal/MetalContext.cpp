#include "metal/MetalContext.h"
#include "metal/AutoreleaseScope.h"
#include "metal/PhysicalPages.h"

#include "metal/MetalCpp.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <mutex>

namespace mtl {
namespace {
std::atomic<uint64_t> CommittedCount;
// The last command buffer committed on each queue, until a mapping update waits for it to complete.
std::mutex CommitMutex;
auto &LastCommitted() {
    static auto *const commands = new std::vector<std::pair<MTL::CommandQueue *, NS::SharedPtr<MTL::CommandBuffer>>>;
    return *commands;
}

// Returns once every queue has completed its committed command buffers, so no submission is in flight and no GPU work reads a sparse page.
void WaitCompleted() {
    const AutoreleaseScope pool;
    std::vector<std::pair<MTL::CommandQueue *, NS::SharedPtr<MTL::CommandBuffer>>> pending;
    {
        const std::scoped_lock lock{CommitMutex};
        pending = std::exchange(LastCommitted(), {});
    }
    for (const auto &[queue, command] : pending) command->waitUntilCompleted();
}
} // namespace

struct Context::MappingCommand {
    MTL::Buffer *Destination{};
    MTL::Heap *Heap{};
    std::vector<MTL4::UpdateSparseBufferMappingOperation> Operations;
};

void Commit(MTL::CommandBuffer *command) {
    command->commit();
    CommittedCount.fetch_add(1u, std::memory_order_relaxed);
    const std::scoped_lock lock{CommitMutex};
    auto &committed = LastCommitted();
    auto *const queue = command->commandQueue();
    if (const auto entry = std::ranges::find(committed, queue, &std::pair<MTL::CommandQueue *, NS::SharedPtr<MTL::CommandBuffer>>::first); entry != committed.end()) entry->second = NS::RetainPtr(command);
    else committed.emplace_back(queue, NS::RetainPtr(command));
}

uint64_t CommittedCommandBuffers() { return CommittedCount.load(std::memory_order_relaxed); }

NS::SharedPtr<NS::String> Str(std::string_view s) {
    const AutoreleaseScope pool;
    return NS::TransferPtr(NS::String::alloc()->init(std::string{s}.c_str(), NS::UTF8StringEncoding));
}

Context::Context() {
    const AutoreleaseScope pool;
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) throw std::runtime_error("No Metal device.");
    if (!device->hasUnifiedMemory()) throw std::runtime_error("MeshEditor targets unified-memory Apple Silicon.");
    if (!device->supportsPlacementSparse()) throw std::runtime_error("MeshEditor requires Metal placement sparse buffers (macOS 26.4 or later, Apple M2 or later).");
    if (device->argumentBuffersSupport() < MTL::ArgumentBuffersTier2) {
        throw std::runtime_error("The bindless argument buffer needs argument buffer tier 2.");
    }

    auto queue = NS::TransferPtr(device->newCommandQueue());
    if (!queue) throw std::runtime_error("Failed to create a Metal command queue.");
    auto mapping_queue = NS::TransferPtr(device->newMTL4CommandQueue());
    auto mapping_event = NS::TransferPtr(device->newSharedEvent());
    auto execution_event = NS::TransferPtr(device->newSharedEvent());
    if (!mapping_queue || !mapping_event || !execution_event) throw std::runtime_error("Failed to create the Metal mapping queue.");

    const auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    auto residency = NS::TransferPtr(device->newResidencySet(descriptor.get(), nullptr));
    if (!residency) throw std::runtime_error("Failed to create the residency set for canonical Metal pages.");
    residency->commit();
    queue->addResidencySet(residency.get());
    mapping_queue->addResidencySet(residency.get());
    Device = std::move(device);
    Queue = std::move(queue);
    MappingQueue = std::move(mapping_queue);
    MappingEvent = std::move(mapping_event);
    ExecutionEvent = std::move(execution_event);
    Residency = std::move(residency);
}

Context::~Context() {
    const AutoreleaseScope pool;
    if (MappingEvent) {
        DrainMappings();
        TrimPageCache(0u);
        CommitResidency();
        WaitCompleted();
        UnmappedAddresses.clear();
    }
    if (Queue) {
        const std::scoped_lock lock{CommitMutex};
        std::erase_if(LastCommitted(), [&](const auto &entry) { return entry.first == Queue.get(); });
    }
    if (MappingQueue && Residency) MappingQueue->removeResidencySet(Residency.get());
    MappingQueue.reset();
    SparseAddresses.clear();
    PagePool.reset();
    if (Queue && Residency) Queue->removeResidencySet(Residency.get());
    AutoreleaseScope::Release(PendingMappings, PendingRetired, UnmappedAddresses, PendingResidentRemovals);
    AutoreleaseScope::Release(ExecutionEvent, MappingEvent, Residency, Queue, Device);
}

void Context::AddResident(MTL::Allocation *resource) const {
    const AutoreleaseScope pool;
    // A membership change recommits the whole set, so a resident allocation leaves it untouched.
    if (!Residency || !resource || Residency->containsAllocation(resource)) return;
    Residency->addAllocation(resource);
    ResidencyDirty = true;
}

void Context::RemoveResident(MTL::Allocation *resource) const {
    if (!Residency || !resource) return;
    PendingResidentRemovals.push_back(NS::RetainPtr(resource));
}

void Context::UpdateMappings(const std::vector<MappingCommand> &commands) const {
    if (commands.empty()) return;
    WaitCompleted();
    UnmappedAddresses.clear();
    for (const auto &command : commands) {
        MappingQueue->updateBufferMappings(command.Destination, command.Heap, command.Operations.data(), command.Operations.size());
    }
    MappingQueue->signalEvent(MappingEvent.get(), ++MappingSerial);
    if (!MappingEvent->waitUntilSignaledValue(MappingSerial, 30000u)) throw std::runtime_error("A Metal sparse mapping update did not complete.");
}

void Context::UpdatePendingMappings() const {
    // Collect each buffer's consecutive heap runs into one native update with multiple mapping operations.
    std::vector<MappingCommand> commands;
    std::unordered_map<MTL::Buffer *, size_t> last_batch;
    for (const auto &mapping : PendingMappings) {
        auto *const destination = mapping.Destination.get();
        auto entry = last_batch.find(destination);
        if (entry == last_batch.end() || commands[entry->second].Heap != mapping.Heap.get()) {
            last_batch[destination] = commands.size();
            commands.push_back({destination, mapping.Heap.get(), {}});
            entry = last_batch.find(destination);
        }
        auto &operations = commands[entry->second].Operations;
        for (uint64_t offset = 0u; offset < mapping.Count;) {
            const auto first = mapping.First + offset, heap_first = mapping.HeapFirst + offset;
            if (!operations.empty()) {
                auto &last = operations.back();
                if (last.bufferRange.length < PhysicalSlabPages && last.bufferRange.location + last.bufferRange.length == first &&
                    last.heapOffset + last.bufferRange.length == heap_first) {
                    const auto count = std::min(uint64_t(PhysicalSlabPages) - last.bufferRange.length, mapping.Count - offset);
                    last.bufferRange.length += count;
                    offset += count;
                    continue;
                }
            }
            const auto count = std::min(uint64_t(PhysicalSlabPages), mapping.Count - offset);
            operations.push_back({MTL::SparseTextureMappingModeMap, NS::Range{first, count}, heap_first});
            offset += count;
        }
    }
    // Unmaps follow the maps, so a retired address with a pending map ends unmapped.
    for (const auto &retired : PendingRetired) {
        if (retired.Mapped.empty()) continue;
        auto &command = commands.emplace_back(MappingCommand{retired.Addresses.get(), nullptr, {}});
        for (const auto [first, count] : retired.Mapped) command.Operations.push_back({MTL::SparseTextureMappingModeUnmap, NS::Range{first, count}, 0u});
    }
    UpdateMappings(commands);
    PendingMappings.clear();
    for (auto &retired : PendingRetired) {
        SparseAddresses.erase(retired.Addresses.get());
        UnmappedAddresses.push_back(std::move(retired.Addresses));
    }
    PendingRetired.clear();
}

void Context::CommitResidency() const {
    const AutoreleaseScope pool;
    for (const auto &resource : PendingResidentRemovals) Residency->removeAllocation(resource.get());
    if (!PendingResidentRemovals.empty()) {
        PendingResidentRemovals.clear();
        ResidencyDirty = true;
    }
    if (Residency && std::exchange(ResidencyDirty, false)) {
        WaitCompleted();
        Residency->commit();
    }
    if (!PendingMappings.empty()) UpdatePendingMappings();
}

void Context::DrainMappings() const {
    const AutoreleaseScope pool;
    CommitResidency();
    if (!PendingRetired.empty()) UpdatePendingMappings();
}

void Context::TrimPageCache(uint64_t keep_bytes) const {
    const AutoreleaseScope pool;
    uint64_t retired_bytes = 0u;
    for (const auto &retired : PendingRetired) retired_bytes += retired.Pages.size() * PhysicalPageBytes;
    // A pending unmap can refer to the pages of an empty slab, so the unmaps run before its heap releases.
    const bool trim = PagePool && PagePool->CachedBytes() > keep_bytes;
    if (trim || retired_bytes > keep_bytes) DrainMappings();
    if (!trim) return;
    auto *fence = Queue->commandBuffer();
    OrderAfterGpuWork(fence);
    Commit(fence);
    fence->waitUntilCompleted();
    PagePool->TrimCache(keep_bytes);
    CommitResidency();
}

NS::SharedPtr<MTL::Buffer> Context::ReserveSparseAddresses(uint64_t bytes) const {
    const AutoreleaseScope pool;
    if (!std::has_single_bit(bytes)) throw std::invalid_argument("Sparse address reservations require a power-of-two size.");
    auto buffer = NS::TransferPtr(Device->newBuffer(bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256));
    if (!buffer) throw std::runtime_error("Failed to reserve sparse Metal addresses.");
    SparseAddresses.emplace(buffer.get(), buffer);
    return buffer;
}

bool Context::OwnsSparseAddresses(MTL::Buffer *buffer) const { return SparseAddresses.contains(buffer); }

void Context::RetireSparseAddresses(NS::SharedPtr<MTL::Buffer> buffer, std::vector<std::pair<uint64_t, uint64_t>> mapped, std::vector<std::shared_ptr<PhysicalPage>> pages) const {
    if (!buffer) return;
    PendingRetired.push_back({std::move(buffer), std::move(mapped), std::move(pages)});
}

PhysicalPagePool &Context::Pages() const {
    if (!PagePool) PagePool = std::make_shared<PhysicalPagePool>(*this);
    return *PagePool;
}

void Context::OrderAfterGpuWork(MTL::CommandBuffer *next) const {
    const AutoreleaseScope pool;
    CommitResidency();
    auto *barrier = Queue->commandBuffer();
    barrier->encodeSignalEvent(ExecutionEvent.get(), ++ExecutionSerial);
    Commit(barrier);
    next->encodeWait(ExecutionEvent.get(), ExecutionSerial);
}

void Context::MapBufferPages(MTL::Buffer *buffer, MTL::Heap *heap, uint64_t first, uint64_t count, uint64_t heap_first) const {
    if (!count) return;
    if (!PendingMappings.empty()) {
        auto &last = PendingMappings.back();
        if (last.Destination.get() == buffer && last.Heap.get() == heap && last.First + last.Count == first && last.HeapFirst + last.Count == heap_first) {
            last.Count += count;
            return;
        }
    }
    PendingMappings.push_back({NS::RetainPtr(buffer), NS::RetainPtr(heap), first, count, heap_first});
}
} // namespace mtl
