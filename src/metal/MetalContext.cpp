#include "metal/MetalContext.h"
#include "metal/AutoreleaseScope.h"
#include "metal/PhysicalPages.h"

#include "metal/MetalCpp.h"

#include <atomic>
#include <bit>
#include <format>
#include <unordered_set>
#include <utility>

namespace mtl {
namespace {
std::atomic<uint64_t> CommittedCount;

void ObserveCommand(MTL::CommandBuffer *command, std::string_view label, MTL::SharedEvent *mapping, MTL::SharedEvent *execution) {
    command->setLabel(Str(label).get());
    command->addCompletedHandler([mapping = NS::RetainPtr(mapping), execution = NS::RetainPtr(execution)](MTL::CommandBuffer *completed) mutable {
        const AutoreleaseScope pool;
        const auto [mapping_event, execution_event] = std::pair{std::move(mapping), std::move(execution)};
        if (const auto *error = completed->error()) {
            std::fprintf(stderr, "Metal command %s failed (mapping %llu, execution %llu): %s\n", completed->label()->utf8String(), (unsigned long long)mapping_event->signaledValue(), (unsigned long long)execution_event->signaledValue(), error->description()->utf8String());
        }
    });
}
} // namespace
struct Context::MappingSubmission {
    struct Command {
        MTL::Buffer *Destination{};
        MTL::Heap *Heap{};
        std::vector<MTL4::UpdateSparseBufferMappingOperation> Updates;
    };
    std::vector<Mapping> Resources;
    std::vector<Command> Commands;
};

struct Context::MappingRetirement {
    NS::SharedPtr<MTL::CommandBuffer> Readers;
    std::vector<Mapping> Ranges;
    std::vector<NS::SharedPtr<MTL::Buffer>> Addresses;
    std::vector<NS::SharedPtr<MTL::Heap>> Heaps;
    std::vector<MTL4::UpdateSparseBufferMappingOperation> Unmaps;
    // The retirement event value that completes the unmaps, once they are published.
    std::optional<uint64_t> Serial;
};

struct Context::PageRetirement {
    NS::SharedPtr<MTL::CommandBuffer> Readers;
    std::vector<std::vector<std::shared_ptr<PhysicalPage>>> Pages;
};

void Commit(MTL::CommandBuffer *command) {
    command->commit();
    CommittedCount.fetch_add(1u, std::memory_order_relaxed);
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
    auto retirement_event = NS::TransferPtr(device->newSharedEvent());
    if (!mapping_queue || !mapping_event || !execution_event || !retirement_event) throw std::runtime_error("Failed to create the Metal mapping queue.");

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
    RetirementEvent = std::move(retirement_event);
    Residency = std::move(residency);
}

Context::~Context() {
    const AutoreleaseScope pool;
    if (MappingEvent) {
        auto *fence = Queue->commandBuffer();
        fence->encodeSignalEvent(ExecutionEvent.get(), ++ExecutionSerial);
        Commit(fence);
        fence->waitUntilCompleted();
        DrainMappings();
        TrimPageCache(0u);
    }
    if (MappingQueue && Residency) MappingQueue->removeResidencySet(Residency.get());
    MappingQueue.reset();
    if (MappingEvent) CommitResidency();
    SparseAddresses.clear();
    PagePool.reset();
    if (Queue && Residency) Queue->removeResidencySet(Residency.get());
    AutoreleaseScope::Release(MappingSubmissions, MappingRetirements, PageRetirements, PendingMappings, PendingUnmaps, PendingSparseAddresses, PendingPageRetirements, PendingResidentRemovals);
    AutoreleaseScope::Release(RetirementEvent, ExecutionEvent, MappingEvent, Residency, Queue, Device);
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

void Context::CollectCompletedWork() const {
    const AutoreleaseScope pool;
    // A dead alias cannot read its old physical pages. Recycle those pages as
    // soon as readers finish, independently of the virtual cleanup backlog.
    while (!PageRetirements.empty() && PageRetirements.front()->Readers->status() == MTL::CommandBufferStatusCompleted)
        PageRetirements.pop_front();
    const auto completed = MappingEvent->signaledValue();
    // Native resource teardown must not overlap mapping work.
    // Reap as a batch only at an idle point on the mapping timeline.
    // No CPU wait is needed.
    if (completed == MappingSerial && RetirementEvent->signaledValue() == RetirementSerial) {
        MappingSubmissions.clear();
        for (const auto &resource : PendingResidentRemovals) Residency->removeAllocation(resource.get());
        if (!PendingResidentRemovals.empty()) ResidencyDirty = true;
        if (Residency && std::exchange(ResidencyDirty, false)) Residency->commit();
        PendingResidentRemovals.clear();
    }
    while (!MappingRetirements.empty()) {
        const auto &retirement = *MappingRetirements.front();
        if (!retirement.Serial || RetirementEvent->signaledValue() < *retirement.Serial) break;
        for (const auto &buffer : retirement.Addresses) SparseAddresses.erase(buffer.get());
        MappingRetirements.pop_front();
    }
}

void Context::PublishRetirements() const {
    // Retired addresses have no readers once their fence completes, so each such retirement unmaps all of its ranges at once.
    // Mapping and unmapping share one native queue so aliases of the same physical heap never have concurrent page-table updates.
    // Heaps stay owned until the unmaps complete.
    for (auto &retirement : MappingRetirements) {
        if (retirement->Serial) continue;
        if (retirement->Readers->status() != MTL::CommandBufferStatusCompleted) break;
        for (const auto &range : retirement->Ranges)
            retirement->Unmaps.push_back({MTL::SparseTextureMappingModeUnmap, NS::Range{range.First, range.Count}, 0u});
        for (size_t i = 0u; i < retirement->Ranges.size(); ++i)
            MappingQueue->updateBufferMappings(retirement->Ranges[i].Destination.get(), nullptr, &retirement->Unmaps[i], 1u);
        if (!retirement->Unmaps.empty()) MappingQueue->signalEvent(RetirementEvent.get(), ++RetirementSerial);
        retirement->Serial = RetirementSerial;
    }
}

void Context::CommitResidency() const {
    const AutoreleaseScope pool;
    CollectCompletedWork();
    if (Residency && std::exchange(ResidencyDirty, false)) Residency->commit();
    if (!PendingMappings.empty()) {
        auto submission = std::make_shared<MappingSubmission>();
        // Collect each buffer's consecutive heap runs into one native update with multiple mapping operations.
        auto &commands = submission->Commands;
        std::unordered_map<MTL::Buffer *, size_t> last_batch;
        for (const auto &mapping : PendingMappings) {
            const auto destination = mapping.Destination.get();
            auto entry = last_batch.find(destination);
            if (entry == last_batch.end() || commands[entry->second].Heap != mapping.Heap.get()) {
                last_batch[destination] = commands.size();
                commands.push_back({destination, mapping.Heap.get(), {}});
                entry = last_batch.find(destination);
            }
            auto &operations = commands[entry->second].Updates;
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
        submission->Resources = std::move(PendingMappings);
        ++MappingSerial;
        MappingSubmissions.push_back(submission);
        // Mapping updates touch the page tables of resources whose aliases may
        // still be in flight. Order both queues explicitly, including growth
        // and newly allocated ranges, before publishing the new mappings.
        auto *before = Queue->commandBuffer();
        ObserveCommand(before, std::format("Before sparse mapping {} (execution {})", MappingSerial, ExecutionSerial + 1u), MappingEvent.get(), ExecutionEvent.get());
        before->encodeSignalEvent(ExecutionEvent.get(), ++ExecutionSerial);
        Commit(before);
        MappingQueue->wait(ExecutionEvent.get(), ExecutionSerial);
        for (const auto &operation : submission->Commands)
            MappingQueue->updateBufferMappings(operation.Destination, operation.Heap, operation.Updates.data(), operation.Updates.size());
        MappingQueue->signalEvent(MappingEvent.get(), MappingSerial);
        auto *barrier = Queue->commandBuffer();
        uint64_t mapped = 0u;
        for (const auto &mapping : submission->Resources) mapped += mapping.Count;
        ObserveCommand(barrier, std::format("After sparse mapping {} ({} commands, {} pages)", MappingSerial, submission->Commands.size(), mapped), MappingEvent.get(), ExecutionEvent.get());
        barrier->encodeWait(MappingEvent.get(), MappingSerial);
        Commit(barrier);
    }
    if (!PendingUnmaps.empty() || !PendingSparseAddresses.empty() || !PendingPageRetirements.empty()) {
        auto retirement = std::make_unique<MappingRetirement>();
        retirement->Ranges = std::move(PendingUnmaps);
        retirement->Addresses = std::move(PendingSparseAddresses);
        auto pages = std::make_unique<PageRetirement>();
        pages->Pages = std::move(PendingPageRetirements);
        std::unordered_set<MTL::Heap *> heaps;
        for (const auto &batch : pages->Pages)
            for (const auto &page : batch)
                if (heaps.insert(page->Heap()).second) retirement->Heaps.push_back(NS::RetainPtr(page->Heap()));
        retirement->Readers = NS::RetainPtr(Queue->commandBuffer());
        retirement->Readers->encodeSignalEvent(ExecutionEvent.get(), ++ExecutionSerial);
        Commit(retirement->Readers.get());
        pages->Readers = retirement->Readers;
        PageRetirements.push_back(std::move(pages));
        MappingRetirements.push_back(std::move(retirement));
    }
    PublishRetirements();
}

bool Context::DrainMappings() const {
    const AutoreleaseScope pool;
    CommitResidency();
    if (MappingEvent->signaledValue() < MappingSerial) {
        auto *fence = Queue->commandBuffer();
        fence->encodeWait(MappingEvent.get(), MappingSerial);
        Commit(fence);
        fence->waitUntilCompleted();
        if (fence->status() == MTL::CommandBufferStatusError) return false;
    }
    while (!PageRetirements.empty()) {
        auto &retirement = *PageRetirements.front();
        retirement.Readers->waitUntilCompleted();
        if (retirement.Readers->status() == MTL::CommandBufferStatusError) return false;
        CollectCompletedWork();
    }
    // Teardown and explicit memory reclamation may wait for inactive storage.
    // A CPU event wait avoids putting a watchdog-limited GPU wait behind a
    // potentially long native unmap. Normal frames only poll and publish.
    for (;;) {
        CollectCompletedWork();
        if (MappingRetirements.empty()) break;
        auto &retirement = *MappingRetirements.front();
        retirement.Readers->waitUntilCompleted();
        if (retirement.Readers->status() == MTL::CommandBufferStatusError) return false;
        PublishRetirements();
        if (*retirement.Serial && !RetirementEvent->waitUntilSignaledValue(*retirement.Serial, 30000u)) return false;
        CollectCompletedWork();
    }
    return MappingEvent->signaledValue() == MappingSerial;
}

void Context::TrimPageCache(uint64_t keep_bytes) const {
    const AutoreleaseScope pool;
    if (!PagePool || PagePool->CachedBytes() <= keep_bytes) return;
    CommitResidency();
    // Empty slabs can be reused immediately, but their heaps may still back
    // inactive aliases. A frame never waits for that cleanup just to trim cache.
    if (!MappingRetirements.empty() || MappingEvent->signaledValue() != MappingSerial) return;
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

void Context::RetireSparseAddresses(NS::SharedPtr<MTL::Buffer> buffer) const {
    if (buffer) PendingSparseAddresses.push_back(std::move(buffer));
}

void Context::UnmapBufferPages(MTL::Buffer *buffer, uint64_t first, uint64_t count) const {
    if (count) PendingUnmaps.push_back({NS::RetainPtr(buffer), {}, first, count, 0u});
}

void Context::RetainUnmappedPages(std::vector<std::shared_ptr<PhysicalPage>> pages) const {
    if (!pages.empty()) PendingPageRetirements.push_back(std::move(pages));
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
