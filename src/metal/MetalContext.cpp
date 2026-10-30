#include "metal/MetalContext.h"
#include "metal/AutoreleaseScope.h"
#include "metal/Command.h"
#include "metal/PhysicalPages.h"

#include "metal/MetalCpp.h"

#include <algorithm>
#include <bit>
#include <format>

namespace mtl {
namespace {
// Upload storage grows in chunks, and a larger upload gets a chunk of its own.
constexpr uint64_t UploadChunkBytes{1u << 20};
constexpr uint64_t UploadAlignment{16u};
constexpr uint32_t ArgumentBufferBindings{8u}, ArgumentTextureBindings{4u}, ArgumentSamplerBindings{1u};
constexpr uint64_t WaitTimeoutMs{30000u};

NS::SharedPtr<MTL::Buffer> NewSharedBuffer(const Context &ctx, uint64_t bytes) {
    const AutoreleaseScope pool;
    auto handle = NS::TransferPtr(ctx.Device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!handle) throw std::runtime_error("Failed to allocate a Metal buffer.");
    return handle;
}
} // namespace

template<typename T> Resident<T>::Resident() = default;
template<typename T> Resident<T>::Resident(const Context &ctx, NS::SharedPtr<T> handle) : Handle(std::move(handle)), Ctx(&ctx) {
    Ctx->AddResident(Handle.get());
}
template<typename T> Resident<T>::Resident(Resident &&other) noexcept : Handle(std::move(other.Handle)), Ctx(std::exchange(other.Ctx, nullptr)) {}
template<typename T> Resident<T> &Resident<T>::operator=(Resident &&other) noexcept {
    if (this == &other) return *this;
    const AutoreleaseScope pool;
    if (Ctx && Handle) Ctx->RemoveResident(Handle.get());
    Handle = std::move(other.Handle);
    Ctx = std::exchange(other.Ctx, nullptr);
    return *this;
}
template<typename T> Resident<T>::~Resident() {
    const AutoreleaseScope pool;
    if (Ctx && Handle) Ctx->RemoveResident(Handle.get());
    Handle.reset();
}
template struct Resident<MTL::Buffer>;
template struct Resident<MTL::Texture>;

ResidentBuffer::ResidentBuffer(const Context &ctx, uint64_t bytes) : Resident(ctx, NewSharedBuffer(ctx, bytes)) {}
uint64_t ResidentBuffer::Address() const { return Handle->gpuAddress(); }
uint64_t ResidentBuffer::Length() const { return Handle->length(); }
std::span<std::byte> ResidentBuffer::Contents() const { return {static_cast<std::byte *>(Handle->contents()), Handle->length()}; }

struct Context::MappingCommand {
    MTL::Buffer *Destination{};
    MTL::Heap *Heap{};
    std::vector<MTL4::UpdateSparseBufferMappingOperation> Operations;
};

NS::SharedPtr<NS::String> Str(std::string_view s) {
    const AutoreleaseScope pool;
    return NS::TransferPtr(NS::String::alloc()->init(std::string{s}.c_str(), NS::UTF8StringEncoding));
}

Context::Context() {
    const AutoreleaseScope pool;
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) throw std::runtime_error("No Metal device.");
    if (!device->hasUnifiedMemory()) throw std::runtime_error("MeshEditor targets unified-memory Apple Silicon.");
    if (!device->supportsFamily(MTL::GPUFamilyMetal4)) throw std::runtime_error("MeshEditor renders with Metal 4, which this device does not support.");
    if (!device->supportsPlacementSparse()) throw std::runtime_error("MeshEditor requires Metal placement sparse buffers (macOS 26.4 or later, Apple M2 or later).");
    if (device->argumentBuffersSupport() < MTL::ArgumentBuffersTier2) {
        throw std::runtime_error("The bindless argument buffer needs argument buffer tier 2.");
    }

    auto queue = NS::TransferPtr(device->newMTL4CommandQueue());
    auto mapping_event = NS::TransferPtr(device->newSharedEvent());
    auto completion_event = NS::TransferPtr(device->newSharedEvent());
    if (!queue || !mapping_event || !completion_event) throw std::runtime_error("Failed to create the Metal command queue.");

    const auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    auto residency = NS::TransferPtr(device->newResidencySet(descriptor.get(), nullptr));
    if (!residency) throw std::runtime_error("Failed to create the residency set for canonical Metal pages.");
    residency->commit();
    queue->addResidencySet(residency.get());
    Device = std::move(device);
    Queue = std::move(queue);
    MappingEvent = std::move(mapping_event);
    CompletionEvent = std::move(completion_event);
    Residency = std::move(residency);
}

Context::~Context() {
    const AutoreleaseScope pool;
    if (MappingEvent) {
        DrainMappings();
        TrimPageCache(0u);
        CommitResidency();
        UnmappedAddresses.clear();
    }
    // Every commit completes before its resources release.
    WaitIdle();
    Slots.clear();
    SparseAddresses.clear();
    PagePool.reset();
    if (Queue && Residency) Queue->removeResidencySet(Residency.get());
    AutoreleaseScope::Release(PendingMappings, PendingRetired, UnmappedAddresses, PendingResidentRemovals);
    AutoreleaseScope::Release(CompletionEvent, MappingEvent, Residency, Queue, Device);
}

Command Context::Begin() const {
    const AutoreleaseScope pool;
    // Slots whose commits completed take new recordings.
    std::erase_if(InFlightSlots, [&](uint32_t index) {
        if (!Completed(Slots[index]->Serial)) return false;
        Slots[index]->Allocator->reset();
        FreeSlots.push_back(index);
        return true;
    });
    if (FreeSlots.empty()) {
        auto slot = std::make_unique<CommandSlot>();
        NS::Error *error = nullptr;
        slot->Allocator = NS::TransferPtr(Device->newCommandAllocator());
        slot->Buffer = NS::TransferPtr(Device->newCommandBuffer());
        const auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        descriptor->setMaxBufferBindCount(ArgumentBufferBindings);
        descriptor->setMaxTextureBindCount(ArgumentTextureBindings);
        descriptor->setMaxSamplerStateBindCount(ArgumentSamplerBindings);
        descriptor->setInitializeBindings(true);
        slot->Arguments = NS::TransferPtr(Device->newArgumentTable(descriptor.get(), &error));
        slot->FragmentArguments = NS::TransferPtr(Device->newArgumentTable(descriptor.get(), &error));
        if (!slot->Allocator || !slot->Buffer || !slot->Arguments || !slot->FragmentArguments) {
            throw std::runtime_error(std::format("Failed to create Metal command resources:\n{}", error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        FreeSlots.push_back(uint32_t(Slots.size()));
        Slots.push_back(std::move(slot));
    }
    const auto index = FreeSlots.back();
    FreeSlots.pop_back();
    auto &slot = *Slots[index];
    slot.Chunk = 0u;
    slot.Used = 0u;
    slot.Buffer->beginCommandBuffer(slot.Allocator.get());
    return Command{*this, index};
}

uint64_t Context::Commit(Command &&command) const {
    const AutoreleaseScope pool;
    auto &slot = *Slots[command.Index];
    slot.Buffer->endCommandBuffer();
    command.Live = false;
    // Encoding can allocate resources, so residency commits after it.
    CommitResidency();
    const auto serial = ++CommitSerial;
    const auto options = NS::TransferPtr(MTL4::CommitOptions::alloc()->init());
    options->addFeedbackHandler([ring = Feedbacks, serial](MTL4::CommitFeedback *feedback) {
        const AutoreleaseScope handler_pool;
        Feedback entry{serial, {feedback->GPUStartTime(), feedback->GPUEndTime(), {}}};
        if (const auto *error = feedback->error()) entry.Result.Error = error->localizedDescription()->utf8String();
        {
            const std::scoped_lock lock{ring->Mutex};
            ring->Entries[serial % FeedbackRing] = std::move(entry);
        }
        ring->Ready.notify_all();
    });
    for (const auto &[event, value] : slot.Waits) Queue->wait(event.get(), value);
    if (slot.Presented) Queue->wait(slot.Presented.get());
    const MTL4::CommandBuffer *buffers[]{slot.Buffer.get()};
    Queue->commit(buffers, 1u, options.get());
    Queue->signalEvent(CompletionEvent.get(), serial);
    for (const auto &[event, value] : slot.Signals) Queue->signalEvent(event.get(), value);
    if (slot.Presented) {
        Queue->signalDrawable(slot.Presented.get());
        slot.Presented->present();
    }
    AutoreleaseScope::Release(slot.Waits, slot.Signals, slot.Presented);
    slot.Serial = serial;
    InFlightSlots.push_back(command.Index);
    return serial;
}

Completion Context::Run(Command &&command) const { return Wait(Commit(std::move(command))); }

bool Context::Completed(uint64_t serial) const { return CompletionEvent->signaledValue() >= serial; }

Completion Context::Wait(uint64_t serial) const {
    if (serial == 0u) return {};
    if (!CompletionEvent->waitUntilSignaledValue(serial, WaitTimeoutMs)) throw std::runtime_error("A committed Metal command did not complete.");
    std::unique_lock lock{Feedbacks->Mutex};
    Feedbacks->Ready.wait(lock, [&] { return Feedbacks->Entries[serial % FeedbackRing].Serial >= serial; });
    const auto &entry = Feedbacks->Entries[serial % FeedbackRing];
    // A later commit can reuse the ring entry, and its completion implies this one's.
    return entry.Serial == serial ? entry.Result : Completion{};
}

void Context::WaitIdle() const {
    if (CommitSerial) (void)Wait(CommitSerial);
}

void Context::AddResident(MTL::Allocation *resource) const {
    const AutoreleaseScope pool;
    if (!Residency || !resource) return;
    // An allocation added back before its pending removal commits stays in the set.
    if (std::erase_if(PendingResidentRemovals, [&](const auto &pending) { return pending.get() == resource; })) return;
    // A membership change recommits the whole set, so a resident allocation leaves it untouched.
    if (Residency->containsAllocation(resource)) return;
    Residency->addAllocation(resource);
    ResidencyDirty = true;
}

void Context::RemoveResident(MTL::Allocation *resource) const {
    if (!Residency || !resource) return;
    PendingResidentRemovals.push_back(NS::RetainPtr(resource));
}

void Context::UpdateMappings(const std::vector<MappingCommand> &commands) const {
    if (commands.empty()) return;
    WaitIdle();
    UnmappedAddresses.clear();
    for (const auto &command : commands) {
        Queue->updateBufferMappings(command.Destination, command.Heap, command.Operations.data(), command.Operations.size());
    }
    Queue->signalEvent(MappingEvent.get(), ++MappingSerial);
    if (!MappingEvent->waitUntilSignaledValue(MappingSerial, WaitTimeoutMs)) throw std::runtime_error("A Metal sparse mapping update did not complete.");
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
        RemoveResident(retired.Addresses.get());
        UnmappedAddresses.push_back(std::move(retired.Addresses));
    }
    PendingRetired.clear();
}

void Context::CommitResidency() const {
    const AutoreleaseScope pool;
    if (Residency && (ResidencyDirty || !PendingResidentRemovals.empty())) {
        // A command in flight references the set, so every commit completes before the set changes and a removed allocation releases after the commit.
        WaitIdle();
        for (const auto &resource : PendingResidentRemovals) Residency->removeAllocation(resource.get());
        Residency->commit();
        PendingResidentRemovals.clear();
        ResidencyDirty = false;
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
    WaitIdle();
    PagePool->TrimCache(keep_bytes);
    CommitResidency();
}

NS::SharedPtr<MTL::Buffer> Context::ReserveSparseAddresses(uint64_t bytes) const {
    const AutoreleaseScope pool;
    if (!std::has_single_bit(bytes)) throw std::invalid_argument("Sparse address reservations require a power-of-two size.");
    auto buffer = NS::TransferPtr(Device->newBuffer(bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256));
    if (!buffer) throw std::runtime_error("Failed to reserve sparse Metal addresses.");
    AddResident(buffer.get());
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

Command::Command(const Context &ctx, uint32_t slot) : Ctx(ctx), Index(slot) {}
Command::Command(Command &&other) noexcept : Ctx(other.Ctx), Index(other.Index), Live(std::exchange(other.Live, false)) {}
Command::~Command() {
    if (!Live) return;
    const AutoreleaseScope pool;
    auto &slot = Slot();
    slot.Buffer->endCommandBuffer();
    AutoreleaseScope::Release(slot.Waits, slot.Signals, slot.Presented);
    Ctx.FreeSlots.push_back(Index);
    Live = false;
}

Context::CommandSlot &Command::Slot() const { return *Ctx.Slots[Index]; }
MTL4::CommandBuffer *Command::operator*() const { return Slot().Buffer.get(); }
MTL4::ArgumentTable *Command::Arguments() const { return Slot().Arguments.get(); }
MTL4::ArgumentTable *Command::FragmentArguments() const { return Slot().FragmentArguments.get(); }

ComputeEncoder Command::BeginCompute() {
    auto *encoder = (**this)->computeCommandEncoder();
    encoder->setArgumentTable(Arguments());
    State = {};
    return {this, encoder};
}

RenderEncoder Command::BeginRender(const MTL4::RenderPassDescriptor *pass) {
    auto *encoder = (**this)->renderCommandEncoder(pass);
    encoder->setArgumentTable(Arguments(), MTL::RenderStageVertex | MTL::RenderStageObject | MTL::RenderStageMesh);
    encoder->setArgumentTable(FragmentArguments(), MTL::RenderStageFragment);
    State = {};
    return {this, encoder};
}

void SetCullMode(RenderEncoder encoder, MTL::CullMode mode) {
    if (encoder.Cmd->State.Cull == mode) return;
    encoder->setCullMode(mode);
    encoder.Cmd->State.Cull = mode;
}

void SetThreadgroupMemoryLength(ComputeEncoder encoder, uint32_t length, uint32_t index) {
    auto &held = encoder.Cmd->State.ThreadgroupLengths[index];
    if (held == length) return;
    encoder->setThreadgroupMemoryLength(length, index);
    held = length;
}

uint64_t Command::Upload(std::span<const std::byte> bytes) {
    auto &slot = Slot();
    const auto size = (uint64_t(bytes.size()) + UploadAlignment - 1u) / UploadAlignment * UploadAlignment;
    // Advance to the first chunk with room, allocating one of the chunk size or the upload's own size.
    while (slot.Chunk < slot.Uploads.size() && slot.Used + size > slot.Uploads[slot.Chunk].Length()) {
        ++slot.Chunk;
        slot.Used = 0u;
    }
    if (slot.Chunk == slot.Uploads.size()) slot.Uploads.emplace_back(Ctx, std::max(size, UploadChunkBytes));
    auto &chunk = slot.Uploads[slot.Chunk];
    std::memcpy(chunk.Contents().data() + slot.Used, bytes.data(), bytes.size());
    const auto address = chunk.Address() + slot.Used;
    slot.Used += size;
    return address;
}

void Command::SetBytes(std::span<const std::byte> bytes, uint32_t index) { SetAddress(Upload(bytes), index); }
void Command::SetGeometryBytes(std::span<const std::byte> bytes, uint32_t index) { Arguments()->setAddress(Upload(bytes), index); }
void Command::SetFragmentBytes(std::span<const std::byte> bytes, uint32_t index) { FragmentArguments()->setAddress(Upload(bytes), index); }
void Command::SetAddress(uint64_t address, uint32_t index) {
    Arguments()->setAddress(address, index);
    FragmentArguments()->setAddress(address, index);
}
void Command::SetBuffer(const MTL::Buffer *buffer, uint64_t offset, uint32_t index) { SetAddress(buffer->gpuAddress() + offset, index); }
void Command::SetTexture(const MTL::Texture *texture, uint32_t index) {
    Arguments()->setTexture(texture->gpuResourceID(), index);
    FragmentArguments()->setTexture(texture->gpuResourceID(), index);
}
void Command::SetSampler(const MTL::SamplerState *sampler, uint32_t index) {
    Arguments()->setSamplerState(sampler->gpuResourceID(), index);
    FragmentArguments()->setSamplerState(sampler->gpuResourceID(), index);
}

void Command::WaitEvent(MTL::Event *event, uint64_t value) { Slot().Waits.emplace_back(NS::RetainPtr(event), value); }
void Command::SignalEvent(MTL::Event *event, uint64_t value) { Slot().Signals.emplace_back(NS::RetainPtr(event), value); }
void Command::Present(MTL::Drawable *drawable) { Slot().Presented = NS::RetainPtr(drawable); }
} // namespace mtl
