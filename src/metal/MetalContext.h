#pragma once

#include "metal/MetalSchema.h"

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include <Foundation/NSSharedPtr.hpp>

namespace NS {
class String;
} // namespace NS
namespace MTL {
class Allocation;
class Device;
class ResidencySet;
class Buffer;
class Texture;
class Heap;
class Event;
class SharedEvent;
class Drawable;
} // namespace MTL

namespace MTL4 {
class CommandQueue;
class CommandAllocator;
class CommandBuffer;
class ArgumentTable;
} // namespace MTL4

namespace mtl {
struct PhysicalPagePool;
struct PhysicalPage;
struct Context;
struct Command;
NS::SharedPtr<NS::String> Str(std::string_view);

// A resource in the queue's residency set for its lifetime.
template<typename T> struct Resident {
    Resident();
    Resident(const Context &, NS::SharedPtr<T>);
    Resident(const Resident &) = delete;
    Resident &operator=(const Resident &) = delete;
    Resident(Resident &&) noexcept;
    Resident &operator=(Resident &&) noexcept;
    ~Resident();

    T *get() const { return Handle.get(); }
    T *operator->() const { return Handle.get(); }
    explicit operator bool() const { return bool(Handle); }

    NS::SharedPtr<T> Handle;
    const Context *Ctx{};
};

// A shared buffer resident for its lifetime.
struct ResidentBuffer : Resident<MTL::Buffer> {
    ResidentBuffer() = default;
    ResidentBuffer(const Context &, uint64_t bytes);

    MTL::Buffer *operator*() const { return get(); }
    uint64_t Address() const;
    uint64_t Length() const;
    std::span<std::byte> Contents() const;
};

// What Metal reported once a committed command completed.
struct Completion {
    double GpuStartTime{}, GpuEndTime{};
    std::string Error; // Empty when the command ran without a GPU error.
};

// One queue runs every command and every sparse mapping update.
// Each commit signals a serial on the queue after its command, and the queue signals serials in commit order, so a completed serial implies every earlier commit completed.
// Queue-wide residency includes sparse GPU buffers and the physical heaps behind them.
// A page mapping update or a residency set commit runs after every committed command completes, and a mapping update completes before the call that published it returns.
// A retired address unmaps in the next mapping update, and the update after that releases it, so a command that declared it has completed first.
// A retired address's pages return to the pool when its unmap has run, so no page is mapped in two addresses at once.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // Begins recording a command whose encoders bind the command's argument table.
    Command Begin() const;
    // Commits residency, then the command, and returns the serial the queue signals once the command completes.
    uint64_t Commit(Command &&) const;
    // Commits the command and returns once it completes, with Metal's feedback.
    Completion Run(Command &&) const;
    // The serial of the latest commit, which completes after every earlier one.
    uint64_t LatestSerial() const { return CommitSerial; }
    bool Completed(uint64_t serial) const;
    // Returns once the queue completes the serial, with Metal's feedback for the commit it identifies.
    Completion Wait(uint64_t serial) const;

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
    // Arguments are physical pages.
    // CommitResidency maps them after every committed command completes.
    void MapBufferPages(MTL::Buffer *, MTL::Heap *, uint64_t first, uint64_t count, uint64_t heap_first) const;

    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL4::CommandQueue> Queue;
    NS::SharedPtr<MTL::ResidencySet> Residency;

private:
    friend struct Command;
    struct MappingCommand;
    // The allocator, command buffer, argument tables and upload storage one recording uses, reused once its commit completes.
    struct CommandSlot {
        NS::SharedPtr<MTL4::CommandAllocator> Allocator;
        NS::SharedPtr<MTL4::CommandBuffer> Buffer;
        NS::SharedPtr<MTL4::ArgumentTable> Arguments, FragmentArguments;
        std::vector<ResidentBuffer> Uploads;
        uint32_t Chunk{};
        uint64_t Used{};
        uint64_t Serial{};
        // Queue operations the commit surrounds the command with.
        std::vector<std::pair<NS::SharedPtr<MTL::Event>, uint64_t>> Waits, Signals;
        NS::SharedPtr<MTL::Drawable> Presented;
    };
    // Runs the mapping commands and returns once they complete.
    void UpdateMappings(const std::vector<MappingCommand> &) const;
    // Runs one mapping update with every pending map and unmap, then returns the retired pages to the pool and holds the retired addresses until the next update.
    void UpdatePendingMappings() const;
    // Returns once every committed command completes.
    void WaitIdle() const;
    mutable std::shared_ptr<PhysicalPagePool> PagePool;
    mutable bool ResidencyDirty{false};
    mutable std::vector<NS::SharedPtr<MTL::Allocation>> PendingResidentRemovals;
    // Mapping completion and command completion are signaled on separate timelines.
    NS::SharedPtr<MTL::SharedEvent> MappingEvent, CompletionEvent;
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
    // Unmapped addresses that a command encoded before their unmap can still declare, released once every command committed since has completed.
    mutable std::vector<NS::SharedPtr<MTL::Buffer>> UnmappedAddresses;
    mutable std::unordered_map<MTL::Buffer *, NS::SharedPtr<MTL::Buffer>> SparseAddresses;
    mutable uint64_t MappingSerial{}, CommitSerial{};
    mutable std::vector<std::unique_ptr<CommandSlot>> Slots;
    mutable std::vector<uint32_t> FreeSlots, InFlightSlots;
    // Metal reports each commit's completion on its own thread, into the ring entry of the commit's serial.
    // The handlers share the ring's ownership, so a report arriving after the context's destruction stays valid.
    static constexpr uint32_t FeedbackRing{256u};
    struct Feedback {
        uint64_t Serial{};
        Completion Result;
    };
    struct FeedbackRingState {
        std::mutex Mutex;
        std::condition_variable Ready;
        std::array<Feedback, FeedbackRing> Entries;
    };
    std::shared_ptr<FeedbackRingState> Feedbacks{std::make_shared<FeedbackRingState>()};
};
} // namespace mtl
