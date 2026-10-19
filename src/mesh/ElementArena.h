#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/MeshElementSet.h"
#include "metal/BufferArena.h"

#include <set>

struct ElementSetRef {
    uint32_t Index{InvalidOffset};
    explicit operator bool() const { return Index != InvalidOffset; }
    auto operator<=>(const ElementSetRef &) const = default;
};

// A block that gained elements in one insertion, with the slots it gained.
struct ElementBlockGain {
    uint32_t Block;
    std::array<uint32_t, MeshElementBlockWords> Added;
};
// The handles of one insertion and the blocks that gained them.
struct ElementInsert {
    ElementHandleRange Handles;
    std::vector<ElementBlockGain> Blocks;
};

// A domain owns geometry once in 256-slot blocks with live masks, written on the CPU and read by kernels.
// A set's table run names the block of each of its local blocks, so a record-relative handle resolves to an arena address through its set's table.
// A Derived index of each set's blocks with free slots, ordered by their longest free run, places insertions.
template<typename T, uint32_t Size = MeshElementBlockSize>
struct ElementArena {
    static constexpr uint32_t BlockSize = Size, BlockWords = (Size + 31u) / 32u;

    ElementArena(mtl::BufferContext &ctx, SlotType slot_type) : Buffer(ctx, 0, slot_type), Blocks(ctx, SlotType::Buffer), Sets(ctx, SlotType::Buffer), Table(ctx, SlotType::Buffer) {}

    void Track(store::History &history, const std::string &name) {
        Blocks.Track(history, name + ".blocks");
        Sets.Track(history, name + ".sets");
        Table.Track(history, name + ".table");
        Buffer.Track(history, name + ".bytes");
    }
    static uint32_t BlockCount(uint32_t count) { return count / BlockSize + uint32_t(count % BlockSize != 0u); }
    const MeshElementSet &Set(ElementSetRef set) const {
        if (set.Index == InvalidOffset || set.Index >= Sets.Buffer.template Count<MeshElementSet>()) throw std::invalid_argument("Invalid element set.");
        const auto &value = Sets.Get({set.Index, 1})[0];
        if (value.First == InvalidOffset) throw std::invalid_argument("Retired element set.");
        return value;
    }
    uint32_t Capacity() const { return Blocks.HighWaterMark() * BlockSize; }
    uint32_t Count(ElementSetRef set) const { return set ? Set(set).Count : 0u; }
    // The set's table entries in local block order, InvalidOffset at a local block without a block.
    std::span<const uint32_t> Entries(ElementSetRef set) const {
        if (!set) return {};
        const auto &header = Set(set);
        return Table.Get({header.First, header.BlockCount});
    }
    // The bound of the set's handles.
    uint32_t HandleBound(ElementSetRef set) const { return set ? Set(set).BlockCount * BlockSize : 0u; }
    // The set's table run in the table arena.
    SlotOffset TableRef(ElementSetRef set) const { return {Table.Buffer.Slot, set ? Set(set).First : 0u}; }
    // The set headers and table runs kernels resolve a set's index through.
    SetArenaSlots Slots() const { return {Sets.Buffer.Slot, Table.Buffer.Slot, Blocks.Buffer.Slot}; }
    // The arena address of one of the set's handles.
    ElementAddress Address(ElementSetRef set, ElementHandle handle) const { return Resolve<BlockSize>(Entries(set), handle); }
    // Whether the handle names a live element of the set.
    bool Contains(ElementSetRef set, ElementHandle handle) const { return set && Contains(Entries(set), handle); }
    // Whether the handle names a live element of the set whose table entries are `entries`.
    bool Contains(std::span<const uint32_t> entries, ElementHandle handle) const {
        const auto local = handle.Value / BlockSize, slot = handle.Value % BlockSize;
        return local < entries.size() && entries[local] != InvalidOffset && (Blocks.Get({entries[local], 1u})[0].Live[slot / 32u] >> (slot % 32u) & 1u);
    }
    // Visits the arena address runs of the set's `count` handles from `first`, one run per local block.
    void ForEachAddressRun(ElementSetRef set, ElementHandle first, uint32_t count, auto &&visit) const {
        const auto entries = Entries(set);
        for (auto h = first.Value; h < first.Value + count;) {
            const auto end = std::min(first.Value + count, (h / BlockSize + 1u) * BlockSize);
            visit(Range{Resolve<BlockSize>(entries, ElementHandle{h}).Value(), end - h});
            h = end;
        }
    }
    // The record-relative handle of an owned block's arena address.
    ElementHandle Handle(ElementAddress address) const { return {Blocks.Get({address.Value() / BlockSize, 1u})[0].Local * BlockSize + address.Value() % BlockSize}; }
    Range Dense(ElementSetRef set) const {
        if (!set) return {};
        const auto &header = Set(set);
        if (header.Count && !(header.Flags & 1u)) throw std::invalid_argument("Element set has no dense view.");
        return {Table.Get({header.First, 1u})[0] * BlockSize, header.Count};
    }
    // Visits the set's blocks in local order, including its first block when it contains no element.
    void ForEachBlock(ElementSetRef set, auto &&visit) const {
        if (!set) return;
        const auto blocks = Blocks.Buffer.template GetSpan<MeshElementBlock>();
        for (const auto b : Entries(set)) {
            if (b != InvalidOffset) visit(b, blocks[b]);
        }
    }
    // Visits the live slots of one block in ascending order.
    void ForEachSlot(uint32_t block, auto &&visit) const {
        const auto &members = Blocks.Get({block, 1})[0];
        for (uint32_t w = 0u; w < BlockWords; ++w) {
            for (auto bits = members.Live[w]; bits; bits &= bits - 1u) visit(w * 32u + uint32_t(std::countr_zero(bits)));
        }
    }
    // Visits the set's live handles in ascending order with their addresses.
    void ForEach(ElementSetRef set, auto &&visit) const {
        ForEachBlock(set, [&](uint32_t b, const auto &block) {
            for (uint32_t w = 0u; w < BlockWords; ++w) {
                for (auto bits = block.Live[w]; bits; bits &= bits - 1u) {
                    const auto slot = w * 32u + uint32_t(std::countr_zero(bits));
                    visit(ElementHandle{block.Local * BlockSize + slot}, ElementAddress::InBlock<BlockSize>(b, slot));
                }
            }
        });
    }

    void ReserveAdditional(uint32_t count) { ReserveBlocks(BlockCount(count)); }
    void PlanAdditional(uint32_t count) { PlannedBlocks += BlockCount(count); }
    void CommitPlanned() { ReserveBlocks(std::exchange(PlannedBlocks, 0u)); }
    void Mirror(Range range) {
        if (range.Count) Buffer.SetUsedSize(std::max(Buffer.UsedSize, (uint64_t(range.Offset) + range.Count) * sizeof(T)));
    }

    // A dense set whose first `count` handles are live.
    ElementSetRef Create(uint32_t count = 0) {
        auto block_allocation = Blocks.BeginAllocation();
        auto set_allocation = Sets.BeginAllocation();
        auto table_allocation = Table.BeginAllocation();
        const auto blocks = AllocateBlocks(std::max(1u, BlockCount(count)));
        const ElementSetRef set{Sets.Allocate(1).Offset};
        const auto run = Table.Allocate(blocks.Count);
        const auto entries = Table.GetMutable(run);
        const auto members = Blocks.GetMutable(blocks);
        for (uint32_t i = 0; i < blocks.Count; ++i) {
            const auto live = std::min(BlockSize, count > i * BlockSize ? count - i * BlockSize : 0u);
            members[i] = {.Owner = set.Index, .Local = i, .Count = live, .Live = Prefix(live)};
            entries[i] = blocks.Offset + i;
        }
        Sets.GetMutable({set.Index, 1})[0] = {.First = run.Offset, .Capacity = run.Count, .Count = count, .BlockCount = blocks.Count, .Flags = 1u};
        Index(blocks.Offset + blocks.Count - 1u);
        block_allocation.Commit();
        set_allocation.Commit();
        table_allocation.Commit();
        return set;
    }

    // Insert count live elements and return their handles and changed blocks.
    // Create a dense set when absent.
    // Without list, require consecutive handles; otherwise store nonconsecutive handles in list.
    ElementInsert Insert(ElementSetRef &set, uint32_t count, BufferArena<uint32_t> *list) {
        ElementInsert result;
        if (!count) return result;
        if (uint64_t(Count(set)) + count >= InvalidOffset) throw std::length_error("Element count overflow.");
        if (!set) {
            set = Create(count);
            ForEachBlock(set, [&](uint32_t b, const auto &block) { result.Blocks.push_back({b, block.Live}); });
            result.Handles = {.First = 0u, .Count = count};
            return result;
        }
        auto header = Set(set);
        const bool run = !list;
        std::vector<uint32_t> handles;
        uint32_t remaining = count;
        if (count <= BlockSize) {
            for (const auto b : AvailableBlocks(set, run ? count : 1u, run ? 1u : count)) {
                auto &block = Blocks.GetMutable({b, 1})[0];
                const auto base = block.Local * BlockSize;
                auto &gain = result.Blocks.emplace_back(ElementBlockGain{b, {}});
                if (run) {
                    const auto first = FreeRunStart(block.Live, count);
                    for (uint32_t i = 0; i < count; ++i) gain.Added[(first + i) / 32u] |= 1u << ((first + i) % 32u);
                    handles.push_back(base + first);
                    remaining = 0u;
                } else {
                    for (uint32_t w = 0; w < BlockWords && remaining; ++w) {
                        for (auto free = ~block.Live[w] & Prefix(BlockSize)[w]; free && remaining; free &= free - 1u, --remaining) {
                            const auto bit = uint32_t(std::countr_zero(free));
                            gain.Added[w] |= 1u << bit;
                            handles.push_back(base + w * 32u + bit);
                        }
                    }
                }
                for (uint32_t w = 0; w < BlockWords; ++w) block.Live[w] |= gain.Added[w];
                block.Count = LiveCount(block.Live);
                if (!remaining) break;
            }
        }
        if (remaining) {
            auto allocation = Blocks.BeginAllocation();
            const auto fresh = AllocateBlocks(BlockCount(remaining));
            const auto local = Fill(header, fresh);
            const auto members = Blocks.GetMutable(fresh);
            for (uint32_t i = 0; i < fresh.Count; ++i) {
                const auto live = std::min(remaining, BlockSize);
                members[i] = {.Owner = set.Index, .Local = local + i, .Count = live, .Live = Prefix(live)};
                result.Blocks.push_back({fresh.Offset + i, members[i].Live});
                if (count <= BlockSize) {
                    for (uint32_t s = 0; s < live; ++s) handles.push_back((local + i) * BlockSize + s);
                }
                remaining -= live;
            }
            if (count > BlockSize) handles.push_back(local * BlockSize);
            allocation.Commit();
        }
        header.Count += count;
        ++header.Revision;
        header.Flags = 0u;
        Sets.GetMutable({set.Index, 1})[0] = header;
        for (const auto &gain : result.Blocks) Index(gain.Block);
        const bool consecutive = handles.size() == 1u || (handles.back() - handles.front() == count - 1u && std::ranges::adjacent_find(handles, [](uint32_t a, uint32_t b) { return b != a + 1u; }) == handles.end());
        if (consecutive) {
            result.Handles = {.First = handles.front(), .Count = count};
        } else {
            if (list->Buffer.Slot == InvalidSlot) throw std::invalid_argument("A handle list requires a bound buffer.");
            result.Handles = {.Handles = {list->Buffer.Slot, list->Allocate(std::span<const uint32_t>{handles}).Offset}, .Count = count};
        }
        return result;
    }

    // Clears the live bits the sparse work table names, keyed by the set's local blocks, and returns the ascending blocks it names.
    // Blocks left empty leave the set, except its first block.
    // Work naming a local block the set lacks or a slot outside its handles is rejected before any write.
    std::vector<uint32_t> Erase(ElementSetRef set, const mtl::Buffer &storage, ElementWork work)
        requires(Size == MeshElementBlockSize)
    {
        const auto occupied = CheckWork(storage, work);
        if (!occupied) return {};
        const auto entries = Entries(set);
        const auto data = storage.GetSpan<uint32_t>({work.Storage.Offset, WorkHeaderWords + work.Capacity * (WorkBlockWords + 2u)});
        const auto entry = [&](uint32_t i) { return data.subspan(WorkHeaderWords + data[WorkHeaderWords + work.Capacity * WorkBlockWords + i] * WorkBlockWords, WorkBlockWords); };
        for (uint32_t i = 0; i < occupied; ++i) {
            const auto slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + i];
            const auto key = slot < work.Capacity ? data[WorkHeaderWords + slot * WorkBlockWords] : 0u;
            const auto local = key - 1u;
            bool valid = key && local < entries.size() && entries[local] != InvalidOffset && uint64_t(local) * BlockSize < work.Count;
            for (uint32_t w = 0; valid && w < BlockWords; ++w) {
                const auto first = uint64_t(local) * BlockSize + w * 32u;
                const auto available = uint32_t(std::min<uint64_t>(32u, work.Count > first ? work.Count - first : 0u));
                valid = !(entry(i)[1u + w] & ~(available == 32u ? ~0u : (1u << available) - 1u));
            }
            if (!valid) throw std::invalid_argument("Element removal work has invalid or foreign block membership.");
        }
        std::vector<std::pair<uint32_t, LiveMask>> cleared;
        cleared.reserve(occupied);
        for (uint32_t i = 0; i < occupied; ++i) {
            const auto removal = entry(i);
            std::ranges::copy(removal.subspan(1u, BlockWords), cleared.emplace_back(entries[removal[0] - 1u], LiveMask{}).second.begin());
        }
        return ClearSlots(set, std::move(cleared), 0u);
    }

    // Retires the sets, whose ascending blocks the caller compiled.
    void Destroy(std::span<const ElementSetRef> sets, std::span<const uint32_t> blocks) {
        if (sets.empty()) return;
        std::vector<uint32_t> owners;
        std::vector<Range> tables;
        owners.reserve(sets.size());
        for (const auto set : sets) {
            owners.push_back(set.Index);
            const auto &header = Set(set);
            tables.push_back({header.First, header.Capacity});
        }
        if (!std::ranges::is_sorted(owners)) std::ranges::sort(owners);
        if (std::ranges::adjacent_find(owners) != owners.end()) throw std::invalid_argument("Repeated owner in element retirement.");
        // The free-slot index is ordered by owner: unlink each retiring owner run once.
        ForEachIndexRun(owners, [&](size_t first, size_t count) {
            Available.erase(Available.lower_bound({owners[first], 0u, 0u}), Available.lower_bound({owners[first] + uint32_t(count), 0u, 0u}));
        });
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            const Range range{blocks[first], uint32_t(count)};
            std::ranges::fill(Blocks.GetMutable(range), MeshElementBlock{});
            const auto end = std::min<size_t>(Indexed.size(), size_t(range.Offset) + range.Count);
            if (range.Offset < end) std::ranges::fill(std::span{Indexed}.subspan(range.Offset, end - range.Offset), std::pair{InvalidOffset, 0u});
            Blocks.Release(range);
        });
        std::ranges::sort(tables, {}, &Range::Offset);
        CoalesceRanges(tables);
        Table.Release(std::move(tables));
        ForEachIndexRun(owners, [&](size_t first, size_t count) {
            const Range range{owners[first], uint32_t(count)};
            std::ranges::fill(Sets.GetMutable(range), MeshElementSet{});
            Sets.Release(range);
        });
    }

    // Import/current count-scatter emission consumes a dense view of a new set.
    ElementSetRef Allocate(uint32_t count) {
        if (!count) return {};
        return Create(count);
    }
    // A set with the source's table shape and live slots, its blocks fresh and consecutive in table order.
    // Its handles name the same elements as the source's, so values copy without remapping.
    ElementSetRef CloneMembership(ElementSetRef source) {
        if (!source) return {};
        const auto header = Set(source);
        const auto from = Entries(source);
        const auto owned = uint32_t(std::ranges::count_if(from, [](uint32_t b) { return b != InvalidOffset; }));
        auto block_allocation = Blocks.BeginAllocation();
        auto set_allocation = Sets.BeginAllocation();
        auto table_allocation = Table.BeginAllocation();
        const auto blocks = AllocateBlocks(owned);
        const ElementSetRef set{Sets.Allocate(1).Offset};
        const auto run = Table.Allocate(header.BlockCount);
        const auto metadata = Blocks.Buffer.template GetSpan<MeshElementBlock>();
        std::vector<uint32_t> entries(header.BlockCount, InvalidOffset);
        std::vector<MeshElementBlock> members(owned);
        for (uint32_t local = 0u, next = 0u; local < header.BlockCount; ++local) {
            if (from[local] == InvalidOffset) continue;
            const auto &before = metadata[from[local]];
            members[next] = {.Owner = set.Index, .Local = local, .Count = before.Count, .Live = before.Live};
            entries[local] = blocks.Offset + next++;
        }
        std::ranges::copy(members, Blocks.GetMutable(blocks).begin());
        std::ranges::copy(entries, Table.GetMutable(run).begin());
        Sets.GetMutable({set.Index, 1})[0] = {.First = run.Offset, .Capacity = run.Count, .Count = header.Count, .BlockCount = header.BlockCount, .Flags = header.Flags};
        for (uint32_t i = 0u; i < blocks.Count; ++i) Index(blocks.Offset + i);
        block_allocation.Commit();
        set_allocation.Commit();
        table_allocation.Commit();
        return set;
    }
    ElementSetRef Allocate(std::span<const T> values) {
        const auto set = Allocate(uint32_t(values.size()));
        Buffer.Update(BytesOf(values), uint64_t(Dense(set).Offset) * sizeof(T));
        return set;
    }
    // Keeps the first `used` handles of an insertion into the set, a run or listed in `list`, clears the others, and shrinks `inserted` to the used ones.
    // A dense set's allocation is its only insertion, so it stays dense.
    // Blocks left without elements leave the set, except its first block, and a set left without elements retires.
    // Returns the ascending blocks that lost elements.
    std::vector<uint32_t> Shrink(ElementSetRef &set, ElementHandleRange &inserted, std::span<const uint32_t> list, uint32_t used) {
        if (used >= inserted.Count) return {};
        std::vector<uint32_t> blocks;
        if (inserted.Handles.Slot == InvalidSlot) {
            const auto entries = Entries(set);
            std::vector<std::pair<uint32_t, LiveMask>> cleared;
            const uint64_t first = inserted.First + used, end = uint64_t(inserted.First) + inserted.Count;
            for (auto b = first / BlockSize; b * BlockSize < end; ++b) {
                const auto base = b * BlockSize;
                const auto kept = Prefix(uint32_t(std::max(first, base) - base)), covered = Prefix(uint32_t(std::min<uint64_t>(end - base, BlockSize)));
                auto &mask = cleared.emplace_back(entries[b], LiveMask{}).second;
                for (uint32_t w = 0; w < BlockWords; ++w) mask[w] = covered[w] & ~kept[w];
            }
            blocks = ClearSlots(set, std::move(cleared), Set(set).Flags);
        } else {
            std::vector<uint32_t> handles(list.begin() + inserted.Handles.Offset + used, list.begin() + inserted.Handles.Offset + inserted.Count);
            std::ranges::sort(handles);
            blocks = Erase(set, handles);
        }
        inserted.Count = used;
        if (!Count(set)) {
            Release(set);
            set = {};
        }
        return blocks;
    }
    // Clears the set's ascending live handles and returns the ascending blocks that lost them.
    // Blocks left without elements leave the set, except its first block.
    std::vector<uint32_t> Erase(ElementSetRef set, std::span<const uint32_t> handles) {
        if (handles.empty()) return {};
        const auto entries = Entries(set);
        std::vector<std::pair<uint32_t, LiveMask>> cleared;
        uint32_t local = InvalidOffset;
        for (const auto handle : handles) {
            const auto slot = handle % BlockSize;
            if (handle / BlockSize != local) {
                local = handle / BlockSize;
                cleared.emplace_back(entries[local], LiveMask{});
            }
            cleared.back().second[slot / 32u] |= 1u << (slot % 32u);
        }
        return ClearSlots(set, std::move(cleared), 0u);
    }
    // Clears every live slot of the set's ascending blocks, which leave the set unless one is its first block.
    std::vector<uint32_t> Clear(ElementSetRef set, std::span<const uint32_t> blocks) {
        std::vector<std::pair<uint32_t, LiveMask>> cleared;
        cleared.reserve(blocks.size());
        for (const auto b : blocks) cleared.emplace_back(b, Blocks.Get({b, 1})[0].Live);
        return ClearSlots(set, std::move(cleared), 0u);
    }
    // Gives the set an empty fresh block at each ascending local block it lacks, growing its table to cover them, and returns the blocks in that order.
    std::vector<uint32_t> Place(ElementSetRef set, std::span<const uint32_t> locals) {
        std::vector<uint32_t> added;
        if (locals.empty()) return added;
        auto header = Set(set);
        auto allocation = Blocks.BeginAllocation();
        const auto fresh = AllocateBlocks(uint32_t(locals.size()));
        if (locals.back() >= header.BlockCount) Append(header, {InvalidOffset, locals.back() + 1u - header.BlockCount});
        const auto entries = Table.GetMutable({header.First, header.BlockCount});
        const auto members = Blocks.GetMutable(fresh);
        for (uint32_t i = 0u; i < fresh.Count; ++i) {
            if (entries[locals[i]] != InvalidOffset) throw std::logic_error("A placed block replaces a held one.");
            entries[locals[i]] = fresh.Offset + i;
            members[i] = {.Owner = set.Index, .Local = locals[i]};
            added.push_back(fresh.Offset + i);
        }
        ++header.Revision;
        header.Flags = 0u;
        Sets.GetMutable({set.Index, 1})[0] = header;
        allocation.Commit();
        for (const auto b : added) Index(b);
        return added;
    }
    // Moves every block of `donor` after the set's table entries, with each block's local index shifted by the set's block count, and retires the donor.
    // Returns the shift, which the moved blocks' references add.
    uint32_t Adopt(ElementSetRef &set, ElementSetRef &donor) {
        if (!donor) return 0u;
        if (!set) {
            set = std::exchange(donor, {});
            return 0u;
        }
        auto header = Set(set);
        const auto from = Set(donor);
        const auto shift = header.BlockCount;
        const std::vector<uint32_t> moved{Entries(donor).begin(), Entries(donor).end()};
        Append(header, {InvalidOffset, uint32_t(moved.size())});
        std::ranges::copy(moved, Table.GetMutable({header.First + shift, uint32_t(moved.size())}).begin());
        for (const auto b : moved) {
            if (b == InvalidOffset) continue;
            auto &block = Blocks.GetMutable({b, 1u})[0];
            block.Owner = set.Index;
            block.Local += shift;
            Index(b);
        }
        header.Count += from.Count;
        ++header.Revision;
        header.Flags = 0u;
        Sets.GetMutable({set.Index, 1})[0] = header;
        Destroy(std::span{&donor, 1u}, {});
        donor = {};
        return shift;
    }
    // Gives each pair's second block, owned by the set, the live slots of its first block, and the set the flags.
    void AssignLive(ElementSetRef set, std::span<const uvec2> pairs, uint32_t flags) {
        auto header = Set(set);
        for (const auto [from, to] : pairs) {
            const auto live = Blocks.Get({from, 1})[0].Live;
            auto &block = Blocks.GetMutable({to, 1})[0];
            const auto count = LiveCount(live);
            header.Count = header.Count - block.Count + count;
            block.Live = live;
            block.Count = count;
        }
        header.Flags = flags;
        ++header.Revision;
        Sets.GetMutable({set.Index, 1})[0] = header;
        for (const auto pair : pairs) Index(pair.y);
    }
    void Release(ElementSetRef set) {
        if (!set) return;
        std::vector<uint32_t> blocks;
        ForEachBlock(set, [&](uint32_t b, const auto &) { blocks.push_back(b); });
        std::ranges::sort(blocks);
        Destroy(std::span{&set, 1u}, blocks);
    }
    void Reset() {
        Blocks.Reset();
        Sets.Reset();
        Table.Reset();
        Buffer.SetUsedSize(0);
        Indexed.clear();
        Available.clear();
    }
    // Refreshes the free-run index of blocks whose metadata a history restore changed.
    void Reindex(Range blocks) {
        for (uint32_t b = blocks.Offset; b < blocks.Offset + blocks.Count; ++b) Index(b);
    }

    std::span<const T> Get(Range range) const { return Buffer.GetSpan<T>(range); }
    std::span<T> GetMutable(Range range) { return Buffer.GetMutableSpan<T>(range); }
    std::span<const T> Get(ElementSetRef set) const { return Get(Dense(set)); }
    std::span<T> GetMutable(ElementSetRef set) { return GetMutable(Dense(set)); }
    SlottedRange Slotted(ElementSetRef set) const { return {Dense(set), Buffer.Slot}; }

    mtl::Buffer Buffer;
    BufferArena<MeshElementBlock> Blocks;
    BufferArena<MeshElementSet> Sets;
    BufferArena<uint32_t> Table; // Every set's table run

private:
    using LiveMask = std::array<uint32_t, MeshElementBlockWords>;
    uint32_t PlannedBlocks{};
    // Derived: the owner and longest free run each block is indexed under, and the indexed blocks ordered by owner, run and block.
    std::vector<std::pair<uint32_t, uint32_t>> Indexed;
    std::set<std::array<uint32_t, 3>> Available;

    static LiveMask Prefix(uint32_t count) {
        LiveMask live{};
        for (uint32_t w = 0; w < BlockWords; ++w) {
            const auto bits = std::min(32u, count > w * 32u ? count - w * 32u : 0u);
            live[w] = bits == 32u ? ~0u : (1u << bits) - 1u;
        }
        return live;
    }
    static uint32_t LiveCount(const LiveMask &live) {
        uint32_t count = 0u;
        for (const auto word : live) count += uint32_t(std::popcount(word));
        return count;
    }
    // Calls visit(first, count) for each maximal run of dead slots in ascending order, stopping when it returns true.
    static void ForEachFreeRun(const LiveMask &live, auto &&visit) {
        uint32_t first = InvalidOffset;
        for (uint32_t i = 0u; i < BlockSize;) {
            const auto shift = i % 32u, width = 32u - shift;
            const auto bits = live[i / 32u] >> shift;
            if (bits & 1u) {
                if (first != InvalidOffset && visit(first, i - first)) return;
                first = InvalidOffset;
                i += std::min(width, uint32_t(std::countr_one(bits)));
            } else {
                if (first == InvalidOffset) first = i;
                i += bits ? uint32_t(std::countr_zero(bits)) : width;
            }
        }
        if (first != InvalidOffset) visit(first, BlockSize - first);
    }
    static uint32_t LongestFreeRun(const LiveMask &live) {
        uint32_t longest = 0u;
        ForEachFreeRun(live, [&](uint32_t, uint32_t count) { longest = std::max(longest, count); return false; });
        return longest;
    }
    static uint32_t FreeRunStart(const LiveMask &live, uint32_t count) {
        uint32_t start = InvalidOffset;
        ForEachFreeRun(live, [&](uint32_t first, uint32_t length) {
            if (length < count) return false;
            start = first;
            return true;
        });
        return start;
    }
    // Rederives a block's free-run index entry from its metadata.
    void Index(uint32_t b) {
        if (b >= Indexed.size()) Indexed.resize(b + 1u, {InvalidOffset, 0u});
        auto &entry = Indexed[b];
        if (entry.second) Available.erase({entry.first, entry.second, b});
        entry = {InvalidOffset, 0u};
        if (b >= Blocks.Buffer.template Count<MeshElementBlock>()) return;
        const auto &block = Blocks.Get({b, 1})[0];
        const auto run = block.Owner == InvalidOffset ? 0u : LongestFreeRun(block.Live);
        if (!run) return;
        entry = {block.Owner, run};
        Available.insert({block.Owner, run, b});
    }
    // Return blocks with free runs of at least minimum, ordered by run length, until their free capacity reaches slots.
    // Entries that disagree with their block's metadata are refreshed first.
    std::vector<uint32_t> AvailableBlocks(ElementSetRef set, uint32_t minimum, uint32_t slots) {
        while (true) {
            std::vector<uint32_t> blocks, stale;
            uint32_t free = 0u;
            for (auto it = Available.lower_bound({set.Index, minimum, 0u}); it != Available.end() && (*it)[0] == set.Index && free < slots; ++it) {
                const auto b = (*it)[2];
                const auto *block = b < Blocks.Buffer.template Count<MeshElementBlock>() ? &Blocks.Get({b, 1})[0] : nullptr;
                if (!block || block->Owner != set.Index || LongestFreeRun(block->Live) != (*it)[1]) stale.push_back(b);
                else {
                    blocks.push_back(b);
                    free += BlockSize - block->Count;
                }
            }
            if (stale.empty()) return blocks;
            for (const auto b : stale) Index(b);
        }
    }
    // Appends the table entries naming `blocks`, or InvalidOffset entries for an InvalidOffset offset, to the header's run, growing the run when full.
    // Returns the local index of the first appended entry.
    uint32_t Append(MeshElementSet &header, Range blocks) {
        const auto local = header.BlockCount;
        if (local + blocks.Count > header.Capacity) {
            auto allocation = Table.BeginAllocation();
            const auto run = Table.Allocate(std::max(local + blocks.Count, 2u * header.Capacity));
            std::ranges::copy(Table.Get({header.First, local}), Table.GetMutable({run.Offset, local}).begin());
            Table.Release({header.First, header.Capacity});
            header.First = run.Offset;
            header.Capacity = run.Count;
            allocation.Commit();
        }
        const auto entries = Table.GetMutable({header.First + local, blocks.Count});
        for (uint32_t i = 0u; i < blocks.Count; ++i) entries[i] = blocks.Offset == InvalidOffset ? InvalidOffset : blocks.Offset + i;
        header.BlockCount += blocks.Count;
        return local;
    }
    // Place blocks in the first fitting table gap or append them, returning the first local index.
    uint32_t Fill(MeshElementSet &header, Range blocks) {
        const auto entries = Table.Get({header.First, header.BlockCount});
        const auto hole = std::ranges::search_n(entries, blocks.Count, InvalidOffset);
        if (hole.empty()) return Append(header, blocks);
        const auto local = uint32_t(hole.begin() - entries.begin());
        const auto filled = Table.GetMutable({header.First + local, blocks.Count});
        for (uint32_t i = 0u; i < blocks.Count; ++i) filled[i] = blocks.Offset + i;
        return local;
    }
    // Clears the masked slots of the set's blocks, gives the set the flags, and returns the blocks in ascending order.
    // Release empty blocks except the first, then trim empty trailing table entries.
    std::vector<uint32_t> ClearSlots(ElementSetRef set, std::vector<std::pair<uint32_t, LiveMask>> cleared, uint32_t flags) {
        std::ranges::sort(cleared, {}, &std::pair<uint32_t, LiveMask>::first);
        auto header = Set(set);
        header.Flags = flags;
        std::vector<uint32_t> blocks, emptied;
        blocks.reserve(cleared.size());
        // Consecutive blocks capture their metadata in one write.
        for (size_t first = 0u; first < cleared.size();) {
            auto end = first + 1u;
            while (end < cleared.size() && cleared[end].first == cleared[end - 1u].first + 1u) ++end;
            const auto members = Blocks.GetMutable({cleared[first].first, uint32_t(end - first)});
            for (size_t i = first; i < end; ++i) {
                const auto &[b, mask] = cleared[i];
                auto &block = members[i - first];
                for (uint32_t w = 0; w < BlockWords; ++w) block.Live[w] &= ~mask[w];
                const auto count = LiveCount(block.Live);
                header.Count -= block.Count - count;
                block.Count = count;
                blocks.push_back(b);
                if (!count && block.Local) emptied.push_back(b);
            }
            first = end;
        }
        if (!emptied.empty()) {
            const auto metadata = Blocks.Buffer.template GetSpan<MeshElementBlock>();
            for (const auto b : emptied) Table.GetMutable({header.First + metadata[b].Local, 1u})[0] = InvalidOffset;
            const auto entries = Table.Get({header.First, header.BlockCount});
            while (header.BlockCount > 1u && entries[header.BlockCount - 1u] == InvalidOffset) --header.BlockCount;
        }
        ++header.Revision;
        Sets.GetMutable({set.Index, 1})[0] = header;
        ForEachIndexRun(emptied, [&](size_t first, size_t count) {
            const Range range{emptied[first], uint32_t(count)};
            std::ranges::fill(Blocks.GetMutable(range), MeshElementBlock{});
            Blocks.Release(range);
        });
        for (const auto b : blocks) Index(b);
        return blocks;
    }
    // Each listed block's handles are checked against the set's table where the work is applied.
    uint32_t CheckWork(const mtl::Buffer &storage, ElementWork work) const {
        if (work.Storage.Slot != storage.Slot || work.Storage.Slot == InvalidSlot || !std::has_single_bit(work.Capacity)) {
            throw std::invalid_argument("Element mutation requires a sparse work table in its supplied buffer.");
        }
        const uint64_t words = WorkHeaderWords + uint64_t(work.Capacity) * (WorkBlockWords + 2u);
        if ((uint64_t(work.Storage.Offset) + words) * 4u > storage.UsedSize) {
            throw std::out_of_range("Element mutation work exceeds its storage.");
        }
        const auto data = storage.GetSpan<uint32_t>({work.Storage.Offset, WorkHeaderWords});
        if (data[1] || data[0] > work.Capacity || data[0] != data[2]) throw std::invalid_argument("Element mutation work is incomplete or overflowed.");
        return data[0];
    }
    void ReserveBlocks(uint32_t blocks) {
        if (!blocks) return;
        Blocks.ReserveAdditional(blocks);
        Buffer.Reserve((uint64_t(BlockCount(uint32_t(Buffer.UsedSize / sizeof(T)))) + blocks) * BlockSize * sizeof(T));
    }
    Range AllocateBlocks(uint32_t count) {
        if (uint64_t(count) * BlockSize > InvalidOffset) throw std::length_error("Canonical element handle space exhausted.");
        auto allocation = Blocks.BeginAllocation();
        const auto blocks = Blocks.Allocate(count);
        const uint64_t first = uint64_t(blocks.Offset) * BlockSize, capacity = uint64_t(blocks.Count) * BlockSize;
        if (first + capacity > InvalidOffset) throw std::length_error("Canonical element handle space exhausted.");
        Mirror({uint32_t(first), uint32_t(capacity)});
        allocation.Commit();
        return blocks;
    }
};
