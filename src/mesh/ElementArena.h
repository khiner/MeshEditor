#pragma once
#include "mesh/ElementView.h"

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/BindlessBindings.h"
#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/MeshElementSet.h"
#include "metal/Buffer.h"
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
// The i-th handle of a run, or of a list stored in `list`.
inline uint32_t ElementHandleAt(ElementHandleRange handles, std::span<const uint32_t> list, uint32_t i) {
    return handles.Handles.Slot == InvalidSlot ? handles.First + i : list[handles.Handles.Offset + i];
}

// A domain owns geometry once. Sets own linked blocks with live masks, written on the CPU and read by kernels.
// A Derived index of each set's blocks with free slots, ordered by their longest free run, places insertions.
template<typename T>
struct ElementArena {
    ElementArena(mtl::BufferContext &ctx, SlotType slot_type) : Buffer(ctx, 0, slot_type), Blocks(ctx, SlotType::Buffer), Sets(ctx) {}

    void Track(store::History &history, const std::string &name) {
        Blocks.Track(history, name + ".blocks");
        Sets.Track(history, name + ".sets");
        Buffer.Track(history, name + ".bytes");
    }
    static uint32_t BlockCount(uint32_t count) { return count / MeshElementBlockSize + uint32_t(count % MeshElementBlockSize != 0u); }
    const MeshElementSet &Set(ElementSetRef set) const {
        if (set.Index == InvalidOffset || set.Index >= Sets.Buffer.template Count<MeshElementSet>()) throw std::invalid_argument("Invalid element set.");
        const auto &value = Sets.Get({set.Index, 1})[0];
        if (value.First == InvalidOffset) throw std::invalid_argument("Retired element set.");
        return value;
    }
    uint32_t First(ElementSetRef set) const { return set ? Set(set).First * MeshElementBlockSize : 0u; }
    uint32_t Capacity() const { return Blocks.HighWaterMark() * MeshElementBlockSize; }
    uint32_t Count(ElementSetRef set) const { return set ? Set(set).Count : 0u; }
    Range Dense(ElementSetRef set) const {
        if (!set) return {};
        const auto &header = Set(set);
        if (header.Count && !(header.Flags & 1u)) throw std::invalid_argument("Element set has no dense view.");
        return {header.First * MeshElementBlockSize, header.Count};
    }
    // Visits owned blocks in allocation order, including the first retained empty block.
    void ForEachBlock(ElementSetRef set, auto &&visit) const {
        if (!set) return;
        const auto blocks = Membership(set);
        for (auto b = Set(set).First; b != InvalidOffset; b = blocks[b].Next) visit(b, blocks[b]);
    }
    void ForEach(ElementSetRef set, auto &&visit) const {
        uint32_t ordinal = 0u;
        ForEachBlock(set, [&](uint32_t b, const auto &block) {
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                for (auto bits = block.Live[w]; bits; bits &= bits - 1u)
                    visit(b * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits)), ordinal++);
        });
    }

    void ReserveAdditional(uint32_t count) { ReserveBlocks(BlockCount(count)); }
    void PlanAdditional(uint32_t count) { PlannedBlocks += BlockCount(count); }
    void CommitPlanned() { ReserveBlocks(std::exchange(PlannedBlocks, 0u)); }
    void Mirror(Range range) {
        if (range.Count) Buffer.SetUsedSize(std::max(Buffer.UsedSize, (uint64_t(range.Offset) + range.Count) * sizeof(T)));
    }

    // A dense set whose first `count` slots are live.
    ElementSetRef Create(uint32_t count = 0) {
        auto block_allocation = Blocks.BeginAllocation();
        auto set_allocation = Sets.BeginAllocation();
        const auto blocks = AllocateBlocks(std::max(1u, BlockCount(count)));
        const ElementSetRef set{Sets.Allocate(1).Offset};
        auto members = Blocks.GetMutable(blocks);
        for (uint32_t i = 0; i < blocks.Count; ++i) {
            const auto live = std::min(MeshElementBlockSize, count > i * MeshElementBlockSize ? count - i * MeshElementBlockSize : 0u);
            members[i] = {.Next = i + 1u < blocks.Count ? blocks.Offset + i + 1u : InvalidOffset, .Previous = i ? blocks.Offset + i - 1u : InvalidOffset,
                          .Owner = set.Index, .Count = live, .Live = Prefix(live)};
        }
        Sets.GetMutable({set.Index, 1})[0] = {.First = blocks.Offset, .Last = blocks.Offset + blocks.Count - 1u, .Count = count, .BlockCount = blocks.Count, .Flags = 1u};
        Index(blocks.Offset + blocks.Count - 1u);
        block_allocation.Commit();
        set_allocation.Commit();
        return set;
    }

    // Adds `count` live elements to the set, creating it as a dense set when absent, and returns their handles and the blocks that gained them.
    // Without a list the handles form one run, and otherwise they form a run whenever they are consecutive and are listed in `list` when they are not.
    // At most one block of handles fills free slots of the set's own blocks, and more take fresh blocks.
    ElementInsert Insert(ElementSetRef &set, uint32_t count, mtl::Buffer *list) {
        ElementInsert result;
        if (list) list->SetUsedSize(0);
        if (!count) return result;
        if (uint64_t(Count(set)) + count >= InvalidOffset) throw std::length_error("Element count overflow.");
        if (!set) {
            set = Create(count);
            const auto &header = Set(set);
            for (auto b = header.First; b <= header.Last; ++b) result.Blocks.push_back({b, Blocks.Get({b, 1})[0].Live});
            result.Handles = {.First = header.First * MeshElementBlockSize, .Count = count};
            return result;
        }
        auto header = Set(set);
        const bool run = !list;
        std::vector<uint32_t> handles;
        uint32_t remaining = count;
        if (count <= MeshElementBlockSize) {
            for (const auto b : AvailableBlocks(set, run ? count : 1u, run ? 1u : count)) {
                auto &block = Blocks.GetMutable({b, 1})[0];
                auto &gain = result.Blocks.emplace_back(ElementBlockGain{b, {}});
                if (run) {
                    const auto first = FreeRunStart(block.Live, count);
                    for (uint32_t i = 0; i < count; ++i) gain.Added[(first + i) / 32u] |= 1u << ((first + i) % 32u);
                    handles.push_back(b * MeshElementBlockSize + first);
                    remaining = 0u;
                } else {
                    for (uint32_t w = 0; w < MeshElementBlockWords && remaining; ++w)
                        for (auto free = ~block.Live[w]; free && remaining; free &= free - 1u, --remaining) {
                            const auto bit = uint32_t(std::countr_zero(free));
                            gain.Added[w] |= 1u << bit;
                            handles.push_back(b * MeshElementBlockSize + w * 32u + bit);
                        }
                }
                for (uint32_t w = 0; w < MeshElementBlockWords; ++w) block.Live[w] |= gain.Added[w];
                block.Count = LiveCount(block.Live);
                if (!remaining) break;
            }
        }
        if (remaining) {
            auto allocation = Blocks.BeginAllocation();
            const auto fresh = AllocateBlocks(BlockCount(remaining));
            Blocks.GetMutable({header.Last, 1})[0].Next = fresh.Offset;
            auto members = Blocks.GetMutable(fresh);
            for (uint32_t i = 0; i < fresh.Count; ++i) {
                const auto live = std::min(remaining, MeshElementBlockSize);
                members[i] = {.Next = i + 1u < fresh.Count ? fresh.Offset + i + 1u : InvalidOffset, .Previous = i ? fresh.Offset + i - 1u : header.Last,
                              .Owner = set.Index, .Count = live, .Live = Prefix(live)};
                result.Blocks.push_back({fresh.Offset + i, members[i].Live});
                if (count <= MeshElementBlockSize) for (uint32_t s = 0; s < live; ++s) handles.push_back((fresh.Offset + i) * MeshElementBlockSize + s);
                remaining -= live;
            }
            if (count > MeshElementBlockSize) handles.push_back(fresh.Offset * MeshElementBlockSize);
            header.Last = fresh.Offset + fresh.Count - 1u;
            header.BlockCount += fresh.Count;
            allocation.Commit();
        }
        header.Count += count;
        ++header.Revision;
        header.Flags = 0u;
        Sets.GetMutable({set.Index, 1})[0] = header;
        for (const auto &gain : result.Blocks) Index(gain.Block);
        const bool consecutive = handles.size() == 1u || (handles.back() - handles.front() == count - 1u &&
            std::ranges::adjacent_find(handles, [](uint32_t a, uint32_t b) { return b != a + 1u; }) == handles.end());
        if (consecutive) {
            result.Handles = {.First = handles.front(), .Count = count};
        } else {
            if (list->Slot == InvalidSlot) throw std::invalid_argument("A handle list requires a bound buffer.");
            std::ranges::copy(handles, list->SetCount<uint32_t>(count).begin());
            result.Handles = {.Handles = {list->Slot, 0u}, .Count = count};
        }
        return result;
    }

    // Clears the live bits the sparse work table names and returns the ascending blocks it names.
    // Blocks left empty leave the set, except its first block.
    // Work naming another owner's block or a slot outside its domain is rejected before any write.
    std::vector<uint32_t> Erase(ElementSetRef set, const mtl::Buffer &storage, ElementWork work) {
        const auto occupied=CheckWork(storage,work);
        if (!occupied) return {};
        auto header = Set(set);
        const auto data = storage.GetSpan<uint32_t>({work.Storage.Offset, WorkHeaderWords + work.Capacity * (WorkBlockWords + 2u)});
        const auto entry = [&](uint32_t i) { return data.subspan(WorkHeaderWords + data[WorkHeaderWords + work.Capacity * WorkBlockWords + i] * WorkBlockWords, WorkBlockWords); };
        for (uint32_t i = 0; i < occupied; ++i) {
            const auto slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + i];
            const auto key = slot < work.Capacity ? data[WorkHeaderWords + slot * WorkBlockWords] : 0u;
            const auto block = key - 1u;
            bool valid = key && block < Blocks.HighWaterMark() && uint64_t(block) * MeshElementBlockSize < work.Count && Blocks.Get({block, 1})[0].Owner == set.Index;
            for (uint32_t w = 0; valid && w < MeshElementBlockWords; ++w) {
                const auto first = uint64_t(block) * MeshElementBlockSize + w * 32u;
                const auto available = uint32_t(std::min<uint64_t>(32u, work.Count > first ? work.Count - first : 0u));
                valid = !(entry(i)[1u + w] & ~(available == 32u ? ~0u : (1u << available) - 1u));
            }
            if (!valid) throw std::invalid_argument("Element removal work has invalid or foreign block membership.");
        }
        std::vector<std::pair<uint32_t, LiveMask>> cleared;
        cleared.reserve(occupied);
        for (uint32_t i = 0; i < occupied; ++i) {
            const auto removal = entry(i);
            std::ranges::copy(removal.subspan(1u, MeshElementBlockWords), cleared.emplace_back(removal[0] - 1u, LiveMask{}).second.begin());
        }
        std::ranges::sort(cleared, {}, &std::pair<uint32_t, LiveMask>::first);
        header.Flags = 0u;
        return ClearSlots(set, header, cleared);
    }

    void Destroy(ElementSetRef set) {
        Destroy(std::span{&set,1u});
    }
    void Destroy(std::span<const ElementSetRef> sets) {
        if (sets.empty()) return;
        std::vector<uint32_t> blocks;
        for (const auto set : sets) ForEachBlock(set, [&](uint32_t b, const auto &) { blocks.push_back(b); });
        if (!std::ranges::is_sorted(blocks)) std::ranges::sort(blocks);
        Destroy(sets, blocks);
    }
    // The caller already compiled and sorted the membership of these sets.
    void Destroy(std::span<const ElementSetRef> sets, std::span<const uint32_t> blocks) {
        if (sets.empty()) return;
        std::vector<uint32_t> owners;
        owners.reserve(sets.size());
        for (const auto set : sets) owners.push_back(set.Index);
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
            if (range.Offset < end) std::fill(Indexed.begin() + range.Offset, Indexed.begin() + end, std::pair{InvalidOffset, 0u});
            Blocks.Release(range);
        });
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
    ElementSetRef Allocate(std::span<const T> values) {
        const auto set = Allocate(uint32_t(values.size()));
        Buffer.Update(as_bytes(values), uint64_t(First(set)) * sizeof(T));
        return set;
    }
    // Keeps the first `used` handles of an insertion into the set, a run or listed in `list`, clears the others, and shrinks `inserted` to the used ones.
    // A dense set's allocation is its only insertion, so it stays dense.
    // Blocks left without elements leave the set, except its first block, and a set left without elements retires.
    // Returns the ascending blocks that lost elements.
    std::vector<uint32_t> Shrink(ElementSetRef &set, ElementHandleRange &inserted, std::span<const uint32_t> list, uint32_t used) {
        if (used >= inserted.Count) return {};
        std::vector<std::pair<uint32_t, LiveMask>> cleared;
        if (inserted.Handles.Slot == InvalidSlot) {
            const uint64_t first = inserted.First + used, end = uint64_t(inserted.First) + inserted.Count;
            for (auto b = first / MeshElementBlockSize; b * MeshElementBlockSize < end; ++b) {
                const auto base = b * MeshElementBlockSize;
                const auto kept = Prefix(uint32_t(std::max(first, base) - base)), covered = Prefix(uint32_t(std::min<uint64_t>(end - base, MeshElementBlockSize)));
                auto &mask = cleared.emplace_back(uint32_t(b), LiveMask{}).second;
                for (uint32_t w = 0; w < MeshElementBlockWords; ++w) mask[w] = covered[w] & ~kept[w];
            }
        } else {
            std::vector<uint32_t> handles(list.begin() + inserted.Handles.Offset + used, list.begin() + inserted.Handles.Offset + inserted.Count);
            std::ranges::sort(handles);
            for (const auto handle : handles) {
                const auto b = handle / MeshElementBlockSize, slot = handle % MeshElementBlockSize;
                if (cleared.empty() || cleared.back().first != b) cleared.emplace_back(b, LiveMask{});
                cleared.back().second[slot / 32u] |= 1u << (slot % 32u);
            }
        }
        inserted.Count = used;
        const auto blocks = ClearSlots(set, Set(set), cleared);
        if (!Count(set)) {
            Destroy(set);
            set = {};
        }
        return blocks;
    }
    void Release(ElementSetRef set) { if (set) Destroy(set); }
    void Reset() {
        Blocks.Reset(); Sets.Reset(); Buffer.SetUsedSize(0);
        Indexed.clear(); Available.clear();
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
    std::span<const MeshElementBlock> Membership(ElementSetRef set) const {
        return set ? Blocks.Buffer.template GetSpan<MeshElementBlock>() : std::span<const MeshElementBlock>{};
    }

    mtl::Buffer Buffer;
    BufferArena<MeshElementBlock> Blocks;
    BufferArena<MeshElementSet> Sets;

private:
    using LiveMask = std::array<uint32_t, MeshElementBlockWords>;
    uint32_t PlannedBlocks{};
    // Derived: the owner and longest free run each block is indexed under, and the indexed blocks ordered by owner, run and block.
    std::vector<std::pair<uint32_t, uint32_t>> Indexed;
    std::set<std::array<uint32_t, 3>> Available;

    static LiveMask Prefix(uint32_t count) {
        LiveMask live{};
        for (uint32_t w = 0; w < MeshElementBlockWords; ++w) {
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
        for (uint32_t i = 0u; i < MeshElementBlockSize;) {
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
        if (first != InvalidOffset) visit(first, MeshElementBlockSize - first);
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
    // The set's indexed blocks whose longest free run is at least `minimum`, in ascending run order, until they hold `slots` free slots.
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
                    free += MeshElementBlockSize - block->Count;
                }
            }
            if (stale.empty()) return blocks;
            for (const auto b : stale) Index(b);
        }
    }
    // Clears the masked slots of the set's ascending blocks, publishes `header` with them, and returns the blocks.
    // Blocks left without elements leave the set, except its first block.
    std::vector<uint32_t> ClearSlots(ElementSetRef set, MeshElementSet header, std::span<const std::pair<uint32_t, LiveMask>> cleared) {
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
                for (uint32_t w = 0; w < MeshElementBlockWords; ++w) block.Live[w] &= ~mask[w];
                const auto count = LiveCount(block.Live);
                header.Count -= block.Count - count;
                block.Count = count;
                blocks.push_back(b);
                if (!count && b != header.First) emptied.push_back(b);
            }
            first = end;
        }
        // Each chain segment of emptied blocks unlinks at its two ends.
        const auto metadata = Blocks.Buffer.template GetSpan<MeshElementBlock>();
        for (size_t i = 0u; i < emptied.size();) {
            auto end = i + 1u;
            while (end < emptied.size() && metadata[emptied[end - 1u]].Next == emptied[end]) ++end;
            const auto previous = metadata[emptied[i]].Previous, next = metadata[emptied[end - 1u]].Next;
            if (previous != InvalidOffset) Blocks.GetMutable({previous, 1})[0].Next = next;
            if (next != InvalidOffset) Blocks.GetMutable({next, 1})[0].Previous = previous;
            else header.Last = previous;
            header.BlockCount -= uint32_t(end - i);
            i = end;
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
    uint32_t CheckWork(const mtl::Buffer &storage,ElementWork work) const {
        if (work.Storage.Slot != storage.Slot || work.Storage.Slot == InvalidSlot || !std::has_single_bit(work.Capacity)) {
            throw std::invalid_argument("Element mutation requires a sparse work table in its supplied buffer.");
        }
        const uint64_t words=WorkHeaderWords+uint64_t(work.Capacity)*(WorkBlockWords+2u);
        if ((uint64_t(work.Storage.Offset)+words)*4u>storage.UsedSize || work.Count>Capacity()) {
            throw std::out_of_range("Element mutation work exceeds its storage or canonical domain.");
        }
        const auto header=storage.GetSpan<uint32_t>({work.Storage.Offset,WorkHeaderWords});
        if (header[1] || header[0]>work.Capacity || header[0]!=header[2]) throw std::invalid_argument("Element mutation work is incomplete or overflowed.");
        return header[0];
    }
    void ReserveBlocks(uint32_t blocks) {
        if (!blocks) return;
        Blocks.ReserveAdditional(blocks);
        Buffer.Reserve((uint64_t(BlockCount(uint32_t(Buffer.UsedSize / sizeof(T)))) + blocks) * MeshElementBlockSize * sizeof(T));
    }
    Range AllocateBlocks(uint32_t count) {
        if (uint64_t(count) * MeshElementBlockSize > InvalidOffset) throw std::length_error("Canonical element handle space exhausted.");
        auto allocation = Blocks.BeginAllocation();
        const auto blocks = Blocks.Allocate(count);
        const uint64_t first = uint64_t(blocks.Offset) * MeshElementBlockSize, capacity = uint64_t(blocks.Count) * MeshElementBlockSize;
        if (first + capacity > InvalidOffset) throw std::length_error("Canonical element handle space exhausted.");
        Mirror({uint32_t(first), uint32_t(capacity)});
        allocation.Commit();
        return blocks;
    }
};
