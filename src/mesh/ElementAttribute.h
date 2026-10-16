#pragma once

#include "mesh/ElementAttributeView.h"
#include "metal/BufferArena.h"

// Grows a record's range of a block arena to `blocks` blocks: in place when the storage after it is free, and otherwise at twice its size or to fit, moving its values.
// Values keep their offsets in the range, so a move changes no reference.
// Returns whether the range changed.
template<typename Block> bool CoverRange(BufferArena<Block> &arena, Range &range, uint32_t blocks) {
    if (blocks <= range.Count) return false;
    if (range.Count && arena.Extend(range, blocks - range.Count)) return true;
    // The released range coalesces with its free neighbors, so a fit overlapping it starts at or before it.
    if (range.Count) arena.Release(range);
    const auto grown = arena.Allocate(std::max(2u * range.Count, blocks));
    arena.Buffer.Move(uint64_t(range.Offset) * sizeof(Block), uint64_t(grown.Offset) * sizeof(Block), uint64_t(range.Count) * sizeof(Block));
    range = grown;
    return true;
}
// Writes `value` to the range's blocks that `blocks` names, capturing them.
template<typename Block> void FillRange(BufferArena<Block> &arena, Range range, Range blocks, const Block &value) {
    if (blocks.Count) std::ranges::fill(arena.GetMutable({range.Offset + blocks.Offset, blocks.Count}), value);
}

// One element layer, whose values each record carrying it holds in one range of the layer's blocks.
// A record's range covers its handle bound in the layer's domain with `entries` blocks per local block, and entry e of handle h sits at slot h % MeshElementBlockSize of the range's block (h / MeshElementBlockSize) * entries + e.
// `Absent` is the value of an element whose source gave the layer none, which an insertion, a join or a boolean gives the elements it adds without one.
template<typename T>
struct ElementAttribute {
    using Block = std::array<T, MeshElementBlockSize>;
    ElementAttribute(mtl::BufferContext &ctx, SlotType type, T absent = T{}) : Absent{absent}, Values(ctx, type) {}

    // The range's binding, which kernels index by handle, or no slot for an empty range.
    SlotOffset Ref(Range range) const { return range.Count ? SlotOffset{Values.Buffer.Slot, range.Offset * MeshElementBlockSize} : SlotOffset{}; }
    // A single-entry range's values by handle.
    ElementAttributeView<T> View(Range range) const { return {Values.Buffer.template GetSpan<T>({range.Offset * MeshElementBlockSize, range.Count * MeshElementBlockSize})}; }
    // The value index of entry `entry` of a handle in the range.
    static uint32_t Index(Range range, uint32_t handle, uint32_t entry = 0u, uint32_t entries = 1u) {
        return (range.Offset + handle / MeshElementBlockSize * entries + entry) * MeshElementBlockSize + handle % MeshElementBlockSize;
    }
    const T &Get(Range range, uint32_t handle, uint32_t entry = 0u, uint32_t entries = 1u) const { return Values.Buffer.template GetSpan<T>({Index(range, handle, entry, entries), 1u})[0]; }
    // The values of a single-entry range at a handle run, captured for a write.
    std::span<T> Edit(Range range, Range handles) { return Values.Buffer.template GetMutableSpan<T>({Index(range, handles.Offset), handles.Count}); }
    // Captures every entry of the handle run before their values change, and nothing for an empty range.
    void Capture(Range range, Range handles, uint32_t entries = 1u) const {
        if (!range.Count || !handles.Count) return;
        if (entries == 1u) return Values.Buffer.CaptureWrite(uint64_t(Index(range, handles.Offset)) * sizeof(T), uint64_t(handles.Count) * sizeof(T));
        const auto first = handles.Offset / MeshElementBlockSize, last = (handles.Offset + handles.Count - 1u) / MeshElementBlockSize;
        Values.CaptureWrite({range.Offset + first * entries, (last + 1u - first) * entries});
    }
    // Grows the range to cover `blocks` local blocks of `entries` entries, and returns whether it changed.
    bool Cover(Range &range, uint32_t blocks, uint32_t entries = 1u) { return CoverRange(Values, range, blocks * entries); }
    // Gives every entry of the local blocks the absent value.
    void Clear(Range range, Range locals, uint32_t entries = 1u) {
        Block absent;
        absent.fill(Absent);
        FillRange(Values, range, {locals.Offset * entries, locals.Count * entries}, absent);
    }
    // Import/initialization only.
    // Covers a dense set's `count` handles from handle 0, holding `source` entry-major over them, or the absent value without one.
    void Initialize(Range &range, uint32_t count, std::span<const T> source = {}, uint32_t entries = 1u) {
        const auto blocks = (count + MeshElementBlockSize - 1u) / MeshElementBlockSize;
        Cover(range, blocks, entries);
        Clear(range, {0u, blocks}, entries);
        if (source.empty()) return;
        const auto values = Values.Buffer.template GetMutableSpan<T>({range.Offset * MeshElementBlockSize, blocks * entries * MeshElementBlockSize});
        for (uint32_t e = 0u; e < entries; ++e) {
            for (uint32_t h = 0u; h < count; h += MeshElementBlockSize) std::copy_n(source.begin() + uint64_t(e) * count + h, std::min(MeshElementBlockSize, count - h), values.begin() + Index({0u, 0u}, h, e, entries));
        }
    }
    void Release(Range &range) {
        if (range.Count) Values.Release(range);
        range = {};
    }

    T Absent;
    BufferArena<Block> Values;
};
