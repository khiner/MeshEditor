#pragma once

#include "Range.h"
#include "gpu/BindlessBindings.h"
#include "gpu/ElementAttributeRef.h"
#include "mesh/ElementAttributeView.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <vector>

// Attribute payload exists only for attached element blocks.
// An attached element block owns one or more consecutive payload blocks, its entries, and entry e of handle h is at Index(h, e).
// The forward table names each element block's first payload block plus one, and the reverse table names each payload block's element block plus one.
// Both tables are allocation metadata, and values have one canonical copy.
template<typename T>
struct ElementAttribute {
    using Block = std::array<T, MeshElementBlockSize>;
    ElementAttribute(mtl::BufferContext &ctx, SlotType type) : Values(ctx, type), Blocks(ctx, SlotType::Buffer), Owners(ctx) {}

    ElementAttributeRef Ref(bool present = true) const { return present ? ElementAttributeRef{Blocks.Buffer.Slot, Values.Buffer.Slot} : ElementAttributeRef{}; }
    ElementAttributeView<T> View(bool present = true) const {
        return present ? ElementAttributeView<T>{Blocks.Buffer.template GetSpan<uint32_t>(), Values.Buffer.template GetSpan<T>()} : ElementAttributeView<T>{};
    }
    uint32_t PayloadBlock(uint32_t block) const {
        return block < Blocks.Buffer.template Count<uint32_t>() ? Blocks.Get({block, 1})[0] : 0u;
    }
    uint32_t EntryCount(uint32_t block) const {
        const auto first = PayloadBlock(block);
        if (!first) return 0u;
        const auto owners = Owners.Buffer.template GetSpan<uint32_t>();
        uint32_t count = 1u;
        while (first - 1u + count < owners.size() && owners[first - 1u + count] == block + 1u) ++count;
        return count;
    }
    // Grow with the owning domain, so enabling a layer at a high handle does
    // not initialize an arena-sized lookup table during that edit.
    void ReserveBlocks(uint32_t count) {
        const auto before = Blocks.Buffer.template Count<uint32_t>();
        if (count <= before) return;
        Blocks.Mirror({0, count});
        std::ranges::fill(Blocks.GetMutable({before, count - before}), 0u);
    }
    // Binds `entries` payload blocks holding `value` to each listed block without a payload.
    // Each block's payload is its own first-fit allocation, so released payloads are reused.
    void Attach(std::span<const uint32_t> blocks, const T &value = T{}, uint32_t entries = 1u) {
        if (blocks.empty()) return;
        if (!entries) throw std::invalid_argument("An attribute payload needs at least one entry.");
        std::vector<uint32_t> unbound{blocks.begin(), blocks.end()};
        std::ranges::sort(unbound);
        unbound.erase(std::unique(unbound.begin(), unbound.end()), unbound.end());
        ReserveBlocks(unbound.back() + 1u);
        std::erase_if(unbound, [&](uint32_t block) { return PayloadBlock(block) != 0u; });
        if (unbound.empty()) return;
        std::vector<Range> payloads;
        payloads.reserve(unbound.size());
        auto allocation = Values.BeginAllocation();
        Values.ReserveAdditional(uint32_t(unbound.size()) * entries);
        for (size_t i = 0u; i < unbound.size(); ++i) payloads.push_back(Values.Allocate(entries));
        allocation.Commit();
        Owners.Mirror({0u, Values.HighWaterMark()});
        Owners.Buffer.CaptureWriteRanges(payloads, sizeof(uint32_t));
        Values.Buffer.CaptureWriteRanges(payloads, sizeof(Block));
        auto *owners = reinterpret_cast<uint32_t *>(Owners.Buffer.Contents().data());
        auto *values = reinterpret_cast<Block *>(Values.Buffer.Contents().data());
        Block filled;
        filled.fill(value);
        ForEachIndexRun(unbound, [&](size_t first, size_t count) {
            auto targets = Blocks.GetMutable({unbound[first], uint32_t(count)});
            for (size_t i = 0u; i < count; ++i) {
                const auto j = first + i;
                targets[i] = payloads[j].Offset + 1u;
                std::fill_n(owners + payloads[j].Offset, entries, unbound[j] + 1u);
                std::fill_n(values + payloads[j].Offset, entries, filled);
            }
        });
    }
    void Release(uint32_t block) {
        Release(std::span{&block, 1u});
    }
    void Release(std::span<const uint32_t> blocks) {
        std::vector<Range> payloads;
        std::vector<uint32_t> attached;
        const auto bindings = Blocks.Buffer.template GetSpan<uint32_t>();
        const auto owners = Owners.Buffer.template GetSpan<uint32_t>();
        for (const auto block : blocks) {
            if (const auto first = block < bindings.size() ? bindings[block] : 0u) {
                uint32_t count = 1u;
                while (first - 1u + count < owners.size() && owners[first - 1u + count] == block + 1u) ++count;
                payloads.push_back({first - 1u, count});
                attached.push_back(block);
            }
        }
        std::ranges::sort(attached);
        ForEachIndexRun(attached, [&](size_t first, size_t count) {
            std::ranges::fill(Blocks.GetMutable({attached[first], uint32_t(count)}), 0u);
        });
        CoalesceRanges(payloads);
        for (const auto run : payloads) {
            std::ranges::fill(Owners.GetMutable(run), 0u);
            Values.Release(run);
        }
    }
    // The value index of `handle`'s entry, below its block's entry count.
    uint32_t Index(uint32_t handle, uint32_t entry = 0u) const {
        const auto first = PayloadBlock(handle / MeshElementBlockSize);
        if (!first) throw std::out_of_range("Attribute payload is absent.");
        return (first - 1u + entry) * MeshElementBlockSize + handle % MeshElementBlockSize;
    }
    Range Payload(uint32_t handle, uint32_t count = 1, uint32_t entry = 0u) const {
        if (count > MeshElementBlockSize - handle % MeshElementBlockSize) throw std::out_of_range("Attribute range crosses an element block.");
        return {Index(handle, entry), count};
    }
    const T &Get(uint32_t handle, uint32_t entry = 0u) const { return Values.Buffer.template GetSpan<T>({Index(handle, entry), 1})[0]; }
    std::span<T> Edit(uint32_t handle, uint32_t count = 1) {
        return Values.Buffer.template GetMutableSpan<T>(Payload(handle, count));
    }
    // Captures every entry of the attached handles before their values change.
    void CaptureHandles(Range handles, uint32_t entries = 1u) const {
        Values.Buffer.CaptureWriteRanges(PayloadRanges(handles, entries), sizeof(T));
    }
    // Import/initialization only.
    // Editing writes affected payload ranges on GPU.
    // `source` is entry-major over the handle run, and an empty source writes defaults.
    void Initialize(Range handles, std::span<const T> source = {}, uint32_t entries = 1u) {
        if (!handles.Count) return;
        if (!source.empty() && source.size() != uint64_t(handles.Count) * entries) throw std::invalid_argument("Attribute input count differs from its elements.");
        std::vector<uint32_t> blocks;
        for (auto b = handles.Offset / MeshElementBlockSize; b <= (handles.Offset + handles.Count - 1u) / MeshElementBlockSize; ++b) blocks.push_back(b);
        Attach(blocks, T{}, entries);
        CaptureHandles(handles, entries);
        auto *values = reinterpret_cast<T *>(Values.Buffer.Contents().data());
        for (uint32_t e = 0u; e < entries; ++e)
            for (uint32_t i = 0u; i < handles.Count;) {
                const auto h = handles.Offset + i;
                const auto count = std::min(handles.Count - i, MeshElementBlockSize - h % MeshElementBlockSize);
                auto *out = values + Index(h, e);
                if (source.empty()) std::fill_n(out, count, T{});
                else std::copy_n(source.begin() + uint64_t(e) * handles.Count + i, count, out);
                i += count;
            }
    }
    BufferArena<Block> Values;
    BufferArena<uint32_t> Blocks, Owners;

private:
    std::vector<Range> PayloadRanges(Range handles, uint32_t entries) const {
        std::vector<Range> ranges;
        for (uint32_t i = 0u; i < handles.Count;) {
            const auto h = handles.Offset + i;
            const auto count = std::min(handles.Count - i, MeshElementBlockSize - h % MeshElementBlockSize);
            for (uint32_t e = 0u; e < entries; ++e) ranges.push_back({Index(h, e), count});
            i += count;
        }
        return ranges;
    }
};
