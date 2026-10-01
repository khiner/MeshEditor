#ifndef ELEMENT_WORK_SHARED_MSL
#define ELEMENT_WORK_SHARED_MSL

#include "Bindless.metal"
#include "gpu/ElementWork.h"

inline uint WorkElement(device const BindlessSet &bindless, ElementWork work, uint invocation) {
    if (work.Storage.Slot == InvalidSlot) return invocation < work.Count ? invocation : InvalidOffset;
    device const uint *data = BindlessBuffer(uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    if (invocation / 256u >= data[2]) return InvalidOffset;
    const uint slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + invocation / 256u];
    device const uint *block = data + WorkHeaderWords + slot * WorkBlockWords;
    const uint bit = invocation % 256u;
    return block[1u + bit / 32u] & (1u << (bit % 32u)) ? (block[0] - 1u) * 256u + bit : InvalidOffset;
}

// Return true only to the invocation that first inserts this element. Sparse
// graph traversals use this to enqueue each dependency once.
inline bool InsertWork(device const BindlessSet &bindless, ElementWork work, uint element) {
    if (element >= work.Count || work.Storage.Slot == InvalidSlot) return false;
    device atomic_uint *data = BindlessBufferMutable(atomic_uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    const uint key = element / 256u + 1u;
    uint slot = WorkHash(key - 1u, work.Capacity), probe = 0u;
    while (probe < work.Capacity) {
        device atomic_uint *block = data + WorkHeaderWords + slot * WorkBlockWords;
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(block, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            const uint index = atomic_fetch_add_explicit(data, 1u, memory_order_relaxed);
            atomic_store_explicit(data + WorkHeaderWords + work.Capacity * WorkBlockWords + index, slot, memory_order_relaxed);
            expected = key;
        }
        if (expected == key) {
            const uint mask = 1u << (element % 32u);
            return !(atomic_fetch_or_explicit(block + 1u + (element % 256u) / 32u, mask, memory_order_relaxed) & mask);
        }
        if (!expected) continue; // Weak CAS may fail spuriously.
        ++probe;
        slot = (slot + 1u) & (work.Capacity - 1u);
    }
    atomic_store_explicit(data + 1u, 1u, memory_order_relaxed);
    return false;
}

inline void MarkWork(device const BindlessSet &bindless, ElementWork work, uint element) {
    InsertWork(bindless,work,element);
}

// Cooperative reductions need one group per live element, rather than one
// group for every bit of a sparse block. Prefixes are over occupied blocks only.
inline uint WorkGroupElement(device const BindlessSet &bindless, ElementWork work, uint ordinal) {
    if (work.Storage.Slot == InvalidSlot) return ordinal < work.Count ? ordinal : InvalidOffset;
    device const uint *data = BindlessBuffer(uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    if (ordinal >= data[5]) return InvalidOffset;
    device const uint *prefix = data + WorkHeaderWords + work.Capacity * (WorkBlockWords + 1u);
    device const uint *slots = data + WorkHeaderWords + work.Capacity * WorkBlockWords;
    uint lo = 0u, hi = data[2];
    while (lo < hi) { const uint middle = lo + (hi - lo) / 2u; if (prefix[slots[middle]] <= ordinal) lo = middle + 1u; else hi = middle; }
    uint rank = ordinal - (lo ? prefix[slots[lo - 1u]] : 0u);
    const uint slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + lo];
    device const uint *block = data + WorkHeaderWords + slot * WorkBlockWords;
    return SelectLiveElement(block + 1u, block[0] - 1u, rank);
}

// Inverse of compact enumeration. Prefixes live at hash slots so lookup
// needs no second directory or scan over the address domain.
inline uint WorkRank(device const BindlessSet &bindless, ElementWork work, uint element) {
    if (element >= work.Count) return InvalidOffset;
    if (work.Storage.Slot == InvalidSlot) return element;
    device const uint *data = BindlessBuffer(uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    const uint key = element / 256u + 1u;
    uint slot = WorkHash(key - 1u, work.Capacity);
    for (uint probe = 0u; probe < work.Capacity; ++probe, slot = (slot + 1u) & (work.Capacity - 1u)) {
        device const uint *block = data + WorkHeaderWords + slot * WorkBlockWords;
        if (!block[0]) return InvalidOffset;
        if (block[0] != key) continue;
        const uint word = (element % 256u) / 32u, bit = element % 32u;
        if (!(block[word + 1u] & (1u << bit))) return InvalidOffset;
        uint rank = data[WorkHeaderWords + work.Capacity * (WorkBlockWords + 1u) + slot];
        for (uint w = word + 1u; w < 8u; ++w) rank -= popcount(block[w + 1u]);
        return rank - popcount(block[word + 1u] & (~0u << bit));
    }
    return InvalidOffset;
}

inline void FinishWork(device const BindlessSet &bindless, ElementWork work, uint tid, threadgroup uint *totals) {
    if (work.Storage.Slot == InvalidSlot) return;
    device uint *data = BindlessBufferMutable(uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    device uint *prefix = data + WorkHeaderWords + work.Capacity * (WorkBlockWords + 1u);
    const uint slice = (data[0] + 255u) / 256u, begin = tid * slice, end = min(begin + slice, data[0]);
    uint sum = 0u;
    for (uint i = begin; i < end; ++i) {
        const uint slot = data[WorkHeaderWords + work.Capacity * WorkBlockWords + i];
        for (uint word = 1u; word <= 8u; ++word) sum += popcount(data[WorkHeaderWords + slot * WorkBlockWords + word]);
        prefix[slot] = sum;
    }
    const uint scanned = simd_prefix_inclusive_sum(sum);
    if ((tid & 31u) == 31u) totals[tid / 32u] = scanned;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = scanned - sum;
    for (uint i = 0u; i < tid / 32u; ++i) before += totals[i];
    for (uint i = begin; i < end; ++i) prefix[data[WorkHeaderWords + work.Capacity * WorkBlockWords + i]] += before;
    if (tid == 255u) data[5] = before + sum;
    if (tid == 0u) { data[2] = data[0]; data[3] = data[4] = data[6] = data[7] = 1u; }
}

// Source work contains indices and membership only. Polygon traversal keeps
// canonical halfedges so a loop can cross independently enumerated work blocks.
struct ElementWorkDomain {
    device const BindlessSet &B;
    ElementWork Work;
    uint Origin;
    uint Handle(uint index) const {
        const uint local = WorkGroupElement(B, Work, index);
        return local == InvalidOffset ? local : Origin + local;
    }
    uint Index(uint handle) const { return handle < Origin || handle == InvalidOffset ? InvalidOffset : WorkRank(B, Work, handle - Origin); }
};

#endif
