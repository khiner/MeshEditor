#ifndef ELEMENT_WORK_SHARED_MSL
#define ELEMENT_WORK_SHARED_MSL

#include "Bindless.metal"
#include "ElementWork.metal"

inline uint WorkWordCount(ElementWork work) { return (work.Count + 31u) / 32u; }

inline uint WorkElement(device const BindlessSet &bindless, ElementWork work, uint invocation) {
    if (work.Storage.Slot == INVALID_SLOT) return invocation < work.Count ? invocation : INVALID_OFFSET;
    device atomic_uint *data = BindlessBufferMutable(atomic_uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    const uint words = WorkWordCount(work);
    // The finalized arguments freeze the input word count while expansion appends output words.
    if (invocation / 32u >= atomic_load_explicit(data + words * 2u + 4u, memory_order_relaxed) / 32u) return INVALID_OFFSET;
    const uint word = atomic_load_explicit(data + words + invocation / 32u, memory_order_relaxed);
    const uint bit = invocation % 32u;
    return (atomic_load_explicit(data + word, memory_order_relaxed) & (1u << bit)) != 0u ? word * 32u + bit : INVALID_OFFSET;
}

inline void MarkWork(device const BindlessSet &bindless, ElementWork work, uint element) {
    if (element >= work.Count || work.Storage.Slot == INVALID_SLOT) return;
    device atomic_uint *data = BindlessBufferMutable(atomic_uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    const uint words = WorkWordCount(work), word = element / 32u;
    if (atomic_fetch_or_explicit(data + word, 1u << (element % 32u), memory_order_relaxed) == 0u) {
        const uint index = atomic_fetch_add_explicit(data + words * 2u, 1u, memory_order_relaxed);
        atomic_store_explicit(data + words + index, word, memory_order_relaxed);
    }
}

inline void FinishWork(device const BindlessSet &bindless, ElementWork work) {
    if (work.Storage.Slot == INVALID_SLOT) return;
    device uint *data = BindlessBufferMutable(uint, bindless.Buffer, work.Storage.Slot) + work.Storage.Offset;
    const uint words = WorkWordCount(work);
    data[words * 2u + 1u] = (data[words * 2u] + 7u) / 8u;
    data[words * 2u + 2u] = 1u;
    data[words * 2u + 3u] = 1u;
    data[words * 2u + 4u] = data[words * 2u] * 32u;
    data[words * 2u + 5u] = data[words * 2u + 6u] = 1u;
}

#endif
