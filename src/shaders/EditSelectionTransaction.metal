#ifndef EDITSELECTIONTRANSACTION_MSL
#define EDITSELECTIONTRANSACTION_MSL

// Every output reads the current element-domain mask directly, so topology conversion requires no grid-wide barrier.
#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "gpu/EditSelectionSummary.h"
#include "gpu/EditSelectionOperation.h"
#include "gpu/Element.h"
#include "gpu/FanItemEncoding.h"
#include "gpu/EditSelectionPushConstants.h"

constant uint SelectionSharp = 1u;
constant uint SelectionSmooth = 2u;

struct EditSelectionContext {
    device const BindlessSet &B;
    constant EditSelectionPushConstants &Pc;

    SlotOffset SelectionRange(Element element) const {
        return element == Element::Vertex ? Pc.Selection.VertexBits :
            element == Element::Edge ? Pc.Selection.EdgeBits : Pc.Selection.FaceBits;
    }
    device EditSelectionSummary &Summary() const { return BindlessBufferMutable(EditSelectionSummary, B.Buffer, Pc.Selection.Summary.Slot)[Pc.Selection.Summary.Offset]; }
    uint ElementCount(Element element) const {
        return element == Element::Vertex ? Pc.VertexCount :
            element == Element::Edge ? Pc.EdgeCount : Pc.FaceCount;
    }
    uint PickedLocal() const {
        if (Pc.PickIdSlot == InvalidSlot) return InvalidOffset;
        const SlotOffset source = SelectionRange(Pc.Element);
        const uint pick_id = BindlessBuffer(uint, B.Buffer, Pc.PickIdSlot)[0];
        if (pick_id == 0u || pick_id == InvalidOffset) return InvalidOffset;
        const uint picked_global = pick_id - 1u;
        const uint base = source.Offset * 32u;
        return picked_global >= base && picked_global < base + ElementCount(Pc.Element) ? picked_global - base : InvalidOffset;
    }
    bool SourceSelected(uint element) const {
        if (element >= ElementCount(Pc.Element)) return false;
        const SlotOffset range = SelectionRange(Pc.Element);
        const uint word = BindlessBuffer(uint, B.Buffer, range.Slot)[range.Offset + (element >> 5u)];
        return ((word >> (element & 31u)) & 1u) != 0u;
    }

    device const uint *EdgeIndices() const { return BindlessBuffer(uint, B.IndexBuffer, Pc.EdgeIndices.Slot) + Pc.EdgeIndices.Offset; }
    device const uint *Corners() const { return BindlessBuffer(uint, B.IndexBuffer, Pc.Corners.Slot) + Pc.Corners.Offset; }
    ConnectivityView Connectivity() const {
        return {BindlessBuffer(uint, B.Buffer, Pc.Connectivity.Slot) + Pc.Connectivity.Offset, Pc.VertexCount, Pc.HalfedgeCount, Pc.FaceCount, Pc.ConnectivityFaceStarts != 0u};
    }
    device const uint *Adjacency() const { return BindlessBuffer(uint, B.Buffer, Pc.AdjacencySlot); }
    device const uchar *FaceSharpness() const { return BindlessBuffer(uchar, B.Buffer, Pc.FaceSharpness.Slot) + Pc.FaceSharpness.Offset; }
    device const uchar *EdgeSharpness() const { return BindlessBuffer(uchar, B.Buffer, Pc.EdgeSharpness.Slot) + Pc.EdgeSharpness.Offset; }
    device const Vertex *Vertices() const { return BindlessBuffer(Vertex, B.VertexBuffer, Pc.Vertices.Slot) + Pc.Vertices.Offset; }

    bool VertexIncidentSelected(uint vertex_id, uint offset, uint item_mask) const {
        if (offset == InvalidOffset) return false;
        device const uint *a = Adjacency() + offset;
        const uint items = Pc.VertexCount + 1u;
        for (uint i = a[vertex_id]; i < a[vertex_id + 1u]; ++i) {
            if (SourceSelected(a[items + i] & item_mask)) return true;
        }
        return false;
    }
    bool VertexSelected(uint vertex_id) const {
        if (Pc.Element == Element::Vertex) return SourceSelected(vertex_id);
        const bool face = Pc.Element == Element::Face;
        return VertexIncidentSelected(
            vertex_id, face ? Pc.VertexFanAdjacencyOffset : Pc.VertexEdgeAdjacencyOffset,
            face ? uint(FanItemEncoding::FaceMask) : 0xffffffffu
        );
    }

    bool EdgeAdjacentSelectedFace(uint edge) const {
        const auto conn = Connectivity();
        const uint h = conn.EdgeHalfedge(edge);
        if (SourceSelected(conn.HalfedgeFace(h))) return true;
        const uint opposite = conn.Opposite(h);
        return opposite != InvalidOffset && SourceSelected(conn.HalfedgeFace(opposite));
    }
    bool EdgeSelected(uint edge) const {
        if (Pc.Element == Element::Vertex) {
            return SourceSelected(EdgeIndices()[2u * edge]) && SourceSelected(EdgeIndices()[2u * edge + 1u]);
        }
        if (Pc.Element == Element::Edge) return SourceSelected(edge);
        return EdgeAdjacentSelectedFace(edge);
    }
    bool EdgeTouchesSelection(uint edge) const {
        if (Pc.Element == Element::Vertex) {
            return SourceSelected(EdgeIndices()[2u * edge]) || SourceSelected(EdgeIndices()[2u * edge + 1u]);
        }
        return SourceSelected(edge);
    }

    bool FaceSelected(uint face) const {
        if (Pc.Element == Element::Face) return SourceSelected(face);
        const auto conn = Connectivity();
        const uint2 halfedges = conn.FaceHalfedges(face);
        for (uint h = halfedges.x; h < halfedges.y; ++h) {
            const uint source = Pc.Element == Element::Vertex ? Corners()[h] : conn.Edge(h);
            if (!SourceSelected(source)) return false;
        }
        return true;
    }

    void WriteSelectionWord(SlotOffset range, uint word, uint value) const {
        BindlessBufferMutable(uint, B.Buffer, range.Slot)[range.Offset + word] = value;
    }
};

kernel void PrepareEditSelectionKernel(
    uint word_index [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const EditSelectionContext ctx{bindless, pc};
    const SlotOffset source = ctx.SelectionRange(pc.Element);
    const uint count = ctx.ElementCount(pc.Element);
    const uint word_count = (count + 31u) / 32u;
    if (word_index >= word_count) return;

    device uint *selection = BindlessBufferMutable(uint, bindless.Buffer, source.Slot) + source.Offset;
    device uint *baseline = BindlessBufferMutable(uint, bindless.Buffer, pc.SelectionBaseline.Slot) + pc.SelectionBaseline.Offset;
    const uint old_word = selection[word_index];
    uint new_word = old_word;
    if (pc.Operation == EditSelectionOperation::Clear || pc.Operation == EditSelectionOperation::FillList ||
        pc.Operation == EditSelectionOperation::PickReplace) {
        new_word = 0u;
    } else if (pc.Operation == EditSelectionOperation::Fill) {
        const uint remaining = count - word_index * 32u;
        new_word = remaining >= 32u ? 0xffffffffu : (1u << remaining) - 1u;
    } else if (pc.Operation == EditSelectionOperation::CaptureBaseline) {
        baseline[word_index] = old_word;
        if (word_index == 0u) baseline[word_count] = ctx.Summary().ActiveHandle;
    } else if (pc.Operation == EditSelectionOperation::RestoreBaseline) {
        new_word = baseline[word_index];
    }

    if (pc.Operation == EditSelectionOperation::PickReplace || pc.Operation == EditSelectionOperation::PickToggle) {
        const uint picked_local = ctx.PickedLocal();
        if (picked_local != InvalidOffset && (picked_local >> 5u) == word_index) {
            const uint bit = 1u << (picked_local & 31u);
            if (pc.Operation == EditSelectionOperation::PickToggle && ctx.Summary().ActiveHandle == picked_local) {
                new_word &= ~bit;
            } else {
                new_word |= bit;
            }
        }
    }
    if (new_word != old_word) selection[word_index] = new_word;
}

kernel void FillEditSelectionListKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.SelectionListCount) return;
    const uint element = BindlessBuffer(uint, bindless.Buffer, pc.SelectionList.Slot)[pc.SelectionList.Offset + i];
    const EditSelectionContext ctx{bindless, pc};
    if (element >= ctx.ElementCount(pc.Element)) return;
    const SlotOffset range = ctx.SelectionRange(pc.Element);
    device atomic_uint *words = BindlessBufferMutable(atomic_uint, bindless.Buffer, range.Slot) + range.Offset;
    atomic_fetch_or_explicit(words + (element >> 5u), 1u << (element & 31u), memory_order_relaxed);
}

kernel void ResetEditSelectionSummaryKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i != 0u) return;
    const EditSelectionContext ctx{bindless, pc};
    device EditSelectionSummary &summary = ctx.Summary();
    const uint picked_local = ctx.PickedLocal();
    if (pc.Operation == EditSelectionOperation::Clear || pc.Operation == EditSelectionOperation::FillList ||
        pc.Operation == EditSelectionOperation::ClearActive || pc.Operation == EditSelectionOperation::PickReplace) {
        summary.ActiveHandle = picked_local;
    } else if (pc.Operation == EditSelectionOperation::RestoreBaseline) {
        summary.ActiveHandle = BindlessBuffer(uint, bindless.Buffer, pc.SelectionBaseline.Slot)[
            pc.SelectionBaseline.Offset + (ctx.ElementCount(pc.Element) + 31u) / 32u
        ];
    } else if (pc.Operation == EditSelectionOperation::PickToggle && picked_local != InvalidOffset) {
        summary.ActiveHandle = summary.ActiveHandle == picked_local ? InvalidOffset : picked_local;
    }
    summary.PositionSum = packed_float3(float3(0.0f));
    summary.Mode = pc.Element;
    summary.SelectedCount = 0u;
    summary.SelectedVertexCount = 0u;
    summary.SharpnessFlags = 0u;
}

kernel void DeriveEditSelectionKernel(
    uint chunk_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const EditSelectionContext ctx{bindless, pc};
    uint selected_count = 0u, selected_vertex_count = 0u, sharpness_flags = 0u;
    float3 position_sum{0.0f};

    const uint source_words = (ctx.ElementCount(pc.Element) + 31u) / 32u;
    if ((chunk_index & 1u) == 0u && chunk_index / 2u < source_words) {
        const SlotOffset source = ctx.SelectionRange(pc.Element);
        selected_count = popcount(BindlessBuffer(uint, bindless.Buffer, source.Slot)[source.Offset + chunk_index / 2u]);
    }

    const uint vertex_chunks = (pc.VertexCount + 15u) / 16u;
    uint vertex_bits = 0u;
    if (chunk_index < vertex_chunks) {
        for (uint k = 0u; k < 16u; ++k) {
            const uint vertex_id = chunk_index * 16u + k;
            if (vertex_id >= pc.VertexCount) break;
            const bool selected = ctx.VertexSelected(vertex_id);
            if (selected) vertex_bits |= 1u << k;
            if (selected) position_sum += float3(ctx.Vertices()[vertex_id].Position);
        }
        selected_vertex_count = popcount(vertex_bits);
    }
    const uint vertex_partner_bits = simd_shuffle_xor(vertex_bits, 1u);
    if (pc.Element != Element::Vertex && chunk_index < vertex_chunks && (chunk_index & 1u) == 0u) {
        ctx.WriteSelectionWord(pc.Selection.VertexBits, chunk_index / 2u, vertex_bits | (vertex_partner_bits << 16u));
    }

    const uint edge_chunks = (pc.EdgeCount + 15u) / 16u;
    uint edge_bits = 0u;
    if (chunk_index < edge_chunks) {
        for (uint k = 0u; k < 16u; ++k) {
            const uint edge = chunk_index * 16u + k;
            if (edge >= pc.EdgeCount) break;
            if (ctx.EdgeSelected(edge)) edge_bits |= 1u << k;
            if (pc.Element != Element::Face && ctx.EdgeTouchesSelection(edge)) {
                sharpness_flags |= ctx.EdgeSharpness()[edge] != 0u ? SelectionSharp : SelectionSmooth;
            }
        }
    }
    const uint edge_partner_bits = simd_shuffle_xor(edge_bits, 1u);
    if (pc.Element != Element::Edge && chunk_index < edge_chunks && (chunk_index & 1u) == 0u) {
        ctx.WriteSelectionWord(pc.Selection.EdgeBits, chunk_index / 2u, edge_bits | (edge_partner_bits << 16u));
    }

    const uint face_chunks = (pc.FaceCount + 15u) / 16u;
    uint face_bits = 0u;
    if (chunk_index < face_chunks) {
        for (uint k = 0u; k < 16u; ++k) {
            const uint face = chunk_index * 16u + k;
            if (face >= pc.FaceCount) break;
            if (ctx.FaceSelected(face)) {
                face_bits |= 1u << k;
                if (pc.Element == Element::Face) {
                    sharpness_flags |= ctx.FaceSharpness()[face] != 0u ? SelectionSharp : SelectionSmooth;
                }
            }
        }
    }
    const uint face_partner_bits = simd_shuffle_xor(face_bits, 1u);
    if (pc.Element != Element::Face && chunk_index < face_chunks && (chunk_index & 1u) == 0u) {
        ctx.WriteSelectionWord(pc.Selection.FaceBits, chunk_index / 2u, face_bits | (face_partner_bits << 16u));
    }

    const uint simd_selected = simd_sum(selected_count);
    const uint simd_vertices = simd_sum(selected_vertex_count);
    const uint simd_sharpness = simd_or(sharpness_flags);
    const float3 simd_position = float3(simd_sum(position_sum.x), simd_sum(position_sum.y), simd_sum(position_sum.z));
    if (lane == 0u) {
        device EditSelectionSummary &summary = BindlessBufferMutable(EditSelectionSummary, bindless.Buffer, pc.Selection.Summary.Slot)[pc.Selection.Summary.Offset];
        if (simd_selected != 0u) atomic_fetch_add_explicit((device atomic_uint *)&summary.SelectedCount, simd_selected, memory_order_relaxed);
        if (simd_vertices != 0u) atomic_fetch_add_explicit((device atomic_uint *)&summary.SelectedVertexCount, simd_vertices, memory_order_relaxed);
        if (simd_sharpness != 0u) atomic_fetch_or_explicit((device atomic_uint *)&summary.SharpnessFlags, simd_sharpness, memory_order_relaxed);
        // Each SIMD group owns one partial; the final reduction has a fixed order.
        if (chunk_index < vertex_chunks) {
            BindlessBufferMutable(packed_float3, bindless.Buffer, pc.PositionSumsSlot)[chunk_index / 32u] = packed_float3(simd_position);
        }
    }
}

// One SIMD group reduces the vertex partials without schedule-dependent float atomics.
kernel void SumEditSelectionPositionKernel(
    uint lane [[thread_index_in_simdgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant EditSelectionPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint count = (pc.VertexCount + 511u) / 512u;
    device const packed_float3 *partials = BindlessBuffer(packed_float3, bindless.Buffer, pc.PositionSumsSlot);
    float3 sum{0.0f};
    for (uint i = lane; i < count; i += 32u) sum += float3(partials[i]);
    sum = float3(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z));
    if (lane == 0u) EditSelectionContext{bindless, pc}.Summary().PositionSum = packed_float3(sum);
}

#endif
