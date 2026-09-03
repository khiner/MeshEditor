#ifndef EDITSELECTIONTRANSACTION_MSL
#define EDITSELECTIONTRANSACTION_MSL

// Every output reads the current element-domain mask directly, so topology conversion requires no grid-wide barrier.
#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "EditSelectionSummary.metal"
#include "EditSelectionOperation.metal"
#include "Element.metal"
#include "FanItemEncoding.metal"
#include "EditSelectionPushConstants.metal"

constant uint INVALID_HANDLE = 0xffffffffu;
constant uint SelectionSharp = 1u;
constant uint SelectionSmooth = 2u;

struct EditSelectionContext {
    device const BindlessSet &B;
    constant EditSelectionPushConstants &Pc;

    SlotOffset SelectionRange(uint element) const {
        return element == Element_Vertex ? Pc.Selection.VertexBits :
            element == Element_Edge ? Pc.Selection.EdgeBits : Pc.Selection.FaceBits;
    }
    device EditSelectionSummary &Summary() const { return BindlessBufferMutable(EditSelectionSummary, B.Buffer, Pc.Selection.Summary.Slot)[Pc.Selection.Summary.Offset]; }
    uint ElementCount(uint element) const {
        return element == Element_Vertex ? Pc.VertexCount :
            element == Element_Edge ? Pc.EdgeCount : Pc.FaceCount;
    }
    uint PickedLocal() const {
        if (Pc.PickIdSlot == INVALID_SLOT) return INVALID_HANDLE;
        const SlotOffset source = SelectionRange(Pc.Element);
        const uint pick_id = BindlessBuffer(uint, B.Buffer, Pc.PickIdSlot)[0];
        if (pick_id == 0u || pick_id == INVALID_HANDLE) return INVALID_HANDLE;
        const uint picked_global = pick_id - 1u;
        const uint base = source.Offset * 32u;
        return picked_global >= base && picked_global < base + ElementCount(Pc.Element) ? picked_global - base : INVALID_HANDLE;
    }
    bool SourceSelected(uint element) const {
        if (element >= ElementCount(Pc.Element)) return false;
        const SlotOffset range = SelectionRange(Pc.Element);
        const uint word = BindlessBuffer(uint, B.Buffer, range.Slot)[range.Offset + (element >> 5u)];
        return ((word >> (element & 31u)) & 1u) != 0u;
    }

    device const uint *EdgeIndices() const { return BindlessBuffer(uint, B.IndexBuffer, Pc.EdgeIndices.Slot) + Pc.EdgeIndices.Offset; }
    device const uint *Corners() const { return BindlessBuffer(uint, B.IndexBuffer, Pc.Corners.Slot) + Pc.Corners.Offset; }
    device const uint *Connectivity() const { return BindlessBuffer(uint, B.Buffer, Pc.Connectivity.Slot) + Pc.Connectivity.Offset; }
    device const uint *Adjacency() const { return BindlessBuffer(uint, B.Buffer, Pc.AdjacencySlot); }
    device const uchar *FaceSharpness() const { return BindlessBuffer(uchar, B.Buffer, Pc.FaceSharpness.Slot) + Pc.FaceSharpness.Offset; }
    device const uchar *EdgeSharpness() const { return BindlessBuffer(uchar, B.Buffer, Pc.EdgeSharpness.Slot) + Pc.EdgeSharpness.Offset; }
    device const Vertex *Vertices() const { return BindlessBuffer(Vertex, B.VertexBuffer, Pc.Vertices.Slot) + Pc.Vertices.Offset; }

    device const uint *Opposites() const { return ConnectivityOpposites(Connectivity(), Pc.VertexCount); }
    device const uint *EdgeFirstBits() const { return Opposites() + Pc.HalfedgeCount; }
    device const uint *EdgeFirstRanks() const { return EdgeFirstBits() + ConnectivityWordCount(Pc.HalfedgeCount); }
    device const uint *EdgeHalfedges() const { return BindlessBuffer(uint, B.Buffer, Pc.EdgeHalfedges.Slot) + Pc.EdgeHalfedges.Offset; }

    uint2 FaceHalfedges(uint face) const {
        return ConnectivityFaceHalfedges(Connectivity(), Pc.VertexCount, Pc.HalfedgeCount, Pc.FaceCount, Pc.ConnectivityFaceStarts != 0u, face);
    }
    uint HalfedgeFace(uint halfedge) const {
        return ConnectivityHalfedgeFace(Connectivity(), Pc.VertexCount, Pc.HalfedgeCount, Pc.FaceCount, Pc.ConnectivityFaceStarts != 0u, halfedge);
    }
    uint HalfedgeEdge(uint halfedge) const {
        if (Pc.HalfedgeToEdge.Offset != INVALID_HANDLE) {
            return BindlessBuffer(uint, B.Buffer, Pc.HalfedgeToEdge.Slot)[Pc.HalfedgeToEdge.Offset + halfedge];
        }
        const uint opposite = Opposites()[halfedge];
        const uint first = opposite != INVALID_HANDLE && opposite < halfedge ? opposite : halfedge;
        const uint word = first >> 5u;
        const uint preceding = (1u << (first & 31u)) - 1u;
        return EdgeFirstRanks()[word] + popcount(EdgeFirstBits()[word] & preceding);
    }

    bool VertexIncidentSelected(uint vertex_id, uint offset, uint item_mask) const {
        if (offset == INVALID_HANDLE) return false;
        device const uint *a = Adjacency() + offset;
        const uint items = Pc.VertexCount + 1u;
        for (uint i = a[vertex_id]; i < a[vertex_id + 1u]; ++i) {
            if (SourceSelected(a[items + i] & item_mask)) return true;
        }
        return false;
    }
    bool VertexSelected(uint vertex_id) const {
        if (Pc.Element == Element_Vertex) return SourceSelected(vertex_id);
        const bool face = Pc.Element == Element_Face;
        return VertexIncidentSelected(
            vertex_id, face ? Pc.VertexFanAdjacencyOffset : Pc.VertexEdgeAdjacencyOffset,
            face ? FanItemEncoding_FaceMask : 0xffffffffu
        );
    }

    bool EdgeAdjacentSelectedFace(uint edge) const {
        const uint h = EdgeHalfedges()[edge];
        if (SourceSelected(HalfedgeFace(h))) return true;
        const uint opposite = Opposites()[h];
        return opposite != INVALID_HANDLE && SourceSelected(HalfedgeFace(opposite));
    }
    bool EdgeSelected(uint edge) const {
        if (Pc.Element == Element_Vertex) {
            return SourceSelected(EdgeIndices()[2u * edge]) && SourceSelected(EdgeIndices()[2u * edge + 1u]);
        }
        if (Pc.Element == Element_Edge) return SourceSelected(edge);
        return EdgeAdjacentSelectedFace(edge);
    }
    bool EdgeTouchesSelection(uint edge) const {
        if (Pc.Element == Element_Vertex) {
            return SourceSelected(EdgeIndices()[2u * edge]) || SourceSelected(EdgeIndices()[2u * edge + 1u]);
        }
        return SourceSelected(edge);
    }

    bool FaceSelected(uint face) const {
        if (Pc.Element == Element_Face) return SourceSelected(face);
        const uint2 halfedges = FaceHalfedges(face);
        for (uint h = halfedges.x; h < halfedges.y; ++h) {
            const uint source = Pc.Element == Element_Vertex ? Corners()[h] : HalfedgeEdge(h);
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
    if (pc.Operation == EditSelectionOperation_Clear || pc.Operation == EditSelectionOperation_FillList ||
        pc.Operation == EditSelectionOperation_PickReplace) {
        new_word = 0u;
    } else if (pc.Operation == EditSelectionOperation_Fill) {
        const uint remaining = count - word_index * 32u;
        new_word = remaining >= 32u ? 0xffffffffu : (1u << remaining) - 1u;
    } else if (pc.Operation == EditSelectionOperation_CaptureBaseline) {
        baseline[word_index] = old_word;
        if (word_index == 0u) baseline[word_count] = ctx.Summary().ActiveHandle;
    } else if (pc.Operation == EditSelectionOperation_RestoreBaseline) {
        new_word = baseline[word_index];
    }

    if (pc.Operation == EditSelectionOperation_PickReplace || pc.Operation == EditSelectionOperation_PickToggle) {
        const uint picked_local = ctx.PickedLocal();
        if (picked_local != INVALID_HANDLE && (picked_local >> 5u) == word_index) {
            const uint bit = 1u << (picked_local & 31u);
            if (pc.Operation == EditSelectionOperation_PickToggle && ctx.Summary().ActiveHandle == picked_local) {
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
    if (pc.Operation == EditSelectionOperation_Clear || pc.Operation == EditSelectionOperation_FillList ||
        pc.Operation == EditSelectionOperation_ClearActive || pc.Operation == EditSelectionOperation_PickReplace) {
        summary.ActiveHandle = picked_local;
    } else if (pc.Operation == EditSelectionOperation_RestoreBaseline) {
        summary.ActiveHandle = BindlessBuffer(uint, bindless.Buffer, pc.SelectionBaseline.Slot)[
            pc.SelectionBaseline.Offset + (ctx.ElementCount(pc.Element) + 31u) / 32u
        ];
    } else if (pc.Operation == EditSelectionOperation_PickToggle && picked_local != INVALID_HANDLE) {
        summary.ActiveHandle = summary.ActiveHandle == picked_local ? INVALID_HANDLE : picked_local;
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
    if (pc.Element != Element_Vertex && chunk_index < vertex_chunks && (chunk_index & 1u) == 0u) {
        ctx.WriteSelectionWord(pc.Selection.VertexBits, chunk_index / 2u, vertex_bits | (vertex_partner_bits << 16u));
    }

    const uint edge_chunks = (pc.EdgeCount + 15u) / 16u;
    uint edge_bits = 0u;
    if (chunk_index < edge_chunks) {
        for (uint k = 0u; k < 16u; ++k) {
            const uint edge = chunk_index * 16u + k;
            if (edge >= pc.EdgeCount) break;
            if (ctx.EdgeSelected(edge)) edge_bits |= 1u << k;
            if (pc.Element != Element_Face && ctx.EdgeTouchesSelection(edge)) {
                sharpness_flags |= ctx.EdgeSharpness()[edge] != 0u ? SelectionSharp : SelectionSmooth;
            }
        }
    }
    const uint edge_partner_bits = simd_shuffle_xor(edge_bits, 1u);
    if (pc.Element != Element_Edge && chunk_index < edge_chunks && (chunk_index & 1u) == 0u) {
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
                if (pc.Element == Element_Face) {
                    sharpness_flags |= ctx.FaceSharpness()[face] != 0u ? SelectionSharp : SelectionSmooth;
                }
            }
        }
    }
    const uint face_partner_bits = simd_shuffle_xor(face_bits, 1u);
    if (pc.Element != Element_Face && chunk_index < face_chunks && (chunk_index & 1u) == 0u) {
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
        device atomic_float *position = (device atomic_float *)&summary.PositionSum;
        atomic_fetch_add_explicit(position, simd_position.x, memory_order_relaxed);
        atomic_fetch_add_explicit(position + 1, simd_position.y, memory_order_relaxed);
        atomic_fetch_add_explicit(position + 2, simd_position.z, memory_order_relaxed);
    }
}

#endif
