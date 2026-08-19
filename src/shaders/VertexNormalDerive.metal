#ifndef VERTEXNORMALDERIVE_MSL
#define VERTEXNORMALDERIVE_MSL

// Derives shading normals in two phases, one threadgroup per 256-element tile.
// Phase 0 computes each face's normalized vector-area normal over the tile map's face-tile prefix.
// Positions come from the entry's posed range when it has one, else the vertex arena.
// Phase 1 gathers per-vertex smooth normals over the vertex-fan CSR and sector normals over the seam CSR.
// Each fan item contributes the phase-0 unit face normal weighted by its corner angle (Blender's weighting).
// Sum orders are fixed by the CSR item and fan order, so results are bit-stable across frames.
// The push constants select the dispatch's position and output buffers.
#include "Bindless.metal"
#include "NormalDeriveEntry.metal"
#include "FanItemEncoding.metal"
#include "NormalDerivePushConstants.metal"

// Everything the derive reads and writes, resolved once per thread.
struct DeriveContext {
    Scene S;
    constant NormalDerivePushConstants &Pc;

    device const uint *Adjacency() const { return BindlessBuffer(uint, S.B.Buffer, Pc.AdjacencySlot); }
    device packed_float3 *FaceNormals() const { return BindlessBufferMutable(packed_float3, S.B.Buffer, Pc.FaceNormalSlot); }
    device packed_float3 *VertexNormals() const { return BindlessBufferMutable(packed_float3, S.B.Buffer, Pc.VertexNormalSlot); }
    device packed_float3 *SeamNormals() const { return BindlessBufferMutable(packed_float3, S.B.Buffer, Pc.SeamNormalSlot); }

    // Vertex `i`'s current position: the entry's posed range when it has one, else the vertex arena.
    float3 Position(NormalDeriveEntry entry, uint i) const {
        return entry.PosedPositionOffset != INVALID_OFFSET ?
            float3(S.PosedPositions(Pc.PositionSlot)[entry.PosedPositionOffset + i]) :
            float3(S.Vertices(entry.Vertices.Slot)[entry.Vertices.Offset + i].Position);
    }

    // Unnormalized cross product of triangle `t`'s positions.
    float3 TriangleCross(NormalDeriveEntry entry, uint t) const {
        device const uint *indices = S.Indices(entry.FaceIndices.Slot);
        const uint i0 = indices[entry.FaceIndices.Offset + t * 3u];
        const uint i1 = indices[entry.FaceIndices.Offset + t * 3u + 1u];
        const uint i2 = indices[entry.FaceIndices.Offset + t * 3u + 2u];
        return cross(Position(entry, i1) - Position(entry, i0), Position(entry, i2) - Position(entry, i0));
    }

    // Vertex index at loop position `j` of the face whose fan triangles span [first, first + valence - 2).
    // Fan triangle `t` holds corners [loop 0, loop t - first + 1, loop t - first + 2].
    uint LoopVertexIndex(NormalDeriveEntry entry, uint first, uint valence, uint j) const {
        device const uint *indices = S.Indices(entry.FaceIndices.Slot);
        const uint base = entry.FaceIndices.Offset;
        return j == 0u ? indices[base + first * 3u] :
            j < valence - 1u ? indices[base + (first + j - 1u) * 3u + 1u] :
                               indices[base + (first + valence - 3u) * 3u + 2u];
    }

    // Triangle range [first, end) of face `f`, from the face-first-triangle table.
    // The last face runs to the entry's triangle count.
    uint2 FaceTriangleRange(NormalDeriveEntry entry, uint f) const {
        device const uint *face_first = S.ObjectIds(Pc.FaceFirstTriangleSlot);
        const uint first = face_first[entry.FaceDataOffset + f];
        const uint end = f + 1u < entry.FaceCount ? face_first[entry.FaceDataOffset + f + 1u] : entry.TriangleCount;
        return uint2(first, end);
    }

    // One fan item's contribution: the phase-0 face normal weighted by its corner angle at the loop
    // position. Zero for degenerate faces and corners.
    float3 FanContribution(NormalDeriveEntry entry, uint item) const {
        const uint f = item & FanItemEncoding_FaceMask;
        const uint k = item >> FanItemEncoding_LoopShift;
        const float3 fn = float3(FaceNormals()[entry.FaceNormalOffset + f]);
        if (all(fn == float3(0))) return float3(0);
        const uint2 range = FaceTriangleRange(entry, f);
        const uint first = range.x;
        const uint valence = range.y - range.x + 2u;
        const float3 p = Position(entry, LoopVertexIndex(entry, first, valence, k));
        const float3 e_prev = Position(entry, LoopVertexIndex(entry, first, valence, (k + valence - 1u) % valence)) - p;
        const float3 e_next = Position(entry, LoopVertexIndex(entry, first, valence, (k + 1u) % valence)) - p;
        const float d = length(e_prev) * length(e_next);
        if (d == 0.0f) return float3(0);
        return fn * acos(clamp(dot(e_prev, e_next) / d, -1.0f, 1.0f));
    }

    // Normalized sum of the fan contributions a CSR bucket lists.
    float3 GatherNormal(NormalDeriveEntry entry, uint offsets_base, uint items_base, uint bucket) const {
        device const uint *adjacency = Adjacency();
        const uint begin = adjacency[offsets_base + bucket];
        const uint end = adjacency[offsets_base + bucket + 1u];
        float3 n = float3(0);
        for (uint i = begin; i < end; ++i) n += FanContribution(entry, adjacency[items_base + i]);
        return NormalizeOrZero(n);
    }

    // Face `f`'s normalized vector-area normal, the summed crosses of its fan triangles.
    // A non-planar face shades uniformly under it, and the gather phase reads it back.
    float3 FaceNormal(NormalDeriveEntry entry, uint f) const {
        const uint2 range = FaceTriangleRange(entry, f);
        float3 n = float3(0);
        for (uint t = range.x; t < range.y; ++t) n += TriangleCross(entry, t);
        return NormalizeOrZero(n);
    }
};

kernel void VertexNormalDeriveKernel(
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant NormalDerivePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DeriveContext ctx{scene, pc};
    const uint2 tile = uint2(scene.TileMap(pc.TileMapSlot)[pc.FirstTile + group_id]);
    const NormalDeriveEntry entry = BindlessBuffer(NormalDeriveEntry, bindless.Buffer, pc.EntriesSlot)[tile.x];
    const uint i = tile.y * 256u + local_id;
    if (pc.Phase == 0u) {
        if (i < entry.FaceCount) {
            ctx.FaceNormals()[entry.FaceNormalOffset + i] = packed_float3(ctx.FaceNormal(entry, i));
        }
    } else if (i < entry.VertexCount) {
        ctx.VertexNormals()[entry.VertexNormalOffset + i] =
            packed_float3(ctx.GatherNormal(entry, entry.VertexAdjacencyOffset, entry.VertexAdjacencyOffset + entry.VertexCount + 1u, i));
    } else if (i < entry.VertexCount + entry.SeamCount) {
        const uint s = i - entry.VertexCount;
        ctx.SeamNormals()[entry.SeamNormalOffset + s] =
            packed_float3(ctx.GatherNormal(entry, entry.SeamFanOffset, entry.SeamFanOffset + entry.SeamCount + 1u, s));
    }
}

#endif
