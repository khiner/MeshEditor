// Fast math that keeps Inf and NaN semantics, as the lit pass that reads the computed inputs decodes its surfaces.
#pragma METAL fp math_mode(relaxed)
#include "OpCall.metal"
#include "SurfacePoint.metal"

// Lists each pixel whose surface's material computes inputs, one atomic reservation per SIMD group that lists any.
kernel void SurfacePatternClassify(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    SCENE_ARGS,
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    device uint *work [[buffer(BufferIndex_PatternWork)]]
) {
    const uint2 extent = uint2(visibility.get_width(), visibility.get_height());
    bool listed = false;
    if (all(pixel < extent)) {
        const ResolvedVisibility resolved = ResolveVisibilityId(visibility.read(pixel).r, bindless, view, theme, workspace, pc);
        const Scene scene{bindless, view, theme, workspace};
        listed = resolved.Valid && MeshletPrimitiveTopology(resolved.Meshlet) == uint(MeshPrimitiveTopology::Triangle) &&
            BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[SurfaceMaterialIndex(scene, resolved)].Program.Index != InvalidOffset;
    }
    const uint rank = simd_prefix_exclusive_sum(uint(listed));
    const uint count = simd_sum(uint(listed));
    uint first = 0u;
    if (simd_is_first() && count != 0u) first = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(work), count, memory_order_relaxed);
    first = simd_broadcast_first(first);
    if (listed) work[SurfacePatternPixels + first + rank] = pixel.y * extent.x + pixel.x;
}

// Writes the dispatch arguments of the listed pixels, in threadgroups of 64.
kernel void SurfacePatternArguments(device uint *work [[buffer(BufferIndex_PatternWork)]]) {
    work[1] = (work[0] + 63u) / 64u;
    work[2] = work[3] = 1u;
}

// Calls the program of each listed pixel's material at the pixel's surface point, and writes its output words in order to the pixel's pattern words.
kernel void SurfacePatternDriver(
    uint thread_id [[thread_position_in_grid]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    SCENE_ARGS,
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    OpPrograms programs [[buffer(BufferIndex_Programs)]],
    device const uint *work [[buffer(BufferIndex_PatternWork)]]
) {
    if (thread_id >= work[0]) return;
    const uint width = visibility.get_width(), at = work[SurfacePatternPixels + thread_id];
    const uint2 pixel(at % width, at / width);
    const ResolvedVisibility resolved = ResolveVisibilityId(visibility.read(pixel).r, bindless, view, theme, workspace, pc);
    if (!resolved.Valid || MeshletPrimitiveTopology(resolved.Meshlet) != uint(MeshPrimitiveTopology::Triangle)) return;
    const Scene scene{bindless, view, theme, workspace};
    const MaterialPattern pattern = BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[SurfaceMaterialIndex(scene, resolved)];
    if (pattern.Program.Index == InvalidOffset) return;
    OpPoint p, neighbors[2];
    float2 texcoords[12];
    PatternPoint(p, bindless, view, pattern, neighbors, texcoords);
    PixelSurfacePoint(scene, resolved, float2(pixel) + 0.5f, pattern.Program.Needs, PatternAttributes(bindless, pattern, resolved.Instance.Mesh), p, neighbors, texcoords);
    CallPattern(programs, bindless, pattern, p, BindlessBufferMutable(uint, bindless.Buffer, view.PatternWordSlot) + PatternPixelWord(view, pixel));
}

// Calls the preview's program at each pixel of the preview sphere, and writes its output words in order to the pixel's pattern words.
// The program's record holds its value block, image table and attribute directory, whose absent table its points read.
kernel void SurfacePatternPreviewDriver(
    uint2 pixel [[thread_position_in_grid]],
    texture2d<uint, access::read> triangles [[texture(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    device const PreviewCorner *corners [[buffer(BufferIndex_PreviewCorners)]],
    constant MaterialPattern &pattern [[buffer(BufferIndex_PushConstants)]],
    OpPrograms programs [[buffer(BufferIndex_Programs)]]
) {
    if (pixel.x >= triangles.get_width() || pixel.y >= triangles.get_height()) return;
    const uint triangle = triangles.read(pixel).r;
    if (triangle == VisibilityBackground) return;
    OpPoint p, neighbors[2];
    float2 texcoords[12];
    PatternPoint(p, bindless, view, pattern, neighbors, texcoords);
    PreviewSurfacePoint(bindless, float2(pixel) + 0.5f, triangle, corners, view, pattern.Program.Needs, PatternAttributes(bindless, pattern, InvalidOffset), p, neighbors, texcoords);
    CallPattern(programs, bindless, pattern, p, BindlessBufferMutable(uint, bindless.Buffer, view.PatternWordSlot) + PatternPixelWord(view, pixel));
}

// Writes the dispatch arguments of the ray samples a hit kernel listed, and clears the list for the next hit kernel.
kernel void SurfacePatternHitArguments(device uint *hits [[buffer(BufferIndex_PatternHits)]]) {
    const uint count = hits[0];
    hits[PatternHitsListed] = count;
    hits[1] = (count + 63u) / 64u;
    hits[2] = hits[3] = 1u;
    hits[0] = 0u;
}

// Calls the program of each listed ray sample's first surface's material at the surface, and writes its output words in order to the sample's ring words.
kernel void SurfacePatternHitDriver(
    uint listed [[thread_position_in_grid]],
    SCENE_ARGS,
    instance_acceleration_structure acceleration [[buffer(5)]],
    device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]],
    device const RayLightingSample *queue [[buffer(11)]],
    device const uint *hits [[buffer(BufferIndex_PatternHits)]],
    OpPrograms programs [[buffer(BufferIndex_PatternPrograms)]]
) {
    if (listed >= hits[PatternHitsListed]) return;
    const uint2 entry = PatternHitEntry(hits, listed);
    const RayLightingSample sample = queue[entry.x];
    const Scene scene{bindless, view, theme, workspace};
    const RayGeometry geometry{acceleration, instances, triangles};
    const uint4 hit = PatternHitIdentity(sample, entry.y);
    const MaterialPattern pattern = BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[ResolveRayTriangle(scene, geometry, hit.x, hit.y).Material(scene)];
    OpPoint p, neighbors[2];
    float2 texcoords[12];
    PatternPoint(p, bindless, view, pattern, neighbors, texcoords);
    const uint mesh = scene.InstanceRecords(view.InstanceRecordSlot)[instances[hit.x].InstanceSlot].Mesh;
    HitSurfacePoint(scene, geometry, hit, sample.OriginFootprint.w, pattern.Program.Needs, PatternAttributes(bindless, pattern, mesh), p, neighbors, texcoords);
    CallPattern(programs, bindless, pattern, p, BindlessBufferMutable(uint, bindless.Buffer, view.PatternWordSlot) + PatternRingWord(view, listed));
}

// Fills the point at a transparent fragment with the fields `needs` asks for, in its object's space, and its neighbors with those at the next pixel along x and y.
// `texcoords` takes each set's coordinates with V down, then their derivatives along x, then along y, which the fragment's own texture lookups filter over.
inline void FragmentSurfacePoint(const thread MeshletPatternVaryings &in, uint needs, thread OpPoint &p, thread OpPoint *neighbors, thread float2 *texcoords) {
    const float2 sets[4]{in.TexCoord0, in.TexCoord1, in.TexCoord2, in.TexCoord3};
    if (needs & OpNeedsPosition) p.Position = in.ObjectPosition;
    if (needs & OpNeedsNormal) p.Normal = in.ObjectNormal;
    if (needs & OpNeedsTangent) p.Tangent = float3(in.ObjectTangent);
    if (needs & OpNeedsColor) p.Color = in.VertexColor;
    for (uint s = 0u; s < 4u && (needs & OpNeedsTexcoords); ++s) {
        texcoords[s] = sets[s];
        texcoords[4u + s] = dfdx(sets[s]);
        texcoords[8u + s] = dfdy(sets[s]);
        p.Texcoords[s] = float2(sets[s].x, 1.0f - sets[s].y);
    }
    if (!(needs & OpNeedsNeighbors)) return;
    for (uint axis = 0u; axis < 2u; ++axis) {
        thread OpPoint &neighbor = neighbors[axis];
        neighbor = p;
        if (needs & OpNeedsPosition) neighbor.Position += axis == 0u ? dfdx(in.ObjectPosition) : dfdy(in.ObjectPosition);
        if (needs & OpNeedsNormal) neighbor.Normal += axis == 0u ? dfdx(in.ObjectNormal) : dfdy(in.ObjectNormal);
        if (needs & OpNeedsTangent) neighbor.Tangent += float3(axis == 0u ? dfdx(in.ObjectTangent) : dfdy(in.ObjectTangent));
        if (needs & OpNeedsColor) neighbor.Color += axis == 0u ? dfdx(in.VertexColor) : dfdy(in.VertexColor);
        NeighborTexcoords(neighbor, texcoords, axis, 0xFu, needs);
    }
}

// Calls the program of each transparent fragment of a material that computes inputs at the fragment, writes its output words to a list entry, and lists the entry under the fragment's pixel by its triangle's key.
// A program reading named attributes reads them at the fragment's triangle, which its key resolves, at the fragment's weights.
// A fragment past the list's capacity is left out.
[[early_fragment_tests]] fragment void SurfacePatternListFragment(
    MeshletPatternVaryings in [[stage_in]],
    float3 weights [[barycentric_coord]],
    SCENE_ARGS,
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    OpPrograms programs [[buffer(BufferIndex_PatternPrograms)]]
) {
    const MaterialPattern pattern = BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[in.MaterialIndex];
    OpPoint p, neighbors[2];
    float2 texcoords[12];
    PatternPoint(p, bindless, view, pattern, neighbors, texcoords);
    FragmentSurfacePoint(in, pattern.Program.Needs, p, neighbors, texcoords);
    if (pattern.Program.Attributes.Slot != InvalidSlot) {
        const ResolvedVisibility resolved = ResolveVisibilityId(in.Key, bindless, view, theme, workspace, pc);
        const Scene scene{bindless, view, theme, workspace};
        const MeshletTriangleCorners triangle = ResolveMeshletCorners(scene, resolved.Draw, resolved.Payloads, resolved.Meshlet, resolved.Triangle, resolved.LocalTriangle);
        SurfaceElements(p, bindless, PatternAttributes(bindless, pattern, resolved.Instance.Mesh), triangle.CornerIds, triangle.VertexIds, scene.CornerFace(resolved.Draw, triangle.CornerIds[0]), weights);
        for (uint axis = 0u; axis < 2u && (pattern.Program.Needs & OpNeedsNeighbors); ++axis) SurfaceElements(neighbors[axis], bindless, p.Attributes, triangle.CornerIds, triangle.VertexIds, p.Face, weights);
    }
    device uint *words = BindlessBufferMutable(uint, bindless.Buffer, view.PatternWordSlot);
    const uint entry = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(words), 1u, memory_order_relaxed);
    if (entry >= view.PatternFragmentCapacity) return;
    CallPattern(programs, bindless, pattern, p, words + view.PatternFragmentWords + entry * view.PatternStride);
    const uint2 pixel = uint2(in.Position.xy);
    words[view.PatternFragmentEntries + 2u * entry] = in.Key;
    words[view.PatternFragmentEntries + 2u * entry + 1u] = atomic_exchange_explicit(reinterpret_cast<device atomic_uint *>(words + view.PatternFragmentHeads + pixel.y * view.PatternWidth + pixel.x), entry, memory_order_relaxed);
}
