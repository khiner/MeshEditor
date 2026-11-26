#pragma once
// The surface points at which materials' programs compute their inputs.
#include "OpCall.metal"
#include "PreviewSurface.h"
#include "RayPatternHits.h"
#include "VisibilityDecode.metal"
#include "gpu/MaterialPattern.h"
#include "gpu/OpPoint.h"

// The material record index of a visible surface: its face's on a finest-level triangle, and its primitive's otherwise.
inline uint SurfaceMaterialIndex(const thread Scene &scene, const thread ResolvedVisibility &resolved) {
    if (MeshletCoarse(resolved.Meshlet) || MeshletPrimitiveTopology(resolved.Meshlet) != uint(MeshPrimitiveTopology::Triangle)) return MeshletPrimitiveMaterialIndex(scene, resolved.Primitive);
    return MeshletFaceMaterialIndex(scene, resolved.Draw, scene.TriangleFace(resolved.Draw, resolved.Triangle));
}

// The attribute table of a pattern that reads named attributes on the mesh record, which holds no attribute for a record the pattern's directory lacks, or null for a pattern that reads none.
inline device const OpAttributeRef *PatternAttributes(device const BindlessSet &b, MaterialPattern pattern, uint mesh) {
    const SlotOffset attributes = pattern.Program.Attributes;
    if (attributes.Slot == InvalidSlot) return nullptr;
    device const uint *directory = OpWords(b, attributes);
    const uint table = mesh < directory[0] && directory[2u + mesh] != InvalidOffset ? directory[2u + mesh] : directory[1];
    return reinterpret_cast<device const OpAttributeRef *>(OpWords(b, SlotOffset{attributes.Slot, table}));
}
// Starts a point of the pattern at the view's timeline frame and time, whose texture coordinates and neighbors its surface fills into `texcoords` and `neighbors`.
inline void PatternPoint(thread OpPoint &p, device const BindlessSet &b, constant SceneViewUBO &view, MaterialPattern pattern, thread OpPoint *neighbors, thread float2 *texcoords) {
    OpStart(p, b, pattern.Program, view.PatternFrame, view.PatternTime);
    p.SurfaceTexcoords = texcoords;
    p.Neighbors = neighbors;
}
// Calls the pattern's program at the point, and writes its output words in order to `words`.
inline void CallPattern(OpPrograms programs, device const BindlessSet &b, MaterialPattern pattern, const thread OpPoint &p, device uint *words) {
    uint out[OpMaxOutputWords];
    OpRun(programs, b, pattern.Program, p, out);
    for (uint c = 0u; c < pattern.Words; ++c) words[c] = out[c];
}
// Sets a neighbor's texture coordinates in the sets `sets` holds a bit for, one derivative along `axis` from the point's in `texcoords`.
inline void NeighborTexcoords(thread OpPoint &neighbor, const thread float2 *texcoords, uint axis, uint sets, uint needs) {
    for (uint s = 0u; s < 4u && (needs & OpNeedsTexcoords); ++s) {
        if (!((sets >> s) & 1u)) continue;
        const float2 uv = texcoords[s] + texcoords[(axis + 1u) * 4u + s];
        neighbor.Texcoords[s] = float2(uv.x, 1.0f - uv.y);
    }
}

// Sets the corners, their vertices and the face of a point at weights over a triangle's corners, at which the point's attribute table reads named attributes.
inline void SurfaceElements(thread OpPoint &p, device const BindlessSet &b, device const OpAttributeRef *attributes, uint3 corners, uint3 vertices, uint face, float3 weights) {
    p.Buffers = &b.Buffer[0];
    p.Attributes = attributes;
    for (uint c = 0u; c < 3u; ++c) {
        p.Elements[c] = corners[c];
        p.Vertices[c] = vertices[c];
        p.Weights[c] = weights[c];
    }
    p.Vertex = vertices[0];
    p.Face = face;
}

// The point fields a program reads at perspective weights over a triangle's corners, in its object's space.
struct SurfaceCorners {
    float3 Position[3], Normal[3], Tangent[3];
    float4 Color[3];
};
inline void SurfaceFields(const thread SurfaceCorners &corners, uint needs, float3 weights, thread OpPoint &p) {
    if (needs & OpNeedsPosition) p.Position = PerspectiveValue(weights, corners.Position[0], corners.Position[1], corners.Position[2]);
    if (needs & OpNeedsNormal) p.Normal = PerspectiveValue(weights, corners.Normal[0], corners.Normal[1], corners.Normal[2]);
    if (needs & OpNeedsTangent) p.Tangent = PerspectiveValue(weights, corners.Tangent[0], corners.Tangent[1], corners.Tangent[2]);
    if (needs & OpNeedsColor) p.Color = PerspectiveValue(weights, corners.Color[0], corners.Color[1], corners.Color[2]);
}

// Fills the fields and texture coordinates `needs` asks for at perspective weights over a triangle's corners, and the neighbors with those at the next pixel along x and y.
// `uvs` holds each corner's coordinates in each texture coordinate set, and `sets` a bit per set the triangle's mesh has.
// `texcoords` takes each set's coordinates with V down, then their derivatives along x, then along y, and a set the mesh lacks reads zero.
// The neighbors share the point's elements, frame, time and image table.
inline void WeightedSurfacePoint(const thread SurfaceCorners &corners, const thread float2 (&uvs)[4][3], uint sets, PerspectiveWeights weights, uint needs, thread OpPoint &p, thread OpPoint *neighbors, thread float2 *texcoords) {
    SurfaceFields(corners, needs, weights.Value, p);
    for (uint s = 0u; s < 4u && (needs & OpNeedsTexcoords); ++s) {
        texcoords[s] = texcoords[4u + s] = texcoords[8u + s] = p.Texcoords[s] = float2(0.0f);
        if (!((sets >> s) & 1u)) continue;
        DecodeUv(texcoords[s], texcoords[4u + s], texcoords[8u + s], weights, uvs[s][0], uvs[s][1], uvs[s][2]);
        p.Texcoords[s] = float2(texcoords[s].x, 1.0f - texcoords[s].y);
    }
    if (!(needs & OpNeedsNeighbors)) return;
    for (uint axis = 0u; axis < 2u; ++axis) {
        thread OpPoint &neighbor = neighbors[axis];
        neighbor = p;
        SurfaceFields(corners, needs, weights.Value + (axis == 0u ? weights.Dx : weights.Dy), neighbor);
        NeighborTexcoords(neighbor, texcoords, axis, sets, needs);
    }
}

// Fills the point at a pixel of a visible triangle with the fields `needs` asks for, and its neighbors with those at the next pixel along x and y.
// `texcoords` takes each set's coordinates with V down, then their derivatives along x, then along y, which the lit pass's decode of the pixel gives.
// With an attribute table, the point also holds the triangle's corners at the pixel's weights.
inline void PixelSurfacePoint(const thread Scene &scene, const thread ResolvedVisibility &resolved, float2 pixel, uint needs, device const OpAttributeRef *attributes, thread OpPoint &p, thread OpPoint *neighbors, thread float2 *texcoords) {
    if (!(needs & (OpNeedsPosition | OpNeedsNormal | OpNeedsTangent | OpNeedsColor | OpNeedsTexcoords)) && !attributes) return;
    const DrawData draw = resolved.Draw;
    const MeshletRecord meshlet = resolved.Meshlet;
    const bool coarse = MeshletCoarse(meshlet);
    const MeshletTriangleCorners triangle = ResolveMeshletCorners(scene, draw, resolved.Payloads, meshlet, resolved.Triangle, resolved.LocalTriangle);
    const Transform world = MeshletWorld(scene, draw);
    uint sets = 0u;
    for (uint s = 0u; s < 4u; ++s) sets |= uint(draw.CornerUvs[s].Slot != InvalidSlot) << s;
    float4 clip[3];
    float2 uvs[4][3];
    SurfaceCorners corners;
    for (uint c = 0u; c < 3u; ++c) {
        const uint corner = triangle.CornerIds[c], vertex_id = triangle.VertexIds[c];
        clip[c] = MeshletPosition(scene, draw, world, vertex_id);
        for (uint s = 0u; s < 4u && (needs & OpNeedsTexcoords); ++s) uvs[s][c] = (sets >> s) & 1u ? scene.CornerUv(draw.CornerUvs[s], corner) : float2(0);
        if (needs & OpNeedsPosition) corners.Position[c] = scene.GetLocalPosition(draw, vertex_id);
        if (needs & OpNeedsNormal) corners.Normal[c] = CornerNormal(scene, draw, corner, vertex_id, coarse ? InvalidOffset : scene.CornerFace(draw, corner), coarse, triangle.CoarseNormal);
        if (needs & OpNeedsTangent) corners.Tangent[c] = draw.CornerTangent.Slot != InvalidSlot ? scene.CornerTangent(draw.CornerTangent, corner).xyz : float3(0.0f);
        if (needs & OpNeedsColor) corners.Color[c] = draw.CornerColor.Slot != InvalidSlot ? scene.CornerColor(draw.CornerColor, corner) : float4(1.0f);
    }
    const PerspectiveWeights weights = TriangleWeights(pixel, clip[0], clip[1], clip[2], float2(scene.View.ViewportSize));
    if (attributes) SurfaceElements(p, scene.B, attributes, triangle.CornerIds, triangle.VertexIds, scene.CornerFace(draw, triangle.CornerIds[0]), weights.Value);
    WeightedSurfacePoint(corners, uvs, sets, weights, needs, p, neighbors, texcoords);
}

// Fills the point at a ray hit with the fields `needs` asks for, in its object's space, at the hit's instance, primitive and barycentric coordinates.
// Its neighbors lie one ray footprint along the surface's derivatives, which the hit's material filters its textures over.
// `texcoords` takes each set's coordinates with V down, then their derivatives toward each neighbor.
// With an attribute table, the point also holds the triangle's corners at the hit's weights.
inline void HitSurfacePoint(const thread Scene &scene, const thread RayGeometry &geometry, uint4 hit, float ray_footprint, uint needs, device const OpAttributeRef *attributes, thread OpPoint &p, thread OpPoint *neighbors, thread float2 *texcoords) {
    if (!(needs & (OpNeedsPosition | OpNeedsNormal | OpNeedsTangent | OpNeedsColor | OpNeedsTexcoords)) && !attributes) return;
    const RayTriangle triangle = ResolveRayTriangle(scene, geometry, hit.x, hit.y);
    const DrawData draw = triangle.Draw;
    const float2 barycentric = as_type<float2>(hit.zw);
    SurfaceCorners corners;
    uint3 corner_ids, vertex_ids;
    for (uint c = 0u; c < 3u; ++c) {
        const uint corner = TriangleCornerHandle(scene.B, draw.TriangleSlot, draw.Connectivity, triangle.Triangle, c), vertex_id = scene.CornerVertex(draw, corner);
        corner_ids[c] = corner;
        vertex_ids[c] = vertex_id;
        if (needs & OpNeedsPosition) corners.Position[c] = scene.GetLocalPosition(draw, vertex_id);
        if (needs & OpNeedsNormal) corners.Normal[c] = CornerNormal(scene, draw, corner, vertex_id, scene.CornerFace(draw, corner), false, float3(0.0f));
        if (needs & OpNeedsTangent) corners.Tangent[c] = draw.CornerTangent.Slot != InvalidSlot ? scene.CornerTangent(draw.CornerTangent, corner).xyz : float3(0.0f);
        if (needs & OpNeedsColor) corners.Color[c] = draw.CornerColor.Slot != InvalidSlot ? scene.CornerColor(draw.CornerColor, corner) : float4(1.0f);
    }
    SurfaceFields(corners, needs, float3(1.0f - barycentric.x - barycentric.y, barycentric), p);
    if (attributes) SurfaceElements(p, scene.B, attributes, corner_ids, vertex_ids, scene.CornerFace(draw, corner_ids[0]), float3(1.0f - barycentric.x - barycentric.y, barycentric));
    if (!(needs & (OpNeedsTexcoords | OpNeedsNeighbors))) return;
    // Coordinates and their derivatives are the decoded surface's, which the material's textures sample at.
    const DecodedSurface surface = DecodeRaySurface(scene, geometry, hit.x, hit.y, barycentric);
    const float footprint = HitFootprint(surface, ray_footprint);
    const float2 sets[4]{surface.V.TexCoord0, surface.V.TexCoord1, surface.V.TexCoord2, surface.V.TexCoord3};
    for (uint s = 0u; s < 4u && (needs & OpNeedsTexcoords); ++s) {
        texcoords[s] = sets[s];
        texcoords[4u + s] = surface.UvDx[s] * footprint;
        texcoords[8u + s] = surface.UvDy[s] * footprint;
        p.Texcoords[s] = float2(sets[s].x, 1.0f - sets[s].y);
    }
    if (!(needs & OpNeedsNeighbors)) return;
    for (uint axis = 0u; axis < 2u; ++axis) {
        thread OpPoint &neighbor = neighbors[axis];
        neighbor = p;
        if (needs & OpNeedsPosition) neighbor.Position = p.Position + (corners.Position[axis + 1u] - corners.Position[0]) * footprint;
        NeighborTexcoords(neighbor, texcoords, axis, 0xFu, needs);
    }
}

// Fills the point at a pixel of the preview sphere with the fields `needs` asks for, and its neighbors with those at the next pixel along x and y.
// The sphere's world space is its object's space, its first texture coordinate set holds its coordinates, and its points hold no named attribute.
inline void PreviewSurfacePoint(device const BindlessSet &b, float2 pixel, uint triangle, device const PreviewCorner *corners, constant SceneViewUBO &view, uint needs, device const OpAttributeRef *attributes, thread OpPoint &p, thread OpPoint *neighbors, thread float2 *texcoords) {
    const PreviewTriangle resolved = ResolvePreviewTriangle(pixel, triangle, corners, view);
    const thread PreviewCorner (&c)[3] = resolved.Corners;
    SurfaceCorners fields;
    float2 uvs[4][3];
    for (uint i = 0u; i < 3u; ++i) {
        fields.Position[i] = float3(c[i].Position);
        fields.Normal[i] = float3(c[i].Normal);
        fields.Tangent[i] = float3(c[i].Tangent);
        fields.Color[i] = float4(1.0f);
        uvs[0][i] = float2(c[i].Uv);
    }
    if (attributes) SurfaceElements(p, b, attributes, uint3(0u), uint3(0u), 0u, resolved.Weights.Value);
    WeightedSurfacePoint(fields, uvs, 1u, resolved.Weights, needs, p, neighbors, texcoords);
}
