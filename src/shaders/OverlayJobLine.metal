#ifndef OVERLAYJOBLINE_MSL
#define OVERLAYJOBLINE_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "ObjectExtrasTransform.metal"
#include "Varyings.metal"
#include "AABB.metal"
#include "BoxWire.metal"
#include "ExtrasLineKind.metal"
#include "InstanceRecord.metal"
#include "OverlayDispatch.metal"
#include "OverlayJob.metal"
#include "OverlayJobDrawPushConstants.metal"
#include "OverlayJobKind.metal"

// Each job emits a bounded line group, two dedicated vertices per line. Extras generate their
// geometry procedurally; bounds and tetrahedral jobs read persistent scene buffers.
constant uint OverlayJobGroupLines = OverlayDispatch_LineGroupLines;
constant uint LightRangeSegments = OverlayDispatch_LightRangeSegments;
constant uint SpotConeSegments = OverlayDispatch_SpotConeSegments;
constant float SpotConeDepth = 2.0f;
constant uint ColliderCircleSegments = OverlayDispatch_ColliderCircleSegments;
constant uint ColliderArcSegments = ColliderCircleSegments / 2u;
// A hemisphere reads as its base circle plus four quarter arcs rising to the pole.
constant uint ColliderCapLines = ColliderCircleSegments + 4u * ColliderArcSegments;
using OverlayJobLineOutput = metal::mesh<ObjectLineVaryings, void, OverlayJobGroupLines * 2u, OverlayJobGroupLines, metal::topology::line>;

inline bool IsColliderKind(uint kind) { return kind >= ExtrasLineKind_ColliderBox; }

// A local-space line vertex and the class that decides how it reaches world space.
struct ExtrasLineVertex {
    float3 Position;
    uint Class;
};

inline float3 CirclePoint(uint segment, uint end, uint segments, float radius, float z) {
    const float angle = 2.0f * Pi * float(segment + end) / float(segments);
    return float3(radius * cos(angle), radius * sin(angle), z);
}

// Corner `corner` of a diamond of `radius` spanned by two axes, in the order the wireframe walks them.
inline float3 DiamondPoint(uint corner, float radius, float3 axis1, float3 axis2) {
    const float3 axes[4] = {axis1, axis2, -axis1, -axis2};
    return axes[corner % 4u] * radius;
}

// The halo every light carries: a diamond, two dashed rings, and the drop line to its ground diamond.
inline ExtrasLineVertex LightHaloVertex(uint line, uint end) {
    if (line < 4u) return {DiamondPoint(line + end, 2.7f, float3(1, 0, 0), float3(0, 1, 0)), VCLASS_SCREENSPACE};
    if (line < 12u) return {CirclePoint((line - 4u) * 2u, end, 16u, 9.0f, 0.0f), VCLASS_SCREENSPACE};
    if (line < 22u) return {CirclePoint((line - 12u) * 2u, end, 20u, 9.0f * 1.33f, 0.0f), VCLASS_SCREENSPACE};
    if (line < 23u) return {end == 0u ? float3(0, 1, 0) : float3(0, 0, 0), VCLASS_GROUNDPOINT};
    return {DiamondPoint(line - 23u + end, 3.0f, float3(1, 0, 0), float3(0, 0, 1)), VCLASS_GROUNDPOINT};
}

// One endpoint of a unit hemisphere's wireframe, radius 1 and opening in +Y.
inline float3 ColliderCapPoint(uint line, uint end) {
    if (line < ColliderCircleSegments) {
        const float angle = 2.0f * Pi * float(line + end) / float(ColliderCircleSegments);
        return float3(cos(angle), 0.0f, sin(angle));
    }
    const uint arc = (line - ColliderCircleSegments) / ColliderArcSegments;
    const uint segment = (line - ColliderCircleSegments) % ColliderArcSegments;
    const float angle = 0.5f * M_PI_F * float(segment + end) / float(ColliderArcSegments);
    const float3 axis = arc == 0u ? float3(1, 0, 0) : arc == 1u ? float3(-1, 0, 0) :
        arc == 2u                                               ? float3(0, 0, 1) :
                                                                  float3(0, 0, -1);
    return axis * cos(angle) + float3(0, 1, 0) * sin(angle);
}

// One endpoint of a ring of `radius` at `height`, in the XZ plane.
inline float3 ColliderRingPoint(uint segment, uint end, float radius, float height) {
    const float angle = 2.0f * Pi * float(segment + end) / float(ColliderCircleSegments);
    return float3(cos(angle) * radius, height, sin(angle) * radius);
}

// One endpoint of a collision shape's `line`, in the shape's own space.
inline float3 ColliderWirePoint(OverlayJob job, uint line, uint end) {
    const float4 params = job.Params;
    if (job.ExtrasKind == ExtrasLineKind_ColliderBox) {
        const uint corner = BoxEdgeCorners[line * 2u + end];
        const float3 half_size = params.xyz * 0.5f;
        return mix(-half_size, half_size, float3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u)));
    }
    if (job.ExtrasKind == ExtrasLineKind_ColliderSphere) {
        // Three great circles, one per axis plane.
        const uint circle = line / ColliderCircleSegments;
        const float angle = 2.0f * Pi * float(line % ColliderCircleSegments + end) / float(ColliderCircleSegments);
        const float c = cos(angle) * params.x, s = sin(angle) * params.x;
        return circle == 0u ? float3(c, s, 0) : circle == 1u ? float3(c, 0, s) : float3(0, c, s);
    }

    // Cylinders and capsules share a top and bottom profile joined by four side lines.
    const float radius_top = params.x, radius_bottom = params.y, half_height = params.z * 0.5f;
    const uint profile_lines = job.ExtrasKind == ExtrasLineKind_ColliderCapsule ? ColliderCapLines : ColliderCircleSegments;
    if (line >= profile_lines * 2u) {
        const uint side = line - profile_lines * 2u;
        const float angle = 0.5f * M_PI_F * float(side);
        const float c = cos(angle), s = sin(angle);
        return end == 0u ? float3(radius_top * c, half_height, radius_top * s) :
                           float3(radius_bottom * c, -half_height, radius_bottom * s);
    }
    const bool top = line < profile_lines;
    const uint profile_line = top ? line : line - profile_lines;
    const float radius = top ? radius_top : radius_bottom;
    const float height = top ? half_height : -half_height;
    if (job.ExtrasKind == ExtrasLineKind_ColliderCylinder) return ColliderRingPoint(profile_line, end, radius, height);
    // The bottom cap mirrors the top one through the XZ plane.
    const float3 point = ColliderCapPoint(profile_line, end) * radius;
    return float3(point.x, height + (top ? point.y : -point.y), point.z);
}

// One endpoint of the extras `line`, in the object's local space.
inline ExtrasLineVertex ExtrasLineVertexAt(OverlayJob job, uint line, uint end) {
    if (IsColliderKind(job.ExtrasKind)) return {float3(job.LocalOffset) + ColliderWirePoint(job, line, end), VCLASS_NONE};
    const float4 params = job.Params;
    if (job.ExtrasKind == ExtrasLineKind_Empty) {
        // Three axis line segments: +X, +Y, -Z from the origin.
        if (end == 0u) return {float3(0), VCLASS_NONE};
        return {line == 0u ? float3(1, 0, 0) : line == 1u ? float3(0, 1, 0) : float3(0, 0, -1), VCLASS_NONE};
    }
    if (job.ExtrasKind == ExtrasLineKind_Camera) {
        const float half_w = params.x, half_h = params.y, depth = params.z;
        const bool look_through = params.w != 0.0f;
        const float tria_size = 0.7f * 0.5f, tria_y = half_h + 0.1f * 0.5f;
        const float3 frame[4] = {
            float3(-half_w, -half_h, -depth), float3(half_w, -half_h, -depth),
            float3(half_w, half_h, -depth), float3(-half_w, half_h, -depth)
        };
        // The frame loop, then a wire from each corner to its far endpoint, then the up triangle.
        // Looking through the camera collapses everything but the frame, leaving the lens rect alone.
        if (line < 4u) return {frame[(line + end) % 4u], VCLASS_NONE};
        if (line < 8u) {
            const uint corner = line - 4u;
            if (end == 0u) return {frame[corner], VCLASS_NONE};
            return {look_through ? frame[corner] : float3(0), VCLASS_NONE};
        }
        const float3 tria[3] = {
            float3(-tria_size, tria_y, -depth), float3(tria_size, tria_y, -depth), float3(0, tria_y + tria_size, -depth)
        };
        return {look_through ? frame[0] : tria[(line - 8u + end) % 3u], VCLASS_NONE};
    }

    // Lights lay out their type-specific geometry first, then the shared halo.
    // A point or spot light rings its range; a directional light has no range to show.
    const float range = params.x, outer_radius = params.y, inner_radius = params.z;
    const bool rings_range = job.ExtrasKind != ExtrasLineKind_LightDirectional && range > 0.0f;
    const uint range_lines = rings_range ? LightRangeSegments : 0u;
    if (line < range_lines) return {CirclePoint(line, end, LightRangeSegments, range, 0.0f), VCLASS_BILLBOARD};

    uint local = line - range_lines;
    if (job.ExtrasKind == ExtrasLineKind_LightDirectional) {
        // Eight rays, each a pair of dashes along its direction.
        const uint ray_count = 8u;
        if (local < ray_count * 2u) {
            const uint ray = local / 2u;
            const bool outer = (local % 2u) == 1u;
            const float angle = 2.0f * Pi * float(ray) / float(ray_count);
            const float2 dir = float2(cos(angle), sin(angle));
            const float distance = outer ? (end == 0u ? 18.0f : 20.0f) : (end == 0u ? 14.0f : 16.0f);
            return {float3(dir * distance, 0.0f), VCLASS_SCREENSPACE};
        }
        local -= ray_count * 2u;
    } else if (job.ExtrasKind == ExtrasLineKind_LightSpot) {
        if (local < SpotConeSegments) return {CirclePoint(local, end, SpotConeSegments, outer_radius, -SpotConeDepth), VCLASS_NONE};
        local -= SpotConeSegments;
        const uint inner_lines = inner_radius > 0.0f ? SpotConeSegments : 0u;
        if (local < inner_lines) return {CirclePoint(local, end, SpotConeSegments, inner_radius, -SpotConeDepth), VCLASS_NONE};
        local -= inner_lines;
        if (local < SpotConeSegments) {
            // Spokes from the apex, whose base ends resolve to a silhouette in the transform.
            if (end == 0u) return {float3(0), VCLASS_NONE};
            const float angle = 2.0f * Pi * float(local) / float(SpotConeSegments);
            return {float3(outer_radius * cos(angle), outer_radius * sin(angle), -SpotConeDepth), VCLASS_SPOT_CONE};
        }
        local -= SpotConeSegments;
    }
    return LightHaloVertex(local, end);
}

// Clip position of one extras line endpoint, and the class that placed it.
// Collision shapes hug their object's surface, so their lines push in front of faces.
inline ExtrasLineVertex ExtrasLineClipVertex(
    const thread Scene &scene, constant OverlayJobDrawPushConstants &pc,
    OverlayJob job, uint line, uint end, thread float4 &clip
) {
    auto vertex_at = ExtrasLineVertexAt(job, line, end);
    const Transform world = scene.Models(pc.ModelSlot)[job.InstanceIndex];
    vertex_at.Position = ObjectExtrasWorldPos(scene, world, vertex_at.Position, vertex_at.Class);
    clip = scene.ViewProj() * float4(vertex_at.Position, 1.0f);
    if (IsColliderKind(job.ExtrasKind)) clip.z -= NdcOffsetFactor(scene);
    return vertex_at;
}

inline OverlayJob ResolveOverlayJob(
    device const BindlessSet &bindless, constant OverlayJobDrawPushConstants &pc, uint group
) {
    const uint job = BindlessBuffer(uint, bindless.Buffer, pc.VisibleSlot)[group];
    return BindlessBuffer(OverlayJob, bindless.Buffer, pc.JobsSlot)[job];
}

inline float4 OverlayJobClipVertex(
    const thread Scene &scene, device const BindlessSet &bindless,
    constant OverlayJobDrawPushConstants &pc, OverlayJob job,
    uint element, uint end, thread uint &vertex_class
) {
    vertex_class = VCLASS_NONE;
    if (job.Kind == OverlayJobKind_Extras) {
        float4 clip;
        vertex_class = ExtrasLineClipVertex(scene, pc, job, element, end, clip).Class;
        return clip;
    }
    const Transform world = scene.Models(pc.ModelSlot)[job.InstanceIndex];
    float3 local;
    if (job.Kind == OverlayJobKind_Bounds) {
        const AABB bounds = BindlessBuffer(AABB, bindless.Buffer, pc.BoundsSlot)[job.InstanceIndex];
        if (any(float3(bounds.Min) > float3(bounds.Max))) return float4(2, 2, 2, 1);
        const uint corner = BoxEdgeCorners[element * 2u + end];
        local = mix(
            float3(bounds.Min), float3(bounds.Max),
            float3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u))
        );
    } else {
        const uint point = BindlessBuffer(uint, bindless.Buffer, pc.TetEdgeIndexSlot)[job.IndexOffset + element * 2u + end];
        local = float3(BindlessBuffer(packed_float3, bindless.Buffer, pc.TetPositionSlot)[job.SourceOffset + point]);
    }
    float4 clip = scene.ViewProj() * float4(trs_transform_point(world, local), 1.0f);
    if (job.Kind == OverlayJobKind_TetWire) clip.z -= NdcOffsetFactor(scene);
    return clip;
}

[[mesh]] void OverlayJobLineMesh(
    OverlayJobLineOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayJobDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const OverlayJob job = ResolveOverlayJob(bindless, pc, threadgroup_position.x);
    const uint line_count = job.ElementCount;
    output.set_primitive_count(line_count);
    if (thread_index >= line_count * 2u) return;

    uint vertex_class;
    const float4 clip = OverlayJobClipVertex(
        scene, bindless, pc, job, job.FirstElement + thread_index / 2u, thread_index & 1u, vertex_class
    );

    const uint instance_state = uint(scene.InstanceStates(pc.StateSlot)[job.InstanceIndex]);
    float4 color = job.Kind == OverlayJobKind_TetWire ? WireBaseColor(scene) :
        scene.ObjectSelectionColor(instance_state, WireBaseColor(scene));
    if (job.Kind == OverlayJobKind_Bounds) {
        color = float4(
            (instance_state & STATE_ACTIVE) != 0u ? float3(scene.Theme.Colors.ObjectActive) :
                                                   float3(scene.Theme.Colors.ObjectSelected),
            1.0f
        );
    }
    // The ground line and diamond keep a fixed theme colour, unaffected by selection state.
    if (vertex_class == VCLASS_GROUNDPOINT) color = float4(scene.Theme.Colors.Light);

    const LineVaryings line = MakeLineVertex(clip, color, float2(scene.View.ViewportSize));
    ObjectLineVaryings out{line.Position, line.Color, line.EdgeStart, line.EdgePos, BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[job.InstanceIndex].ObjectId};
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

#endif
