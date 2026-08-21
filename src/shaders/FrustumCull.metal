#ifndef FRUSTUMCULL_MSL
#define FRUSTUMCULL_MSL

// Culls the remaining classic line and point draws in place. Phase 0 clears indirect counts;
// phase 1 appends visible instances to each command's compact remap range.
#include "Bindless.metal"
#include "AABB.metal"
#include "CullEntry.metal"
#include "CullFlag.metal"
#include "FrustumCullPushConstants.metal"
#include "Frustum.metal"

// Field for field, MTLDrawIndexedPrimitivesIndirectArguments.
struct IndirectCommand {
    uint IndexCount;
    uint InstanceCount;
    uint FirstIndex;
    int VertexOffset;
    uint FirstInstance;
};

kernel void FrustumCullKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant FrustumCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    device IndirectCommand *commands = BindlessBufferMutable(IndirectCommand, bindless.Buffer, pc.CommandsSlot);

    if (pc.Phase == 0u) {
        if (i < pc.CommandCount) commands[i].InstanceCount = 0u;
        return;
    }
    if (i >= pc.EntryCount) return;

    const CullEntry entry = BindlessBuffer(CullEntry, bindless.Buffer, pc.CullEntrySlot)[i];
    const uint cmd = entry.Cmd & ~CullFlag_KeepOrder;
    const DrawData draw = scene.Draws(pc.DrawDataSlot)[i];
    // Deforming geometry always draws, at its original remap slot for deterministic instance order.
    // These predicates are uniform across a command's entries (one mesh's DrawData and bounds).
    // Fixed slots therefore never collide with compacted ones.
    const bool fixed_slot = (entry.Cmd & CullFlag_KeepOrder) != 0u || draw.ArmatureDeformOffset != INVALID_OFFSET || draw.MorphDeformOffset != INVALID_OFFSET;
    const float4x4 view_proj = scene.ViewProj();
    bool empty = fixed_slot;
    float3 center = float3(0), ax = float3(0), ay = float3(0), az = float3(0);
    if (!fixed_slot) {
        const AABB aabb = BindlessBuffer(AABB, bindless.Buffer, pc.BoundsSlot)[draw.FirstInstance];
        const float3 aabb_min = float3(aabb.Min), aabb_max = float3(aabb.Max);
        empty = aabb_min.x > aabb_max.x; // Extras and empty meshes: always drawn.
        if (!empty) {
            const Transform t = scene.Models(pc.ModelSlot)[draw.FirstInstance];
            const float3 half_local = (aabb_max - aabb_min) * 0.5f;
            const float3 scale = float3(t.S);
            const float4 rotation = float4(t.R);
            center = trs_transform_point(t, (aabb_min + aabb_max) * 0.5f);
            ax = quat_rotate(rotation, float3(scale.x * half_local.x, 0, 0));
            ay = quat_rotate(rotation, float3(0, scale.y * half_local.y, 0));
            az = quat_rotate(rotation, float3(0, 0, scale.z * half_local.z));
        }
    }

    device uint *visible_indices = BindlessBufferMutable(uint, bindless.Buffer, pc.VisibleIndexSlot);

    if (fixed_slot || empty || in_frustum(view_proj, center, ax, ay, az)) {
        device atomic_uint *instance_count = reinterpret_cast<device atomic_uint *>(&commands[cmd].InstanceCount);
        const uint k = atomic_fetch_add_explicit(instance_count, 1u, memory_order_relaxed);
        visible_indices[fixed_slot || empty ? i : entry.Base + k] = i;
    }
}

#endif
