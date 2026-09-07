#include "VisibilityCoverage.metal"

fragment uint MeshletVisibilityOpaqueFragment(uint primitive_id [[primitive_id]]) {
    return primitive_id;
}

fragment uint MeshletVisibilityPrimitiveFragment(
    float4 position [[position]],
    uint primitive_id [[primitive_id]],
    bool front_facing [[front_facing]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &draw_pc [[buffer(BufferIndex_PushConstants)]]
) {
    // Raster-time coverage decodes IDs against the current draw's list.
    const VisibilityShadingPushConstants pc{
        .PrimitiveSlot = draw_pc.PrimitiveSlot,
        .InstanceSlot = draw_pc.InstanceSlot,
        .InstanceMapSlot = draw_pc.InstanceMapSlot,
        .MeshletSlot = draw_pc.MeshletSlot,
        .MeshletTriangleSlot = draw_pc.MeshletTriangleSlot,
        .MeshletLocalTriangleSlot = draw_pc.MeshletLocalTriangleSlot,
        .MeshletVertexSlot = draw_pc.MeshletVertexSlot,
        .VisibleMeshletSlot = draw_pc.VisibleMeshletSlot,
    };
    const Scene scene{bindless, view, theme, workspace};
    CoveredMeshletObject(scene, pc, primitive_id, position.xy, front_facing, draw_pc.VisibilityTransmission != 0u);
    return primitive_id;
}
