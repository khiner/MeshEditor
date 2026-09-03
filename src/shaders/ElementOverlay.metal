#ifndef ELEMENTOVERLAY_MSL
#define ELEMENTOVERLAY_MSL

#include "LineQuad.metal"
#include "EditSelection.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

inline float4 EditEdgeColor(const thread Scene &scene, uint state, bool direct_selection) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    if ((state & STATE_ACTIVE) != 0u) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if ((state & STATE_SELECTED) == 0u) return WireBaseColor(scene);
    return direct_selection ? float4(float3(colors.EdgeSelected), 1.0f) :
                              float4(float3(colors.EdgeSelectedIncidental), 1.0f);
}

inline float4 EditVertexColor(const thread Scene &scene, uint state) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    if ((state & STATE_ACTIVE) != 0u) return float4(float4(colors.ElementActive).rgb, 1.0f);
    if ((state & STATE_SELECTED) != 0u) return float4(float3(colors.VertexSelected), 1.0f);
    return float4(float3(colors.Vertex), 1.0f);
}

struct EditEdgeOverlay {
    float4 Clip0, Clip1;
    float4 Color0, Color1;
    bool Sharp;
};

inline EdgeQuadVaryings EditEdgeQuadCorner(
    const thread Scene &scene, float4 clip0, float4 clip1, float4 color, bool sharp, uint corner
) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float edge_width = scene.Theme.EdgeWidth;
    // EdgeWidth is a half-width; sharp marks add one band and every edge adds a 0.5px AA fringe.
    const float half_width = edge_width + (sharp ? max(edge_width, 1.0f) : 0.0f) + 0.5f;

    EdgeQuadVaryings out;
    out.Position = line_quad_position(scene, clip0, clip1, corner, half_width);
    if (sharp) out.Position.z -= 5e-7f * abs(out.Position.w);
    out.Color = color;
    out.OuterColor = sharp ? float4(float3(colors.EdgeSharp), 1.0f) : float4(0.0f);
    out.EdgeCoord = line_quad_side(corner) * half_width;
    return out;
}

inline EdgeQuadVaryings EditEdgeQuadCorner(
    const thread Scene &scene, const thread EditEdgeOverlay &edge, uint corner
) {
    const float4 color = line_quad_endpoint(corner) == 0u ? edge.Color0 : edge.Color1;
    return EditEdgeQuadCorner(scene, edge.Clip0, edge.Clip1, color, edge.Sharp, corner);
}

inline PointVaryings EditPointSprite(const thread Scene &scene, float4 position, float4 color) {
    position.z -= NdcOffsetFactor(scene) * 1.5f;
    return PointVaryings{position, PointSize, color};
}

inline PointVaryings ElementPointSprite(
    const thread Scene &scene, DrawData draw, float4 position, uint vertex_id
) {
    const uint state = EditVertexState(scene, draw, vertex_id);
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const float4 color = scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u ?
        scene.ObjectSelectionColor(scene.InstanceState(draw), float4(float3(colors.Vertex), 1.0f)) :
        EditVertexColor(scene, state);
    PointVaryings out = EditPointSprite(scene, position, color);
    if (scene.View.InteractionMode == InteractionMode_Excite &&
        (state & (STATE_SELECTED | STATE_ACTIVE)) == 0u) {
        out.PointSize = 0.0f;
    }
    return out;
}

#endif
