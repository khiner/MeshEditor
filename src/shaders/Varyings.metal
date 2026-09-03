#ifndef VARYINGS_MSL
#define VARYINGS_MSL

// Metal matches shared vertex-fragment fields by `user` name and permits fragment-stage subsets.
#include "ScreenSpace.metal"

struct MeshVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]]; // Point draws render as round sprites.
    float3 WorldNormal [[user(WorldNormal)]];
    float3 FlatWorldNormal [[user(FlatWorldNormal)]] [[flat]];
    float3 WorldPosition [[user(WorldPosition)]];
    float4 Color [[user(Color)]];
    uint FaceOverlayFlags [[user(FaceOverlayFlags)]] [[flat]];
    float2 TexCoord0 [[user(TexCoord0)]];
    float2 TexCoord1 [[user(TexCoord1)]];
    float2 TexCoord2 [[user(TexCoord2)]];
    float2 TexCoord3 [[user(TexCoord3)]];
    uint MaterialIndex [[user(MaterialIndex)]] [[flat]];
    float4 VertexColor [[user(VertexColor)]];
    float4 WorldTangent [[user(WorldTangent)]];
    float WorldScale [[user(WorldScale)]] [[flat]];
    float2 EdgeStart [[user(EdgeStart)]] [[flat]];
    float2 EdgePos [[user(EdgePos)]];
    float3 MotionPrev [[user(MotionPrev)]];
    float3 MotionNext [[user(MotionNext)]];
};

struct MeshletVertexVaryings {
    float4 Position [[position]];
    float3 WorldNormal [[user(WorldNormal)]];
    float3 WorldPosition [[user(WorldPosition)]];
    float4 Color [[user(Color)]];
    float2 TexCoord0 [[user(TexCoord0)]];
    float2 TexCoord1 [[user(TexCoord1)]];
    float2 TexCoord2 [[user(TexCoord2)]];
    float2 TexCoord3 [[user(TexCoord3)]];
    float4 VertexColor [[user(VertexColor)]];
    float4 WorldTangent [[user(WorldTangent)]];
    float3 MotionPrev [[user(MotionPrev)]];
    float3 MotionNext [[user(MotionNext)]];
    // Emit flat per-face attributes on unshared corner vertices because indexed output is nondeterministic on this driver.
    float3 FlatWorldNormal [[user(FlatWorldNormal)]] [[flat]];
    uint FaceOverlayFlags [[user(FaceOverlayFlags)]] [[flat]];
    uint MaterialIndex [[user(MaterialIndex)]] [[flat]];
    float WorldScale [[user(WorldScale)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
    uint ElementId [[user(ElementId)]] [[flat]];
    uint Topology [[user(Topology)]] [[flat]];
    float2 PointCoord [[user(PointCoord)]];
};

inline float3 ShadingWorldNormal(const thread MeshVaryings &v) {
    return (v.FaceOverlayFlags & 4u) != 0u ? v.FlatWorldNormal : v.WorldNormal;
}

struct MeshletPositionVaryings {
    float4 Position [[position]];
};

struct MeshletVisibilityPrimitiveVaryings {
    uint Id [[primitive_id]];
};

inline MeshletVertexVaryings ToMeshletVertexVaryings(MeshVaryings v) {
    return {
        v.Position, v.WorldNormal, v.WorldPosition, v.Color,
        v.TexCoord0, v.TexCoord1, v.TexCoord2, v.TexCoord3, v.VertexColor,
        v.WorldTangent, v.MotionPrev, v.MotionNext,
    };
}

inline MeshVaryings FromMeshletVertexVaryings(MeshletVertexVaryings v) {
    MeshVaryings out{};
    out.Position = v.Position;
    out.WorldNormal = v.WorldNormal;
    out.FlatWorldNormal = v.FlatWorldNormal;
    out.WorldPosition = v.WorldPosition;
    out.Color = v.Color;
    out.FaceOverlayFlags = v.FaceOverlayFlags;
    out.TexCoord0 = v.TexCoord0;
    out.TexCoord1 = v.TexCoord1;
    out.TexCoord2 = v.TexCoord2;
    out.TexCoord3 = v.TexCoord3;
    out.MaterialIndex = v.MaterialIndex;
    out.VertexColor = v.VertexColor;
    out.WorldTangent = v.WorldTangent;
    out.WorldScale = v.WorldScale;
    out.MotionPrev = v.MotionPrev;
    out.MotionNext = v.MotionNext;
    return out;
}

// VertexColor fragment input shared by its vertex producers.
struct LineVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    float2 EdgeStart [[user(EdgeStart)]] [[flat]];
    float2 EdgePos [[user(EdgePos)]];
};

// Line output shared by overlay and object-selection pipelines.
struct ObjectLineVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    float2 EdgeStart [[user(EdgeStart)]] [[flat]];
    float2 EdgePos [[user(EdgePos)]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

struct QuadVaryings {
    float4 Position [[position]];
    float2 TexCoord [[user(TexCoord)]];
};

struct NdcVaryings {
    float4 Position [[position]];
    float2 Ndc [[user(Ndc)]];
};

// Grid-plane output with infinite outer vertices represented through homogeneous w.
struct GridVaryings {
    float4 Position [[position]];
    float4 PlanePos [[user(PlanePos)]];
};

// Object ID output for depth and selection prepasses.
struct ObjectIdVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

struct ObjectIdFragmentVaryings {
    float4 Position [[position]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

struct PointVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    float4 Color [[user(Color)]];
};

// Edge-quad coverage and color output.
struct EdgeQuadVaryings {
    float4 Position [[position]];
    float EdgeCoord [[user(EdgeCoord)]] [[center_no_perspective]];
    float4 Color [[user(Color)]];
    float4 OuterColor [[user(OuterColor)]] [[flat]];
};

// Bone-fill output includes the winding sign for mirrored-instance culling.
struct BoneSolidVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    int Inverted [[user(Inverted)]] [[flat]];
};

// Bone-joint billboard output includes its view-space sphere.
struct BoneSphereVaryings {
    float4 Position [[position]];
    float3 SphereCenter [[user(SphereCenter)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
    float3 ViewPos [[user(ViewPos)]];
    float4 BoneColor [[user(BoneColor)]] [[flat]];
    float4 StateColor [[user(StateColor)]] [[flat]];
    float SphereRadius [[user(SphereRadius)]] [[flat]];
};

// Element ID output for selection passes.
struct ElementIdVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

struct ElementIdFragmentVaryings {
    float4 Position [[position]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

struct OverlayTargets {
    float4 Color [[color(0)]];
    float4 LineData [[color(1)]];
};

struct OverlayTargetsDepth {
    float4 Color [[color(0)]];
    float4 LineData [[color(1)]];
    float Depth [[depth(any)]];
};

// Converts clip space to top-down pixel coordinates.
inline float2 clip_to_frag_co(float4 clip, float2 viewport_size) {
    return ndc_to_uv(clip.xy / clip.w) * viewport_size;
}

inline LineVaryings MakeLineVertex(float4 clip, float4 color, float2 viewport_size) {
    const float2 screen_pos = clip_to_frag_co(clip, viewport_size);
    return {clip, color, screen_pos, screen_pos};
}

// Packs perpendicular direction and signed line distance into [0, 1] for composite antialiasing.
inline float4 pack_line_data(float2 frag_co, float2 edge_start, float2 edge_pos) {
    float2 edge = edge_start - edge_pos;
    const float len = length(edge);
    if (len > 0.0f) {
        edge /= len;
        const float2 perp = float2(-edge.y, edge.x);
        const float dist = dot(perp, frag_co - edge_start);
        return float4(perp * 0.5f + 0.5f, dist * 0.25f + 0.6f, 1.0f);
    }
    // Use a fixed perpendicular for zero-length edges.
    return float4(1.0f, 0.0f, 0.6f, 1.0f);
}

#endif
