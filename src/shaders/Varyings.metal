#ifndef VARYINGS_MSL
#define VARYINGS_MSL

// Vertex-to-fragment interfaces, shared because several vertex shaders feed the same fragment
// shader. Metal matches these by `user` name, and a fragment may declare a subset of what the
// vertex wrote, so each fragment names only the members it reads.
#include "ScreenSpace.metal"

// The mesh transform's full output. WorkspaceLighting, VertexColor, and pbr each read a subset.
struct MeshVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]]; // Point draws render as round sprites.
    float3 WorldNormal [[user(WorldNormal)]];
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
    // Screen-space endpoints driving overlay line antialiasing.
    float2 EdgeStart [[user(EdgeStart)]] [[flat]];
    float2 EdgePos [[user(EdgePos)]];
    // Screen motion across the shutter, written only by the velocity variants.
    float3 MotionPrev [[user(MotionPrev)]];
    float3 MotionNext [[user(MotionNext)]];
};

struct MeshletVaryings {
    float4 Position [[position]];
    float PointSize [[user(PointSize)]];
    float3 WorldNormal [[user(WorldNormal)]];
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
    uint ObjectId [[user(ObjectId)]] [[flat]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

inline MeshletVaryings ToMeshletVaryings(MeshVaryings v) {
    return {
        v.Position, v.PointSize, v.WorldNormal, v.WorldPosition, v.Color, v.FaceOverlayFlags,
        v.TexCoord0, v.TexCoord1, v.TexCoord2, v.TexCoord3, v.MaterialIndex, v.VertexColor,
        v.WorldTangent, v.WorldScale, v.EdgeStart, v.EdgePos, v.MotionPrev, v.MotionNext, 0u, 0u,
    };
}

// What the VertexColor fragment reads, whichever vertex shader produced it.
struct LineVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    float2 EdgeStart [[user(EdgeStart)]] [[flat]];
    float2 EdgePos [[user(EdgePos)]];
};

// Full-screen triangle strip with a [0,1] texture coordinate.
struct QuadVaryings {
    float4 Position [[position]];
    float2 TexCoord [[user(TexCoord)]];
};

// Full-screen triangle strip carrying its NDC position, for the background passes.
struct NdcVaryings {
    float4 Position [[position]];
    float2 Ndc [[user(Ndc)]];
};

// The grid plane, whose outer vertices sit at infinity so w carries the projection.
struct GridVaryings {
    float4 Position [[position]];
    float4 PlanePos [[user(PlanePos)]];
};

// Object id alone, for the depth and selection pre-passes. Point draws size their sprites here too.
struct ObjectIdVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

struct ObjectIdFragmentVaryings {
    float4 Position [[position]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

// A colored point sprite.
struct PointVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    float4 Color [[user(Color)]];
};

// The edge-quad band: a screen-space distance from the edge center, plus its core and mark colors.
struct EdgeQuadVaryings {
    float4 Position [[position]];
    float EdgeCoord [[user(EdgeCoord)]] [[center_no_perspective]];
    float4 Color [[user(Color)]];
    float4 OuterColor [[user(OuterColor)]] [[flat]];
};

// The bone fill: a shaded color plus the winding sign, so a mirrored instance culls the right face.
struct BoneSolidVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    int Inverted [[user(Inverted)]] [[flat]];
};

// The bone joint billboard, carrying the view-space sphere the fragment ray-traces against.
struct BoneSphereVaryings {
    float4 Position [[position]];
    float3 SphereCenter [[user(SphereCenter)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
    float3 ViewPos [[user(ViewPos)]];
    float4 BoneColor [[user(BoneColor)]] [[flat]];
    float4 StateColor [[user(StateColor)]] [[flat]];
    float SphereRadius [[user(SphereRadius)]] [[flat]];
};

// Element id alone, for the element selection passes. Point topologies size their sprites here.
struct ElementIdVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

struct ElementIdFragmentVaryings {
    float4 Position [[position]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

// The two color targets every overlay fragment writes: color, and the packed line AA data.
struct OverlayTargets {
    float4 Color [[color(0)]];
    float4 LineData [[color(1)]];
};

// The overlay targets plus an explicit depth, for the passes that place their own fragment.
struct OverlayTargetsDepth {
    float4 Color [[color(0)]];
    float4 LineData [[color(1)]];
    float Depth [[depth(any)]];
};

// Clip space to the pixel coordinates a fragment reports, which pack_line_data measures against.
inline float2 clip_to_frag_co(float4 clip, float2 viewport_size) {
    return ndc_to_uv(clip.xy / clip.w) * viewport_size;
}

// Pack line edge data for the AA composite pass: the perpendicular direction and the signed
// distance to the line, encoded into [0,1].
inline float4 pack_line_data(float2 frag_co, float2 edge_start, float2 edge_pos) {
    float2 edge = edge_start - edge_pos;
    const float len = length(edge);
    if (len > 0.0f) {
        edge /= len;
        const float2 perp = float2(-edge.y, edge.x);
        const float dist = dot(perp, frag_co - edge_start);
        return float4(perp * 0.5f + 0.5f, dist * 0.25f + 0.6f, 1.0f);
    }
    // Zero-length edge: a stable default, perpendicular pointing right at zero distance.
    return float4(1.0f, 0.0f, 0.6f, 1.0f);
}

#endif
