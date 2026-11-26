#pragma once
// Metal matches shared vertex-fragment fields by `user` name and permits fragment-stage subsets.
#include "ScreenSpace.metal"

struct MeshVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
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
};

// The shading members of a forward vertex.
// Flat per-face attributes go on unshared corner vertices because indexed output is nondeterministic on this driver.
#define MESHLET_SHADING_VARYINGS \
    float4 Position [[position]]; \
    float3 WorldNormal [[user(WorldNormal)]]; \
    float3 WorldPosition [[user(WorldPosition)]]; \
    float4 Color [[user(Color)]]; \
    float2 TexCoord0 [[user(TexCoord0)]]; \
    float2 TexCoord1 [[user(TexCoord1)]]; \
    float2 TexCoord2 [[user(TexCoord2)]]; \
    float2 TexCoord3 [[user(TexCoord3)]]; \
    float4 VertexColor [[user(VertexColor)]]; \
    float4 WorldTangent [[user(WorldTangent)]]; \
    float3 FlatWorldNormal [[user(FlatWorldNormal)]] [[flat]]; \
    uint FaceOverlayFlags [[user(FaceOverlayFlags)]] [[flat]]; \
    uint MaterialIndex [[user(MaterialIndex)]] [[flat]]; \
    float WorldScale [[user(WorldScale)]] [[flat]];

struct MeshletVertexVaryings {
    MESHLET_SHADING_VARYINGS
    uint ObjectId [[user(ObjectId)]] [[flat]];
    uint ElementId [[user(ElementId)]] [[flat]];
    int Facing [[user(Facing)]] [[flat]];
    uint Topology [[user(Topology)]] [[flat]];
    float2 PointCoord [[user(PointCoord)]];
};

// The forward varyings of a transparent triangle of a material that computes inputs: the shading members, then the surface in its object's space and the triangle's key.
struct MeshletPatternVaryings {
    MESHLET_SHADING_VARYINGS
    uint Topology [[user(Topology)]] [[flat]];
    float2 PointCoord [[user(PointCoord)]];
    float3 ObjectPosition [[user(ObjectPosition)]];
    float3 ObjectNormal [[user(ObjectNormal)]];
    // Half precision keeps the mesh output within its size limit.
    half3 ObjectTangent [[user(ObjectTangent)]];
    uint Key [[user(Key)]] [[flat]]; // The visibility id of the fragment's triangle, which identifies it among the frame's transparent triangles
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

// Sets the shading members a forward vertex shares with the mesh varyings.
template<typename V> inline void SetVertexVaryings(thread V &out, MeshVaryings v) {
    out.Position = v.Position;
    out.WorldNormal = v.WorldNormal;
    out.WorldPosition = v.WorldPosition;
    out.Color = v.Color;
    out.TexCoord0 = v.TexCoord0;
    out.TexCoord1 = v.TexCoord1;
    out.TexCoord2 = v.TexCoord2;
    out.TexCoord3 = v.TexCoord3;
    out.VertexColor = v.VertexColor;
    out.WorldTangent = v.WorldTangent;
}

template<typename V> inline MeshVaryings FromMeshletVaryings(V v) {
    return {
        .Position = v.Position,
        .WorldNormal = v.WorldNormal,
        .FlatWorldNormal = v.FlatWorldNormal,
        .WorldPosition = v.WorldPosition,
        .Color = v.Color,
        .FaceOverlayFlags = v.FaceOverlayFlags,
        .TexCoord0 = v.TexCoord0,
        .TexCoord1 = v.TexCoord1,
        .TexCoord2 = v.TexCoord2,
        .TexCoord3 = v.TexCoord3,
        .MaterialIndex = v.MaterialIndex,
        .VertexColor = v.VertexColor,
        .WorldTangent = v.WorldTangent,
        .WorldScale = v.WorldScale,
    };
}

struct QuadVaryings {
    float4 Position [[position]];
    float2 TexCoord [[user(TexCoord)]];
};

struct NdcVaryings {
    float4 Position [[position]];
    float2 Ndc [[user(Ndc)]];
};

struct GridVaryings {
    float4 Position [[position]];
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

struct EdgeQuadVaryings {
    float4 Position [[position]];
    float2 EdgeCoord [[user(EdgeCoord)]] [[center_no_perspective]];
    float EdgeLength [[user(EdgeLength)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
    float4 Color [[user(Color)]];
    float4 OuterColor [[user(OuterColor)]] [[flat]];
};

// Bone-fill output includes the winding sign for mirrored-instance culling.
struct BoneSolidVaryings {
    float4 Position [[position]];
    float4 Color [[user(Color)]];
    int Inverted [[user(Inverted)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
};

struct BoneSphereVaryings {
    float4 Position [[position]];
    float3 SphereCenter [[user(SphereCenter)]] [[flat]];
    uint ObjectId [[user(ObjectId)]] [[flat]];
    float3 ViewPos [[user(ViewPos)]];
    float4 BoneColor [[user(BoneColor)]] [[flat]];
    float4 StateColor [[user(StateColor)]] [[flat]];
    float SphereRadius [[user(SphereRadius)]] [[flat]];
};

struct ElementIdVaryings {
    float4 Position [[position]];
    float PointSize [[point_size]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

struct ElementIdFragmentVaryings {
    float4 Position [[position]];
    uint ElementId [[user(ElementId)]] [[flat]];
};

struct FaceIdFragmentVaryings {
    float4 Position [[position]];
    uint ElementId [[user(ElementId)]] [[flat]];
    int Facing [[user(Facing)]] [[flat]]; // Zero accepts both sides, positive accepts front faces, negative accepts back faces.
};

inline bool FaceCulled(int facing, bool front_facing) {
    return facing != 0 && front_facing != (facing > 0);
}

struct OverlayTargets {
    float4 Color [[color(0)]];
};

struct OverlayTargetsDepth {
    float4 Color [[color(0)]];
    float Depth [[depth(any)]];
};
