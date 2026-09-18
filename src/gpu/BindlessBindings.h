#pragma once

#include "gpu/Types.h"

#ifndef __METAL_VERSION__
#include <string_view>
#endif

enum class BindKind : uint8_t {
    Uniform,
    UniformDynamic,
    Image,
    Sampler,
    Buffer,
};

// Motion blur binds SceneViewUBO through a dynamic offset.
enum class SlotType : uint8_t {
    SceneViewUBO,
    ViewportThemeUBO,
    WorkspaceLightsUBO,
    Image,
    Sampler,
    CubeSampler,
    VertexBuffer,
    IndexBuffer,
    ModelBuffer,
    Buffer,
    ObjectIdBuffer,
    BoundsEntryBuffer,
    InstanceStateBuffer,
    BoneDeformBuffer,
    ArmatureDeformBuffer,
    MorphTargetBuffer,
    MorphWeightBuffer,
    VertexClassBuffer,
    LightBuffer,
    MaterialBuffer,
    PrimitiveMaterialBuffer,
    ElementPrimitiveBuffer,
    CornerTangentBuffer,
    CornerColorBuffer,
    CornerUvBuffer,
    Count
};
GPU_CONSTANT uint32_t SlotTypeCount = uint32_t(SlotType::Count);

// Shared capacities keep CPU and GPU slot indices identical.
// Corpus peaks are 40 buffers, 85 samplers, and 14 images, and arena-backed resources do not scale with scene size.
constexpr uint32_t SlotCapacity(BindKind kind) {
    switch (kind) {
        case BindKind::Uniform:
        case BindKind::UniformDynamic: return 1;
        case BindKind::Image:
        case BindKind::Sampler: return 1024;
        case BindKind::Buffer: return 256;
    }
    return 0;
}
// Byte size of one argument-table entry. Uniforms bind by buffer index and take no entry.
constexpr uint32_t SlotStride(BindKind kind) {
    switch (kind) {
        case BindKind::Uniform:
        case BindKind::UniformDynamic: return 0;
        case BindKind::Image:
        case BindKind::Buffer: return 8;
        case BindKind::Sampler: return 16;
    }
    return 0;
}

GPU_CONSTANT uint32_t BufferIndex_Bindless = 0;
GPU_CONSTANT uint32_t BufferIndex_PushConstants = 1;
GPU_CONSTANT uint32_t BufferIndex_SceneView = 2;
GPU_CONSTANT uint32_t BufferIndex_ViewportTheme = 3;
GPU_CONSTANT uint32_t BufferIndex_WorkspaceLights = 4;

#ifdef __METAL_VERSION__
struct BindlessSampler2D {
    texture2d<float> Texture;
    sampler Sampler;
};
struct BindlessSamplerCube {
    texturecube<float> Texture;
    sampler Sampler;
};
using BindlessBufferRef = device const uchar *;
#else
// The CPU writes 8-byte GPU resource ids and addresses into each entry.
struct BindlessSampler2D {
    uint64_t Texture, Sampler;
};
struct BindlessSamplerCube {
    uint64_t Texture, Sampler;
};
using BindlessBufferRef = uint64_t;
#endif

// Tier-2 argument buffer. Members follow SlotType order after the uniform bindings.
// The table uses device address space because it exceeds constant-space limits and contains device addresses.
// Image-writing shaders instantiate the same layout with a writable image type.
template<typename ImageT> struct BindlessSetT {
    GpuArray<ImageT, SlotCapacity(BindKind::Image)> Image;
    GpuArray<BindlessSampler2D, SlotCapacity(BindKind::Sampler)> Sampler;
    GpuArray<BindlessSamplerCube, SlotCapacity(BindKind::Sampler)> CubeSampler;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> VertexBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> IndexBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> ModelBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> Buffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> ObjectIdBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> BoundsEntryBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> InstanceStateBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> BoneDeformBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> ArmatureDeformBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> MorphTargetBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> MorphWeightBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> VertexClassBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> LightBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> MaterialBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> PrimitiveMaterialBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> ElementPrimitiveBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> CornerTangentBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> CornerColorBuffer;
    GpuArray<BindlessBufferRef, SlotCapacity(BindKind::Buffer)> CornerUvBuffer;
};
GPU_CONSTANT uint32_t BindlessTableSize = 79872;

#ifdef __METAL_VERSION__
using BindlessSet = BindlessSetT<texture2d<float, access::read>>;
using BindlessSetImageWrite = BindlessSetT<texture2d<float, access::write>>;

// Writable slots cast away the table's const view.
#define BindlessBuffer(T, table, slot) reinterpret_cast<device const T *>((table)[(slot)])
#define BindlessBufferMutable(T, table, slot) reinterpret_cast<device T *>(const_cast<device uchar *>((table)[(slot)]))
#else
using BindlessSet = BindlessSetT<uint64_t>;

struct BindingDef {
    BindKind Kind;
    std::string_view Name;
};
constexpr GpuArray<BindingDef, SlotTypeCount> BindingDefs{{
    {BindKind::UniformDynamic, "SceneViewUBO"},
    {BindKind::Uniform, "ViewportThemeUBO"},
    {BindKind::Uniform, "WorkspaceLightsUBO"},
    {BindKind::Image, "Image"},
    {BindKind::Sampler, "Sampler"},
    {BindKind::Sampler, "CubeSampler"},
    {BindKind::Buffer, "VertexBuffer"},
    {BindKind::Buffer, "IndexBuffer"},
    {BindKind::Buffer, "ModelBuffer"},
    {BindKind::Buffer, "Buffer"},
    {BindKind::Buffer, "ObjectIdBuffer"},
    {BindKind::Buffer, "BoundsEntryBuffer"},
    {BindKind::Buffer, "InstanceStateBuffer"},
    {BindKind::Buffer, "BoneDeformBuffer"},
    {BindKind::Buffer, "ArmatureDeformBuffer"},
    {BindKind::Buffer, "MorphTargetBuffer"},
    {BindKind::Buffer, "MorphWeightBuffer"},
    {BindKind::Buffer, "VertexClassBuffer"},
    {BindKind::Buffer, "LightBuffer"},
    {BindKind::Buffer, "MaterialBuffer"},
    {BindKind::Buffer, "PrimitiveMaterialBuffer"},
    {BindKind::Buffer, "ElementPrimitiveBuffer"},
    {BindKind::Buffer, "CornerTangentBuffer"},
    {BindKind::Buffer, "CornerColorBuffer"},
    {BindKind::Buffer, "CornerUvBuffer"},
}};

// Byte offset of a slot type's first table entry.
constexpr uint32_t BindlessOffset(SlotType type) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < uint32_t(type); ++i) offset += SlotStride(BindingDefs[i].Kind) * SlotCapacity(BindingDefs[i].Kind);
    return offset;
}
static_assert(BindlessOffset(SlotType::Count) == BindlessTableSize, "BindlessSet member order matches SlotType");
#endif
static_assert(sizeof(BindlessSet) == BindlessTableSize, "BindlessSet size");
