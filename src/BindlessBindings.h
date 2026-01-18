#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

enum class SlotType : uint8_t {
    Uniform,
    Image,
    Sampler,
    VertexBuffer,
    IndexBuffer,
    ModelBuffer,
    Buffer,
    ObjectIdBuffer,
    FaceNormalBuffer,
    DrawDataBuffer,
    Count
};

constexpr size_t SlotTypeCount{static_cast<size_t>(SlotType::Count)};

enum class BindKind : uint8_t {
    uniform,
    image,
    sampler,
    buffer,
};

struct BindingDef {
    BindKind Kind;
    std::string_view Name;
};

constexpr std::array<BindingDef, SlotTypeCount> BindingDefs{{
    {BindKind::uniform, "Uniform"},
    {BindKind::image, "Image"},
    {BindKind::sampler, "Sampler"},
    {BindKind::buffer, "VertexBuffer"},
    {BindKind::buffer, "IndexBuffer"},
    {BindKind::buffer, "ModelBuffer"},
    {BindKind::buffer, "Buffer"},
    {BindKind::buffer, "ObjectIdBuffer"},
    {BindKind::buffer, "FaceNormalBuffer"},
    {BindKind::buffer, "DrawDataBuffer"},
}};

constexpr const BindingDef &GetBinding(SlotType type) { return BindingDefs[static_cast<size_t>(type)]; }
