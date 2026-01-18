#pragma once

#include <array>
#include <cstdint>
#include <string_view>

enum class BindKind : uint8_t {
    Uniform,
    Image,
    Sampler,
    Buffer,
};

struct BindlessBindingField {
    BindKind Kind;
    std::string_view Name;
};

constexpr std::array BindlessBindingFields{
    BindlessBindingField{BindKind::Uniform, "Uniform"},
    BindlessBindingField{BindKind::Image, "Image"},
    BindlessBindingField{BindKind::Sampler, "Sampler"},
    BindlessBindingField{BindKind::Buffer, "VertexBuffer"},
    BindlessBindingField{BindKind::Buffer, "IndexBuffer"},
    BindlessBindingField{BindKind::Buffer, "ModelBuffer"},
    BindlessBindingField{BindKind::Buffer, "Buffer"},
    BindlessBindingField{BindKind::Buffer, "ObjectIdBuffer"},
    BindlessBindingField{BindKind::Buffer, "FaceNormalBuffer"},
    BindlessBindingField{BindKind::Buffer, "DrawDataBuffer"},
};
