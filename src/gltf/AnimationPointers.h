#pragma once

#include "animation/AnimationData.h"
#include "gpu/Transform.h"

#include <fastgltf/types.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The glTF Object Model pointers MeshEditor animates and the channel targets they name.
namespace gltf {
// The array a pointer indexes into.
enum class PointerSpace : uint8_t {
    Node,
    Material,
    Camera,
    Light,
    ImageLight,
};

struct PointerRow {
    std::string Template; // The pointer with "{}" in place of the index.
    PointerSpace Space;
    ChannelTarget Target; // Index 0. A weights target has Count 0 until bound to a mesh instance, and Count 1 for one weight.
};
const std::vector<PointerRow> &PointerRows();

// The row a classic channel path names, and the path of a node row.
const PointerRow &NodePathRow(fastgltf::AnimationPath);
fastgltf::AnimationPath NodePath(const PointerRow &);

struct ParsedPointer {
    const PointerRow *Row;
    uint32_t Index;
    std::optional<uint32_t> Element; // The second index of a row with two.
};
std::optional<ParsedPointer> ParsePointer(std::string_view pointer);

// The row whose target matches a channel target, treating a bone's pose delta as a node pose.
const PointerRow *RowOf(const ChannelTarget &);

// Converts a bone channel between glTF's absolute joint transform and MeshEditor's rest-relative delta.
// Cubic tangents take the linear part of the conversion.
void ConvertBoneChannel(AnimationChannel &, const Transform &rest, bool to_delta);
} // namespace gltf
