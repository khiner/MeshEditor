#pragma once

#include "state/Slot.h"

struct GltfNode;
struct SourceIndex;
struct MeshSourceLayout;
struct HasVariantMappings;
namespace gltf {
struct SourceAssets;
struct MaterialSourceMeta;
} // namespace gltf

using GltfSchema = state::Schema<state::GltfBase, state::GltfCount, GltfNode, SourceIndex, MeshSourceLayout, HasVariantMappings, gltf::SourceAssets, gltf::MaterialSourceMeta>;
template<typename T>
    requires GltfSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = GltfSchema::Of<T>();
