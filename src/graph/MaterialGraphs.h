#pragma once

#include "gltf/GltfScene.h"
#include "graph/Cook.h"
#include "materialx/Library.h"
#include "ops/Compile.h"
#include "state/Entity.h"

#include <optional>
#include <vector>

namespace state {
struct Registry;
} // namespace state

namespace graph {
// Derived on a material entity whose graph links a Surface input: the program the linked inputs compile to on the surface domain, which graphs of the same shape share.
// Program is its index in the program registry, or none where a node of the graph faults.
// Base is the word of the material's op value block the program reads its values from, and Images the document texture each slot of its image table reads.
// Attributes holds the named attributes the program reads, by attribute table slot.
struct MaterialProgram {
    std::optional<uint32_t> Program;
    uint32_t Needs{}, Inputs{}, Words{}, Base{};
    std::vector<state::Entity> Images;
    std::vector<ops::Attribute> Attributes;
};

void RegisterMaterialGraphHandlers(state::Registry &);
// Notes that a literal of a material's graph, or of a subgraph a graph may use, changed, which the previews show.
void NoteMaterialValues(state::Registry &);
// The preview of the material on a sphere under fixed lighting, or of the first output of the op node at `path` in its graph shown at the sphere's points, or none for a node whose output has no value.
// Renders it when the material, its graph or its textures changed since the image a panel holds was rendered, and the panel's hold keeps the image.
std::optional<ImageTextures> MaterialPreviewOf(state::Registry &, state::Entity material, std::span<const state::Entity> path);
// The material as a MaterialX document: its graph's op nodes as MaterialX nodes, a graph output for each linked Surface input, and the material's values for the gltf_pbr inputs no output computes.
// A subgraph's nodes are written in place of the subgraph node, and a blur reads as its input, as the material computes them.
std::expected<mx::MaterialDocument, std::string> MaterialXOf(const state::Registry &, state::Entity material);
// The MaterialX document text of the material, holding the subgraphs of MeshEditor's own nodes it uses.
std::expected<std::string, std::string> WriteMaterialX(const state::Registry &, state::Entity material);
// The baked inputs of the materials whose graphs compute inputs, by mesh that draws one, and a message for each material, input or mesh that exports the material's values in place of a bake.
struct MaterialBake {
    std::vector<gltf::BakedMaterial> Materials;
    std::vector<std::string> Unbaked;
};
// The inputs each material's graph computes, baked at `size` by `size` texels.
// An input that reads nothing of a mesh but its first texture coordinates bakes once over the unit square, by the program on the UV domain, for every mesh drawing the material.
// Another bakes over the first texture coordinates of each mesh drawing the material, by the program on the texel domain, where every texel beyond the layout takes the value at the nearest point of the layout, so each mip level of the texture holds the layout's values.
// A mesh without texture coordinates, an input whose node does not evaluate on texels, and a material whose inputs conflict with its textures bake nothing and are listed.
MaterialBake BakeMaterialInputs(state::Registry &, uint32_t size);
// Compiles the graph of each material whose graph changed, keeps the faults of its nodes, and writes every material's pattern where any computes inputs.
void SettleMaterialGraphs(state::Registry &);
// Replaces the program registry and its drivers, and compiles every material's graph into the new registry.
void ResetPrograms(state::Registry &);
} // namespace graph
