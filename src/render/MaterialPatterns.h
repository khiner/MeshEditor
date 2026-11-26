#pragma once

#include "Range.h"
#include "gpu/MaterialPattern.h"
#include "gpu/PbrFeature.h"
#include "metal/Buffer.h"
#include "metal/Program.h"
#include "render/RenderSchema.h"

#include <memory>

struct GpuBuffers;
struct MeshStore;
struct SceneViewUBO;
namespace state {
struct Registry;
} // namespace state

// Derived: the pattern of each of the Count materials by record index, then the tables they read, while some material's graph computes inputs.
struct MaterialPatterns {
    std::unique_ptr<mtl::Buffer> Records;
    uint32_t Count{};
    std::vector<Range> Values; // Each pattern's words of the op value store
    uint32_t Words{}; // The most words a pattern writes
    // The pattern pass's pipelines and the ray hit driver's, created by the first frame that draws a patterned material.
    struct Passes {
        mtl::ComputePipeline Classify, Arguments, HitArguments;
        mtl::ComputeDriver &Driver, &HitDriver;
    };
    std::unique_ptr<Passes> Pipelines;
    // The material preview's pipelines and sphere, created by the first preview: the sphere's triangle ids, the display, the sphere's corners and their count, the program driver once a preview evaluates a program, and the lit pipelines by features.
    struct PreviewPasses {
        mtl::RenderPipeline Sphere;
        mtl::ComputePipeline Display;
        mtl::Buffer Corners;
        uint32_t CornerCount;
        mtl::ComputeDriver *Driver{};
        std::vector<std::pair<PbrFeatureMask, mtl::RenderPipeline>> Lit;
    };
    std::optional<PreviewPasses> Previews;
};
void Reset(MaterialPatterns &, state::Registry &);
// The pattern of each material by record index, or none while no material's graph computes inputs.
std::span<const MaterialPattern> PatternRecords(const MaterialPatterns &);
// Patterns and the bits of the lobes that the computed inputs, a mask by PbrInput, make live.
PbrFeatureMask PatternFeatures(uint32_t inputs);
// The mesh attributes the pattern reads, which coarse clusters keep, where a mesh with `authored_tangents` keeps its tangents.
uint32_t PatternLodAttributes(const MaterialPattern &, bool authored_tangents);

// The features of the patterned materials the view's shown instances draw, with Patterns among them when any draws one, and whether any is blended or transmissive.
struct DrawnPatterns {
    PbrFeatureMask Features{};
    bool Transparent{};
};
DrawnPatterns DrawnPatternFeatures(const MaterialPatterns &, const MeshStore &, const GpuBuffers &, const SceneViewUBO &);
