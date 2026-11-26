#pragma once

#include "metal/Program.h"
#include "ops/Compile.h"
#include "ops/OpsSchema.h"

#include "Range.h"
#include "gpu/OpPoint.h"
#include "gpu/SlotOffset.h"

struct MeshPipelines;
struct MeshStore;
struct PositionProgram;
namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

namespace ops {
// A graph's program for a domain, or its nodes' faults, and the program's index in the program registry once linked.
struct Compiled {
    std::expected<Program, std::vector<NodeFault>> Program;
    std::optional<uint32_t> Index{};
};
// The drivers that call programs, registered with the registry's program registry, and the programs compiled for graphs.
// Elements evaluates at every live element of a record's domain into attributes, Masks into selection masks, Positions as a position edit's per-vertex input, and Point at one point.
// Texels evaluates at each texel of a grid over a record's UV layout into an image, and Grid at each texel of a grid over the unit square of texture coordinates, each registered when an image first needs it.
struct Drivers {
    mtl::ComputeDriver &Elements, &Masks, &Positions, &Point;
    mtl::ProgramRegistry &Registry;
    const MeshPipelines &Pipelines; // The image passes, and the pipelines of the drivers registered on first use
    mtl::ComputeDriver *Texels{}, *Grid{};
    std::unordered_map<store::Hash128, Compiled, store::Hash128Hasher> Programs{}; // By the hash of the graph and domain
    std::vector<store::Hash128> Unlinked{};
};
// Creates the program registry and registers the drivers on first use.
Drivers &GetDrivers(state::Registry &);
// The hash of the graph's shape on the domain, which holds its literal values and the words it reads, and none of the words' values.
store::Hash128 ShapeOf(const Graph &, Domain);
// The graph's compilation for the domain, which graphs of the same shape share.
Compiled &CompileShared(Drivers &, const Graph &, Domain);
// The domain the record holds its colors on: vertices for a record without faces, and else corners.
Domain ColorDomain(const MeshStore &, uint32_t id);
// Links the compiled programs no index was given yet, in one batch.
void LinkCompiled(state::Registry &, Drivers &);

// The named attribute domain of a mesh domain.
AttributeDomain AttributeDomainOf(Domain);
Domain DomainOf(AttributeDomain);

// The table of a program on the domain that reads each attribute, an attribute the domain stores or one it reads through its vertex or face, or a weighted point's through its corners' vertices or face, and no values slot for InvalidOffset.
std::vector<OpAttributeRef> AttributeTable(const MeshStore &, Domain, std::span<const uint32_t> attributes);
// Records the linked program at every live element of the record's domain, writing output o to `outputs[o]`.
// `values` is its value block in a buffer slot, and `images` its image table.
void RecordElements(mtl::ComputeChain &, const Drivers &, const MeshStore &, uint32_t id, Domain, const Compiled &, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, std::span<const NamedAttributeRef> outputs, float frame, float time);
// Records the boolean program at every live element of the record's domain, setting the bit in `masks` of each element it returns true at.
// Mask block b covers element block b at word masks.Offset + 8b, and the masks hold zero before.
void RecordMasks(mtl::ComputeChain &, const Drivers &, const MeshStore &, uint32_t id, Domain, const Compiled &, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, SlotOffset masks, float frame, float time);
// The program's call at each vertex of the record as the per-vertex input of a position edit, which reads its first output.
PositionProgram PositionInput(const Drivers &, const MeshStore &, uint32_t id, const Compiled &, SlotOffset values, std::vector<OpAttributeRef> attributes, std::vector<OpImageTexture> images, float frame, float time);
// Records the program at one point, and returns the chain's scratch words that hold its output words once the chain submits.
Range RecordPoint(mtl::ComputeChain &, const Drivers &, const Compiled &, SlotOffset values, std::span<const OpImageTexture> images, float frame, float time);

// A Width by Height texel grid over UV set UvSet of a record, whose texels evaluate within Margin texels of the layout.
struct TexelGrid {
    uint32_t UvSet, Width, Height, Margin;
};
// The floats a texel of an image of values of the components takes: one or two for that many, and else four.
constexpr uint32_t TexelStride(uint32_t components) { return components <= 2u ? components : 4u; }
// Records the program at each texel of the grid over the record's UV layout, writing its first output's TexelStride floats per texel to `output`, row by row from the top, and zero at texels beyond the margin.
// The record has a layout in the grid's UV set.
void RecordTexels(mtl::ComputeChain &, Drivers &, const MeshStore &, uint32_t id, const Compiled &, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, TexelGrid, SlotOffset output, float frame, float time);

// A grid of Width by Height texels over the unit square of texture coordinates, with Apron more texels beyond each side.
struct UvGrid {
    uint32_t Width, Height, Apron;
};
// The texels of the grid with its apron.
constexpr uint32_t GridTexels(UvGrid grid) { return (grid.Width + 2u * grid.Apron) * (grid.Height + 2u * grid.Apron); }
// Records the program on the UV domain at each texel of the grid, writing its first output's TexelStride floats per texel to `output`, row by row from the top.
void RecordGrid(mtl::ComputeChain &, Drivers &, const Compiled &, SlotOffset values, std::span<const OpImageTexture> images, UvGrid, SlotOffset output, float frame, float time);
// Records a blur of the texels of the grid, Stride floats each, into the texels of its square, row by row from the top.
// Each texel takes the mean of the texels around it, clamped to the grid, weighted by a box `size` wide in texture coordinates along each axis, or by a Gaussian whose deviation is a sixth of that.
void RecordBlur(mtl::ComputeChain &, const MeshPipelines &, UvGrid, SlotOffset texels, uint32_t stride, float size, bool gaussian, SlotOffset output);
} // namespace ops
