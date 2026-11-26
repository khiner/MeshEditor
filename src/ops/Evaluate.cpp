#include "ops/Evaluate.h"
#include "Variant.h"
#include "gpu/BakeJob.h"
#include "gpu/ImageFilterPushConstants.h"
#include "gpu/OpDriver.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/PositionOperations.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

#include <bit>

namespace ops {
namespace {
// The stored values the domain's point fields read.
OpGeometry GeometryOf(const MeshStore &meshes, uint32_t id, Domain domain) {
    const auto &a = meshes.Arenas();
    const auto layers = meshes.Layers(id);
    const bool vertex = domain == Domain::Vertex, corner = domain == Domain::Corner;
    return {
        .Positions = vertex || corner ? meshes.Slots().Vertices : InvalidSlot,
        .Normals = vertex || corner ? a.BaseVertexNormals.Buffer.Slot : domain == Domain::Face ? a.BaseFaceNormals.Buffer.Slot : InvalidSlot,
        .FaceNormals = corner ? a.BaseFaceNormals.Buffer.Slot : InvalidSlot,
        .FaceSharpness = corner ? a.FaceSharpness.Buffer.Slot : InvalidSlot,
        .ClassMode = corner ? meshes.Get(id).Classification : InvalidOffset,
        .CornerSectors = corner ? layers.CornerSectors : SlotOffset{},
        .NormalSectors = corner ? layers.NormalSectors : SlotOffset{},
        .CustomNormals = corner ? layers.CustomNormals : SlotOffset{},
        .Tangents = corner ? layers.CornerTangent : SlotOffset{},
        .Colors = vertex ? layers.VertexColor : corner ? layers.CornerColor : SlotOffset{},
        .Texcoords = corner ? layers.CornerUvs : GpuArray<SlotOffset, 4>{},
        .ColorAbsent = vertex ? a.VertexColors.Absent : a.CornerColors.Absent,
        .Connectivity = corner ? meshes.GetConnectivityRef(id) : ConnectivityRef{},
    };
}

// The program's call at the live elements of the record's domain, without its named attribute and image tables.
// A texel's call has its record's corner geometry.
OpCall CallOf(const MeshStore &meshes, uint32_t id, Domain domain, const Compiled &compiled, SlotOffset values, float frame, float time) {
    const auto stored = StoringDomain(domain);
    return {
        .Program = {.Index = *compiled.Index, .Needs = compiled.Program->Needs, .Values = values},
        .Frame = frame,
        .Time = time,
        .Blocks = BlocksOf(meshes, id, ElementDomainOf(AttributeDomainOf(stored))),
        .Geometry = GeometryOf(meshes, id, stored),
    };
}
template<typename T> SlotOffset UploadTable(mtl::ComputeChain &chain, std::span<const T> table) { return table.empty() ? SlotOffset{} : chain.Upload(std::as_bytes(table), alignof(T)); }
// The driver of the pass, registered on first use.
mtl::ComputeDriver &Registered(Drivers &drivers, mtl::ComputeDriver *&driver, MeshPass pass) {
    if (!driver) driver = &mtl::AddDriver(drivers.Registry, drivers.Pipelines[pass]);
    return *driver;
}
// Records the driver over `threads` threads of the job, whose call reads the attribute and image tables, with the jobs the other pending lanes gather for the driver at the same step.
void GatherDriver(mtl::ComputeChain &chain, const mtl::ComputeDriver &driver, auto job, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, uint32_t threads) {
    job.Call.Program.Attributes = UploadTable(chain, attributes);
    job.Call.Program.Images = UploadTable(chain, images);
    mtl::GatherPass(chain, driver, job, threads);
}
} // namespace

store::Hash128 ShapeOf(const Graph &graph, Domain domain) {
    std::vector<std::byte> shape;
    const auto put = [&](const auto &value) { shape.append_range(std::as_bytes(std::span{&value, 1})); };
    const auto put_text = [&](std::string_view text) {
        put(text.size());
        shape.append_range(std::as_bytes(std::span{text}));
    };
    put(domain);
    put(graph.Colors);
    for (const auto &node : graph.Nodes) {
        put(node.Op.index());
        if (const auto *name = std::get_if<std::string>(&node.Op)) put_text(*name);
        else put(std::get<uint32_t>(node.Op));
        put(node.Output.value_or(mx::Type::MultiOutput));
        put(node.Inputs.size());
        for (const auto &[name, source] : node.Inputs) {
            put_text(name);
            put(source.index());
            std::visit(
                overloaded{
                    [&](const Literal &literal) {
                        put(literal.Type);
                        if (const auto *text = std::get_if<std::string>(&literal.Value)) put_text(*text);
                        else shape.append_range(std::as_bytes(std::span{std::get<std::vector<uint32_t>>(literal.Value)}));
                    },
                    [&](Word word) {
                        put(word.Type);
                        put(word.Offset);
                    },
                    [&](auto other) { put(other); }, // A link or an image
                },
                source
            );
        }
    }
    for (const auto &output : graph.Outputs) put(output);
    return store::HashBytes(shape);
}

Domain ColorDomain(const MeshStore &meshes, uint32_t id) { return BlocksOf(meshes, id, ElementDomainOf(AttributeDomain::Face)).Count ? Domain::Corner : Domain::Vertex; }

Compiled &CompileShared(Drivers &drivers, const Graph &graph, Domain domain) {
    const auto hash = ShapeOf(graph, domain);
    if (const auto found = drivers.Programs.find(hash); found != drivers.Programs.end()) return found->second;
    drivers.Unlinked.push_back(hash);
    return drivers.Programs.emplace(hash, Compiled{Compile(mx::Libraries(), graph, domain)}).first->second;
}

void LinkCompiled(state::Registry &r, Drivers &drivers) {
    std::vector<Compiled *> pending;
    std::vector<std::string> sources;
    for (const auto &hash : drivers.Unlinked) {
        auto &compiled = drivers.Programs.at(hash);
        if (!compiled.Program) continue;
        pending.push_back(&compiled);
        sources.push_back(compiled.Program->Source);
    }
    drivers.Unlinked.clear();
    if (pending.empty()) return;
    const auto indices = mtl::AddPrograms(r.Resources.get<mtl::ProgramRegistry>(), sources);
    for (size_t i = 0; i < pending.size(); ++i) pending[i]->Index = indices[i];
}

AttributeDomain AttributeDomainOf(Domain domain) {
    switch (domain) {
        case Domain::Vertex: return AttributeDomain::Vertex;
        case Domain::Edge: return AttributeDomain::Edge;
        case Domain::Face: return AttributeDomain::Face;
        case Domain::Corner: return AttributeDomain::Corner;
        case Domain::Point:
        case Domain::Texel:
        case Domain::Uv:
        case Domain::Surface: break;
    }
    std::unreachable(); // One point, a texel grid and surface points store no attribute
}
Domain DomainOf(AttributeDomain domain) {
    switch (domain) {
        case AttributeDomain::Vertex: return Domain::Vertex;
        case AttributeDomain::Edge: return Domain::Edge;
        case AttributeDomain::Face: return Domain::Face;
        case AttributeDomain::Corner: return Domain::Corner;
    }
    std::unreachable();
}

Drivers &GetDrivers(state::Registry &r) {
    if (auto *drivers = r.Resources.find<Drivers>()) return *drivers;
    auto &programs = r.Resources.emplace<mtl::ProgramRegistry>(r.Resources.get<mtl::LibraryCache>().Ctx);
    const auto &pipelines = GetMeshPipelines(r);
    const auto add = [&](MeshPass pass) -> mtl::ComputeDriver & { return mtl::AddDriver(programs, pipelines[pass]); };
    return r.Resources.emplace<Drivers>(add(MeshPass::OpElements), add(MeshPass::OpMasks), add(MeshPass::OpPositions), add(MeshPass::OpPoint), programs, pipelines);
}

std::vector<OpAttributeRef> AttributeTable(const MeshStore &meshes, Domain domain, std::span<const uint32_t> attributes) {
    std::vector<OpAttributeRef> table;
    const bool weighted = Weighted(domain);
    for (const auto attribute : attributes) {
        const auto held = attribute == InvalidOffset ? StoringDomain(domain) : DomainOf(meshes.GetAttribute(attribute).Domain);
        const auto element = held == StoringDomain(domain) ? (weighted ? OpElement::Corners : OpElement::Own) : !Through(held, domain) ? std::optional<OpElement>{} :
            held == Domain::Vertex                                                                                                    ? (weighted ? OpElement::Vertices : OpElement::Vertex) :
                                                                                                                                        OpElement::Face;
        table.push_back(attribute != InvalidOffset && element ? OpAttributeRef{meshes.AttributeRef(attribute), *element} : OpAttributeRef{});
    }
    return table;
}

void RecordElements(mtl::ComputeChain &chain, const Drivers &drivers, const MeshStore &meshes, uint32_t id, Domain domain, const Compiled &compiled, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, std::span<const NamedAttributeRef> outputs, float frame, float time) {
    OpElementsJob job{.Call = CallOf(meshes, id, domain, compiled, values, frame, time), .OutputCount = uint32_t(outputs.size())};
    for (size_t o = 0; o < outputs.size(); ++o) job.Outputs[o] = {outputs[o], *StorageOf(compiled.Program->Outputs[o])};
    GatherDriver(chain, drivers.Elements, job, attributes, images, job.Call.Blocks.Count * MeshElementBlockSize);
}

void RecordMasks(mtl::ComputeChain &chain, const Drivers &drivers, const MeshStore &meshes, uint32_t id, Domain domain, const Compiled &compiled, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, SlotOffset masks, float frame, float time) {
    const OpMasksJob job{.Call = CallOf(meshes, id, domain, compiled, values, frame, time), .Masks = masks};
    GatherDriver(chain, drivers.Masks, job, attributes, images, job.Call.Blocks.Count * MeshElementBlockSize);
}

PositionProgram PositionInput(const Drivers &drivers, const MeshStore &meshes, uint32_t id, const Compiled &compiled, SlotOffset values, std::vector<OpAttributeRef> attributes, std::vector<OpImageTexture> images, float frame, float time) {
    return {&drivers.Positions, CallOf(meshes, id, Domain::Vertex, compiled, values, frame, time), std::move(attributes), std::move(images)};
}

Range RecordPoint(mtl::ComputeChain &chain, const Drivers &drivers, const Compiled &compiled, SlotOffset values, std::span<const OpImageTexture> images, float frame, float time) {
    uint32_t words = 0;
    for (const auto type : compiled.Program->Outputs) words += mx::Words(type);
    const auto result = chain.Scratch.Allocate(words);
    const OpPointPushConstants pc{
        .Program = {.Index = *compiled.Index, .Needs = compiled.Program->Needs, .Values = values, .Images = UploadTable(chain, images)},
        .Result = {chain.Scratch.Buffer.Slot, result.Offset},
        .Frame = frame,
        .Time = time,
        .Words = words,
    };
    chain.Threads(drivers.Point, pc, 1u, 1u);
    return result;
}

void RecordTexels(mtl::ComputeChain &chain, Drivers &drivers, const MeshStore &meshes, uint32_t id, const Compiled &compiled, SlotOffset values, std::span<const OpAttributeRef> attributes, std::span<const OpImageTexture> images, TexelGrid grid, SlotOffset output, float frame, float time) {
    const auto &pipelines = drivers.Pipelines;
    const auto scratch = chain.Scratch.Buffer.Slot;
    const auto triangles = BlocksOf(meshes, id, MeshStore::ElementDomain::Triangle);
    const auto slots = triangles.Count * MeshElementBlockSize, texels = grid.Width * grid.Height;
    BakeJob job{
        .Triangles = triangles,
        .TrianglesSlot = meshes.Arenas().Triangles.Buffer.Slot,
        .Uvs = meshes.Layers(id).CornerUvs[grid.UvSet],
        .Counts = {scratch, TakeScratch(chain, slots + 1u).Offset},
        .Texels = {scratch, TakeScratch(chain, texels).Offset},
        .Jumped = {scratch, TakeScratch(chain, grid.Margin ? texels : 0u).Offset},
        .Width = grid.Width,
        .Height = grid.Height,
    };
    // Each texel holds the triangle whose point it evaluates at: a covering triangle, or else the nearest one the margin's jumps reach.
    FillScratch(chain, pipelines, std::array{ScratchFillRun{job.Texels.Offset, texels, 0u}});
    mtl::GatherPass(chain, pipelines[MeshPass::BakeTriangleCounts], job, slots + 1u, BakeTile);
    ScanScratch(chain, pipelines, std::array{Range{job.Counts.Offset, slots + 1u}});
    mtl::GatherPass(chain, pipelines[MeshPass::BakeCoverTexels], job, std::max(texels, slots), BakeTile);
    if (grid.Margin) {
        for (auto step = std::bit_ceil(grid.Margin);; step /= 2u) {
            job.Step = std::max(step, 1u);
            mtl::GatherPass(chain, pipelines[MeshPass::BakeMarginJump], job, texels, BakeTile);
            std::swap(job.Texels, job.Jumped);
            if (step == 0u) break;
        }
    }
    const auto components = mx::Words(compiled.Program->Outputs.front());
    const OpTexelsJob call{
        .Call = CallOf(meshes, id, Domain::Texel, compiled, values, frame, time),
        .Texels = job.Texels,
        .Output = output,
        .TrianglesSlot = job.TrianglesSlot,
        .Width = grid.Width,
        .Height = grid.Height,
        .Margin = grid.Margin,
        .UvSet = grid.UvSet,
        .Components = components,
        .Stride = TexelStride(components),
    };
    GatherDriver(chain, Registered(drivers, drivers.Texels, MeshPass::OpTexels), call, attributes, images, texels);
}

void RecordGrid(mtl::ComputeChain &chain, Drivers &drivers, const Compiled &compiled, SlotOffset values, std::span<const OpImageTexture> images, UvGrid grid, SlotOffset output, float frame, float time) {
    const auto components = mx::Words(compiled.Program->Outputs.front());
    const OpGridPushConstants pc{
        .Program = {.Index = *compiled.Index, .Needs = compiled.Program->Needs, .Values = values, .Images = UploadTable(chain, images)},
        .Output = output,
        .Frame = frame,
        .Time = time,
        .Width = grid.Width,
        .Height = grid.Height,
        .Apron = grid.Apron,
        .Components = components,
        .Stride = TexelStride(components),
    };
    chain.Threads(Registered(drivers, drivers.Grid, MeshPass::OpGrid), pc, GridTexels(grid));
}

void RecordBlur(mtl::ComputeChain &chain, const MeshPipelines &pipelines, UvGrid grid, SlotOffset texels, uint32_t stride, float size, bool gaussian, SlotOffset output) {
    const auto rows_height = grid.Height + 2u * grid.Apron;
    // The rows pass blurs each grid row into the square's columns, and the columns pass blurs those into the square's rows.
    const SlotOffset rows{chain.Scratch.Buffer.Slot, TakeScratch(chain, uint64_t(grid.Width) * rows_height * stride).Offset};
    const ImageFilterPushConstants along_rows{
        .Input = texels,
        .Output = rows,
        .InWidth = grid.Width + 2u * grid.Apron,
        .InHeight = rows_height,
        .OutWidth = grid.Width,
        .OutHeight = rows_height,
        .Offset = grid.Apron,
        .Stride = stride,
        .Gaussian = gaussian,
        .Width = size * float(grid.Width),
    };
    chain.Threads(pipelines[MeshPass::BlurRows], along_rows, grid.Width * rows_height);
    const ImageFilterPushConstants along_columns{
        .Input = rows,
        .Output = output,
        .InWidth = grid.Width,
        .InHeight = rows_height,
        .OutWidth = grid.Width,
        .OutHeight = grid.Height,
        .Offset = grid.Apron,
        .Stride = stride,
        .Gaussian = gaussian,
        .Width = size * float(grid.Height),
    };
    chain.Threads(pipelines[MeshPass::BlurColumns], along_columns, grid.Width * grid.Height);
}
} // namespace ops
