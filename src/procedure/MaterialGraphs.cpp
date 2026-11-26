#include "procedure/MaterialGraphs.h"
#include "SortUnique.h"
#include "animation/Keying.h"
#include "gltf/GltfScene.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "ops/Evaluate.h"
#include "procedure/OpNodes.h"
#include "project/Assets.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/MaterialPatterns.h"
#include "render/MaterialPreview.h"
#include "render/RenderChanges.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "state/Registry.h"
#include "timeline/Timeline.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"

#include <format>

namespace procedure {
// The materials whose graphs changed, the meshes' nodes among them, and anything else every pattern reads: the material records and the document's textures.
// Meshes holds the meshes whose records or material assignments changed, whose named attributes a pattern may read.
// Revision counts the changes of anything a preview shows, and each preview the panels hold keeps the revision it shows.
struct MaterialGraphChanges {
    state::DirtySet Changed, Meshes;
    uint64_t Revision{};
    struct Preview {
        state::Entity Material;
        NodePath Path;
        uint64_t Revision;
        std::weak_ptr<const PreviewImage> Image;
    };
    std::vector<Preview> Previews;
};

namespace {
// The attribute table of a material's program on the weighted domain at record `id`, which reads each named attribute of its name and storage type the record holds.
std::vector<OpAttributeRef> MaterialAttributes(const MeshStore &meshes, uint32_t id, ops::Domain domain, std::span<const ops::Attribute> attributes) {
    const auto found = attributes | std::views::transform([&](const ops::Attribute &attribute) {
                           const auto held = meshes.FindAttribute(id, attribute.Name);
                           return held != InvalidOffset && meshes.GetAttribute(held).Type == ops::StorageOf(attribute.Type) ? held : InvalidOffset;
                       }) |
        std::ranges::to<std::vector>();
    return ops::AttributeTable(meshes, domain, found);
}

// The type of a Surface input's computed value, from its socket's types.
constexpr mx::Type SurfaceInputType(uint32_t socket) {
    const auto types = detail::SurfaceSockets[socket].Types;
    return types == detail::Float ? mx::Type::Float : types == detail::Normal3 ? mx::Type::Vector3 : mx::Type::Color3;
}
// A pattern's words hold three for each input whose socket takes more than a float, and one for each other.
static_assert(std::size(detail::SurfaceSockets) == uint32_t(PbrInput::Count));
static_assert([] {
    uint32_t triples = 0u;
    for (uint32_t s = 0; s < uint32_t(PbrInput::Count); ++s) triples |= uint32_t(detail::SurfaceSockets[s].Types != detail::Float) << s;
    return triples == PbrInputTriples;
}());

// The material's graph planned and laid out, its Surface node, and the linked Surface inputs by socket with the op output each reads, where the output has a value.
struct SurfaceLinks {
    CookPlan Plan;
    OpLayout Layout;
    uint32_t Surface;
    std::vector<std::pair<uint32_t, uint16_t>> Outputs;
    std::vector<uint16_t> Sockets;
};
SurfaceLinks LinksOf(const state::Registry &r, state::Entity material) {
    SurfaceLinks links{.Plan = Plan(r, material)};
    const auto &plan = links.Plan;
    links.Layout = LayoutOf(r, plan);
    links.Surface = IndexOf(plan, SurfaceNodeOf(r, material));
    for (uint16_t s = 0; links.Surface != NoNode && s < uint32_t(PbrInput::Count); ++s) {
        const auto &link = plan.Inputs[plan.InputFirst[links.Surface] + s];
        if (link.FromNode == NoNode || !IsInlineOp(r, plan, link.FromNode) || !OutputType(r, plan, links.Layout, link.FromNode, link.FromSocket)) continue;
        links.Outputs.emplace_back(link.FromNode, link.FromSocket);
        links.Sockets.push_back(s);
    }
    return links;
}
std::vector<uint32_t> FiltersOf(const state::Registry &r, const CookPlan &plan) {
    std::vector<uint32_t> filters;
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n)
        if (IsFilterOp(r, plan, n)) filters.push_back(n);
    return filters;
}

// Makes the graph's output `output` read its value converted to the type, by a convert node of plan node `node`.
void ConvertOutput(OpGraph &graph, size_t output, mx::Type type, uint32_t node) {
    graph.Graph.Nodes.push_back({.Op = std::string{"convert"}, .Output = type, .Inputs = {{"in", graph.Graph.Outputs[output]}}});
    graph.Nodes.push_back(node);
    graph.Graph.Outputs[output] = {uint32_t(graph.Graph.Nodes.size() - 1u), 0u};
}
// The graph computing the outputs for the Surface sockets, each converted to its input's type.
OpGraph SurfaceGraph(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, std::span<const std::pair<uint32_t, uint16_t>> outputs, std::span<const uint16_t> sockets, uint32_t surface) {
    auto graph = GraphOf(r, plan, layout, outputs, FiltersOf(r, plan));
    for (size_t o = 0; o < outputs.size(); ++o) {
        const auto target = SurfaceInputType(sockets[o]);
        if (*OutputType(r, plan, layout, outputs[o].first, outputs[o].second) != target) ConvertOutput(graph, o, target, surface);
    }
    return graph;
}

// A material's linked Surface inputs compiled into one program, which links with the other materials' before the material keeps it.
struct CompiledMaterial {
    state::Entity Material;
    CookPlan Plan;
    OpGraph Graph;
    std::vector<uint16_t> Sockets;
    const ops::Compiled *Compiled;
};
// The settle pass writes a material's layout, faults and program directly, outside an action's Apply, as the cook does for a geometry procedure, since the graph derives them.
// Lays out the material's value block and compiles its linked Surface inputs, or drops its nodes' faults and removes its program where it links none.
std::optional<CompiledMaterial> CompileMaterial(state::Registry &r, state::Entity material) {
    auto [plan, layout, surface, outputs, sockets] = LinksOf(r, material);
    LayoutOpValues(r, material, plan, layout);
    if (outputs.empty()) {
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) Drop<NodeError>(r, PathOf(plan, n));
        r.remove<MaterialProgram>(material);
        return {};
    }
    auto graph = SurfaceGraph(r, plan, layout, outputs, sockets, surface);
    const auto &compiled = ops::CompileShared(ops::GetDrivers(r), graph.Graph, ops::Domain::Surface);
    return CompiledMaterial{material, std::move(plan), std::move(graph), std::move(sockets), &compiled};
}
// Keeps the linked program on the material, or its faults on the material's nodes, and drops the fault of each other node.
void KeepProgram(state::Registry &r, const CompiledMaterial &m) {
    const auto &program = m.Compiled->Program;
    std::vector<bool> faulted(m.Plan.Nodes.size());
    MaterialProgram kept{.Base = m.Graph.Base};
    if (!program) {
        for (const auto &fault : program.error()) faulted[m.Graph.Nodes[fault.Node]] = true;
        KeepFaults(r, m.Plan, m.Graph, program.error());
    } else {
        kept.Program = *m.Compiled->Index;
        kept.Needs = program->Needs;
        for (const auto s : m.Sockets) kept.Inputs |= 1u << s;
        for (const auto type : program->Outputs) kept.Words += mx::Words(type);
        for (const auto &image : m.Graph.Images) kept.Images.push_back(image.Texture);
        kept.Attributes = program->Attributes;
    }
    for (uint32_t n = 0; n < m.Plan.Nodes.size(); ++n) {
        if (!faulted[n]) Drop<NodeError>(r, PathOf(m.Plan, n));
    }
    r.emplace_or_replace<MaterialProgram>(m.Material, std::move(kept));
}
// Compiles the materials' graphs, links their programs in one batch, and keeps each program or its faults.
void CompileMaterials(state::Registry &r, std::span<const state::Entity> materials) {
    std::vector<CompiledMaterial> compiled;
    for (const auto material : materials)
        if (auto program = CompileMaterial(r, material)) compiled.push_back(std::move(*program));
    if (!compiled.empty()) ops::LinkCompiled(r, ops::GetDrivers(r));
    for (const auto &program : compiled) KeepProgram(r, program);
}

// Appends the pattern's image table of the textures, and its attribute directory where its program reads named attributes, to the words, at the word offsets the pattern takes.
// The directory holds the count of mesh records, the word of a table holding no attribute, then each record's table word, which is InvalidOffset for a record `drawn` lacks.
void AppendTables(const state::Registry &r, std::vector<uint32_t> &words, MaterialPattern &pattern, std::span<const state::Entity> textures, std::span<const ops::Attribute> attributes, std::span<const uint32_t> drawn) {
    pattern.Program.Images.Offset = uint32_t(words.size());
    for (const auto texture : textures) words.append_range(std::bit_cast<std::array<uint32_t, 2>>(DocumentImage(r, texture).Texture));
    if (attributes.empty()) return;
    const auto append = [&](std::span<const OpAttributeRef> table) {
        const auto at = uint32_t(words.size());
        words.append_range(std::span{reinterpret_cast<const uint32_t *>(table.data()), table.size() * sizeof(OpAttributeRef) / sizeof(uint32_t)});
        return at;
    };
    pattern.Program.Attributes.Offset = uint32_t(words.size());
    const uint32_t ids = drawn.empty() ? 0u : drawn.back() + 1u;
    words.push_back(ids);
    words.push_back(0u);
    words.resize(words.size() + ids, InvalidOffset);
    words[pattern.Program.Attributes.Offset + 1u] = append(std::vector<OpAttributeRef>(attributes.size()));
    const auto &meshes = r.Resources.get<const MeshStore>();
    for (const auto id : drawn) words[pattern.Program.Attributes.Offset + 2u + id] = append(MaterialAttributes(meshes, id, ops::Domain::Surface, attributes));
}

// Writes the pattern of each material by record index, then each pattern's image table and attribute directory, into a new records buffer, or releases it where no material computes inputs.
// Records that hold what the buffer holds leave it as it is.
// Returns whether the records changed.
bool WritePatterns(state::Registry &r) {
    auto &patterns = r.Resources.get<MaterialPatterns>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto programs = r.view<const MaterialProgram, const Material>();
    if (std::ranges::none_of(programs, [&](state::Entity e) { return r.get<const MaterialProgram>(e).Program.has_value(); })) {
        const bool held = bool(patterns.Records);
        if (held) Reset(patterns, r);
        return held;
    }
    const auto count = buffers.Materials.Count<PBRMaterial>();
    patterns.Count = count;
    patterns.Values.clear();
    patterns.Words = 0u;
    // A pattern reading named attributes reads, for the mesh drawing it, the attribute table resolved on that mesh's record, which each drawn record holds.
    std::vector<uint32_t> drawn;
    if (std::ranges::any_of(programs, [&](state::Entity e) { return !r.get<const MaterialProgram>(e).Attributes.empty(); })) {
        for (const auto e : r.view<const EvaluatedMesh>())
            if (const auto id = DrawnStoreId(r, e)) drawn.push_back(*id);
        for (const auto e : r.view<const MeshHandle>())
            if (const auto id = DrawnStoreId(r, e)) drawn.push_back(*id);
        SortUnique(drawn);
    }
    std::vector<MaterialPattern> records(count);
    std::vector<uint32_t> words(count * sizeof(MaterialPattern) / sizeof(uint32_t)); // The records, then their tables
    for (const auto [material, program, record] : programs.each()) {
        if (!program.Program || record.Index >= count) continue;
        const auto *block = r.try_get<const OpValueBlock>(material);
        auto &pattern = records[record.Index] = {
            .Program = {.Index = *program.Program, .Needs = program.Needs, .Values = {buffers.OpValues.Buffer.Slot, (block ? block->Block.Offset : 0u) + program.Base}},
            .Inputs = program.Inputs,
            .Words = program.Words,
        };
        AppendTables(r, words, pattern, program.Images, program.Attributes, drawn);
        if (block) patterns.Values.push_back(block->Block);
        patterns.Words = std::max(patterns.Words, program.Words);
    }
    std::memcpy(words.data(), records.data(), count * sizeof(MaterialPattern));
    // Each record's image table and attribute directory are in the records buffer.
    const auto set_slots = [&](std::span<MaterialPattern> records, uint32_t slot) {
        for (auto &pattern : records) {
            pattern.Program.Images.Slot = slot;
            if (pattern.Program.Attributes.Offset) pattern.Program.Attributes.Slot = slot;
        }
    };
    if (patterns.Records) {
        set_slots({reinterpret_cast<MaterialPattern *>(words.data()), count}, patterns.Records->Slot);
        if (std::ranges::equal(std::as_bytes(std::span{words}), patterns.Records->Contents().first(patterns.Records->UsedSize))) return false;
    }
    patterns.Records = std::make_unique<mtl::Buffer>(buffers.Ctx, std::as_bytes(std::span{words}), SlotType::Buffer);
    set_slots(patterns.Records->GetMutableSpan<MaterialPattern>({0u, count}), patterns.Records->Slot);
    return true;
}
// The text of a document texture in a MaterialX document: its source file's path, or its name for a texture without one.
std::string TextureFile(const state::Registry &r, state::Entity texture) {
    const auto *materialized = r.valid(texture) ? r.try_get<const MaterializedTexture>(texture) : nullptr;
    if (!materialized) return {};
    const auto *sources = r.try_get<const gltf::SourceAssets>(state::Document);
    if (sources && materialized->SourceImageIndex < sources->Images.size()) {
        const auto &image = sources->Images[materialized->SourceImageIndex];
        if (image.Source == gltf::Image::SourceKind::External) return project::ResolveAsset(r, image.SourcePath).string();
    }
    return materialized->Params.Name;
}

// The value of a word input of plan node `n`'s op: the constant of the port the input reads, or else the node's literal, or none for an input the op leaves out.
std::optional<std::vector<uint32_t>> WordValue(const state::Registry &r, const CookPlan &plan, uint32_t n, std::string_view input, mx::Type type) {
    const auto &authored = Authored(r, plan.Nodes[n]);
    const auto &op = authored.get<const Op>(plan.Nodes[n].Entity);
    const auto socket = InputSocket(*DefinitionOf(op.Node), input);
    const auto *constant = socket ? &plan.Inputs[plan.InputFirst[n] + *socket].Constant : nullptr;
    if (constant && *constant) return std::vector<uint32_t>{(*constant)->Words.begin(), (*constant)->Words.begin() + mx::Words(type)};
    if (!op.Literals.contains(input)) return {};
    return std::vector<uint32_t>{std::from_range, LiteralWords(authored, plan.Nodes[n].Entity, input, type)};
}

// Whether every texture coordinate node of the graph reads the first set, whose coordinates a texel of the unit square holds.
bool ReadsFirstTexcoords(const state::Registry &r, const CookPlan &plan, const OpGraph &graph) {
    const auto &lib = mx::Libraries();
    for (uint32_t i = 0; i < graph.Graph.Nodes.size(); ++i) {
        const auto &node = graph.Graph.Nodes[i];
        const auto *declaration = std::get_if<uint32_t>(&node.Op);
        if ((declaration ? std::string_view{lib.Declarations[*declaration].Node} : std::string_view{std::get<std::string>(node.Op)}) != "texcoord") continue;
        const auto index = std::ranges::find(node.Inputs, "index", &ops::Input::Name);
        if (index == node.Inputs.end()) continue;
        const auto *word = std::get_if<ops::Word>(&index->Source);
        if (!word) return false;
        // An index the op leaves out reads the first set.
        const auto value = WordValue(r, plan, graph.Nodes[i], "index", word->Type);
        if (value && (value->empty() || value->front() != 0u)) return false;
    }
    return true;
}

uint32_t PortOf(const mx::Declaration &d, std::string_view name) { return uint32_t(std::ranges::find(d.Inputs, name, &mx::Port::Name) - d.Inputs.begin()); }
} // namespace

std::expected<mx::MaterialDocument, std::string> MaterialXOf(const state::Registry &r, state::Entity material) {
    const auto &lib = mx::Libraries();
    const auto &name = r.get<const Material>(material).Name;
    const auto links = LinksOf(r, material);
    const auto &plan = links.Plan;
    const auto graph = GraphOf(r, plan, links.Layout, links.Outputs, FiltersOf(r, plan));
    mx::MaterialDocument document{.Name = name, .Graph = {.Name = "NG_" + name}};
    auto &nodes = document.Graph.Nodes;
    const auto output_type = [&](const mx::NodeOutput &from) { return lib.Declarations[nodes[from.Node].Declaration].Outputs[from.Output].Type; };
    // A value of another type reads through a convert node.
    const auto converted = [&](mx::Source source, mx::Type from, mx::Type to, std::string node_name) -> mx::Source {
        if (from == to) return source;
        nodes.push_back({.Name = std::move(node_name), .Declaration = *mx::FindDeclaration(lib, "convert", std::array{mx::TypedInput{"in", from}}, to), .Inputs = {{0u, std::move(source)}}});
        return mx::NodeOutput{uint32_t(nodes.size() - 1u), 0u};
    };
    std::vector<uint32_t> index(graph.Graph.Nodes.size());
    for (uint32_t i = 0; i < graph.Graph.Nodes.size(); ++i) {
        const auto &node = graph.Graph.Nodes[i];
        const auto n = graph.Nodes[i];
        const auto &authored = Authored(r, plan.Nodes[n]);
        const auto &op = authored.get<const Op>(plan.Nodes[n].Entity);
        // Each input's value or link, and its type.
        std::vector<std::tuple<std::string, mx::Source, mx::Type>> sources;
        for (const auto &input : node.Inputs) {
            if (const auto *link = std::get_if<ops::Link>(&input.Source)) {
                const mx::NodeOutput from{index[link->Node], link->Output};
                sources.emplace_back(input.Name, from, output_type(from));
            } else if (const auto *image = std::get_if<ops::Image>(&input.Source)) {
                sources.emplace_back(input.Name, mx::Value{TextureFile(r, graph.Images[image->Slot].Texture)}, mx::Type::Filename);
            } else if (const auto *word = std::get_if<ops::Word>(&input.Source)) {
                // A field read's default and an input the op leaves out take the declaration's default.
                const auto value = std::ranges::contains(graph.Reads, i, &OpRead::Node) ? std::nullopt : WordValue(r, plan, n, input.Name, word->Type);
                if (value) sources.emplace_back(input.Name, mx::Value{*value}, word->Type);
            } else {
                const auto &literal = std::get<ops::Literal>(input.Source);
                sources.emplace_back(input.Name, literal.Value, literal.Type);
            }
        }
        // A node the graph names by its MaterialX node resolves from its inputs' types.
        const auto declaration = [&]() -> std::optional<uint32_t> {
            if (const auto *resolved = std::get_if<uint32_t>(&node.Op)) return *resolved;
            const auto typed = sources | std::views::transform([](const auto &source) { return mx::TypedInput{std::get<0>(source), std::get<2>(source)}; }) | std::ranges::to<std::vector>();
            const auto resolution = ops::Resolve(lib, std::get<std::string>(node.Op), typed, node.Output);
            return resolution ? std::optional{resolution->Declaration} : std::nullopt;
        }();
        if (!declaration) return std::unexpected{std::format("Node '{}' of material '{}' has no MaterialX declaration for its inputs", op.Node, name)};
        const auto &d = lib.Declarations[*declaration];
        mx::GraphNode written{.Name = std::format("{}{}", d.Node, i), .Declaration = *declaration};
        for (auto &[input, source, type] : sources) {
            const auto port = PortOf(d, input);
            const auto to = d.Inputs[port].Type;
            const bool value = to == mx::Type::String || to == mx::Type::Filename;
            written.Inputs.push_back({port, value ? std::move(source) : converted(std::move(source), type, to, std::format("convert{}_{}", i, input))});
        }
        index[i] = uint32_t(nodes.size());
        nodes.push_back(std::move(written));
    }
    std::vector<std::optional<uint32_t>> computed(std::size(detail::SurfaceSockets));
    for (size_t o = 0; o < links.Sockets.size(); ++o) {
        const auto socket = links.Sockets[o];
        const auto &from = graph.Graph.Outputs[o];
        const mx::NodeOutput output{index[from.Node], from.Output};
        const std::string input{detail::SurfaceSockets[socket].Name};
        document.Graph.Outputs.push_back(converted(output, output_type(output), SurfaceInputType(socket), "convert_" + input));
        document.Outputs.emplace_back(input, SurfaceInputType(socket));
        computed[socket] = uint32_t(o);
    }
    const auto &record = r.Resources.get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>()[r.get<const Material>(material).Index];
    const auto words = [](auto... values) { return mx::Value{std::vector<uint32_t>{std::bit_cast<uint32_t>(values)...}}; };
    const auto color = [&](vec3 c) { return words(c.x, c.y, c.z); };
    // Each gltf_pbr input a Surface input computes reads its graph output, and the material's values give the others.
    const auto input = [&](std::string_view name, mx::Type type, mx::Value value) {
        if (!std::holds_alternative<std::string>(value) || !std::get<std::string>(value).empty()) document.Surface.push_back({.Name = std::string{name}, .Type = type, .Source = std::move(value)});
    };
    const auto surface = [&](PbrInput pbr, mx::Value value) {
        const auto s = uint32_t(pbr);
        const auto &socket = detail::SurfaceSockets[s];
        if (computed[s]) document.Surface.push_back({.Name = socket.Name, .Type = SurfaceInputType(s), .Source = *computed[s], .TangentNormal = socket.Types == detail::Normal3});
        else input(socket.Name, SurfaceInputType(s), std::move(value));
    };
    surface(PbrInput::BaseColor, color(vec3{record.BaseColorFactor}));
    surface(PbrInput::Metallic, words(record.MetallicFactor));
    surface(PbrInput::Roughness, words(record.RoughnessFactor));
    surface(PbrInput::Normal, mx::Value{std::string{}});
    surface(PbrInput::Occlusion, mx::Value{std::string{}});
    input("transmission", mx::Type::Float, words(record.Transmission.Factor));
    surface(PbrInput::Specular, words(record.Specular.Factor));
    surface(PbrInput::SpecularColor, color(record.Specular.ColorFactor));
    input("ior", mx::Type::Float, words(record.Ior));
    input("alpha", mx::Type::Float, words(record.BaseColorFactor.w));
    input("alpha_mode", mx::Type::Integer, mx::Value{std::vector<uint32_t>{uint32_t(record.AlphaMode)}});
    input("alpha_cutoff", mx::Type::Float, words(record.AlphaCutoff));
    surface(PbrInput::Iridescence, words(record.Iridescence.Factor));
    input("iridescence_ior", mx::Type::Float, words(record.Iridescence.Ior));
    surface(PbrInput::IridescenceThickness, words(record.Iridescence.ThicknessMaximum));
    surface(PbrInput::SheenColor, color(record.Sheen.ColorFactor));
    surface(PbrInput::SheenRoughness, words(record.Sheen.RoughnessFactor));
    surface(PbrInput::Clearcoat, words(record.Clearcoat.Factor));
    surface(PbrInput::ClearcoatRoughness, words(record.Clearcoat.RoughnessFactor));
    surface(PbrInput::ClearcoatNormal, mx::Value{std::string{}});
    surface(PbrInput::Emissive, color(record.EmissiveFactor));
    input("emissive_strength", mx::Type::Float, words(record.EmissiveStrength));
    surface(PbrInput::Thickness, words(record.Volume.ThicknessFactor));
    // A zero attenuation distance is infinite, which the shader's default gives.
    if (record.Volume.AttenuationDistance > 0.f) input("attenuation_distance", mx::Type::Float, words(record.Volume.AttenuationDistance));
    input("attenuation_color", mx::Type::Color3, color(record.Volume.AttenuationColor));
    surface(PbrInput::AnisotropyStrength, words(record.Anisotropy.Strength));
    surface(PbrInput::AnisotropyRotation, words(record.Anisotropy.Rotation));
    return document;
}

std::expected<std::string, std::string> WriteMaterialX(const state::Registry &r, state::Entity material) {
    const auto document = MaterialXOf(r, material);
    if (!document) return std::unexpected{document.error()};
    const auto &lib = mx::Libraries();
    const auto declarations = document->Graph.Nodes | std::views::transform(&mx::GraphNode::Declaration) | std::ranges::to<std::vector>();
    const auto definitions = mx::OwnDefinitions(lib, declarations);
    if (!definitions) return std::unexpected{std::format("Material '{}': {}", document->Name, definitions.error())};
    return mx::WriteMaterial(lib, *document, *definitions);
}

MaterialBake BakeMaterialInputs(state::Registry &r, uint32_t size) {
    MaterialBake bake;
    if (r.view<const MaterialProgram>().empty()) return bake;
    auto &meshes = r.Resources.get<MeshStore>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &drivers = ops::GetDrivers(r);
    const auto materials = MaterialsByIndex(r);
    const auto frame = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
    const float time = animation::FrameSeconds(r, frame);
    // Submits the program `record(chain, images, output_slot)` records for the graph's input `socket`, and reads back the texels it wrote as a baked input.
    const auto bake_input = [&](uint16_t socket, const OpGraph &graph, state::Entity mesh, auto &&record) {
        const auto components = mx::Words(SurfaceInputType(socket));
        const auto stride = ops::TexelStride(components);
        const mtl::Buffer output{buffers.Ctx, uint64_t(size) * size * stride * sizeof(float), SlotType::Buffer, mtl::BufferLifetime::Workspace};
        const auto images = graph.Images | std::views::transform([&](const OpImage &image) { return DocumentImage(r, image.Texture).Texture; }) | std::ranges::to<std::vector>();
        mtl::ComputeChain chain{meshes.BufferContext()};
        record(chain, images, output.Slot);
        chain.Submit();
        const auto texels = output.GetSpan<float>({0u, size * size * stride});
        gltf::BakedInput input{.Input = PbrInput(socket), .Mesh = mesh, .Width = size, .Height = size, .Components = components};
        input.Texels.reserve(size_t(size) * size * components);
        for (uint32_t t = 0; t < size * size; ++t) input.Texels.append_range(texels.subspan(size_t(t) * stride, components));
        return std::make_shared<const gltf::BakedInput>(std::move(input));
    };
    // A material's inputs that bake: the graph of each, its program over the unit square where its values depend on no mesh, and that program's bake.
    struct BakingInput {
        uint16_t Socket;
        OpGraph Graph;
        const ops::Compiled *Square;
        std::shared_ptr<const gltf::BakedInput> Shared;
    };
    struct Baking {
        std::vector<BakingInput> Inputs;
        uint32_t Values; // The word of the op value store its programs read their values from
    };
    // Each input bakes from a program of its own.
    std::map<state::Entity, Baking> bakings;
    const auto compile = [&](state::Entity material) {
        const auto [held, added] = bakings.try_emplace(material);
        if (!added) return;
        auto &baking = held->second;
        const auto [plan, layout, surface, outputs, sockets] = LinksOf(r, material);
        const auto *block = r.try_get<const OpValueBlock>(material);
        baking.Values = block ? block->Block.Offset : 0u;
        for (size_t o = 0; o < outputs.size(); ++o) {
            auto graph = SurfaceGraph(r, plan, layout, std::span{outputs}.subspan(o, 1u), std::span{sockets}.subspan(o, 1u), surface);
            const auto &compiled = ops::CompileShared(drivers, graph.Graph, ops::Domain::Texel);
            if (!compiled.Program) {
                const auto input = detail::SurfaceSockets[sockets[o]].Name;
                bake.Unbaked.push_back(std::format("Material '{}' computes its {} from a node that does not evaluate on texels, so the {} exports as the material's value", r.get<const Material>(material).Name, input, input));
                continue;
            }
            // An input that reads nothing of a mesh but its first texture coordinates bakes once over the unit square.
            constexpr uint32_t MeshFields = OpNeedsPosition | OpNeedsNormal | OpNeedsTangent | OpNeedsColor | OpNeedsIndex;
            const bool independent = !(compiled.Program->Needs & MeshFields) && compiled.Program->Attributes.empty() && ReadsFirstTexcoords(r, plan, graph);
            const auto *square = independent ? &ops::CompileShared(drivers, graph.Graph, ops::Domain::Uv) : nullptr;
            baking.Inputs.push_back({sockets[o], std::move(graph), square, {}});
        }
        const auto computed = baking.Inputs | std::views::transform([](const BakingInput &input) { return PbrInput(input.Socket); }) | std::ranges::to<std::vector>();
        if (auto conflict = gltf::BakeConflict(r, material, computed)) {
            bake.Unbaked.push_back(std::format("{}, so the material exports with its values", *conflict));
            baking.Inputs.clear();
        }
    };
    std::vector<state::Entity> mesh_entities{std::from_range, r.view<const EvaluatedMesh>()};
    for (const auto [_, owned] : r.view<const ProcedureOwnedMesh>().each()) {
        if (r.valid(owned.MeshEntity)) mesh_entities.push_back(owned.MeshEntity);
    }
    SortUnique(mesh_entities);
    // Calls `visit(mesh_entity, record id, material)` for each material that computes inputs on each mesh that draws triangles with it.
    const auto each_drawn = [&](auto &&visit) {
        for (const auto mesh_entity : mesh_entities) {
            const auto id = GetMesh(r, mesh_entity).GetStoreId();
            const auto &record = meshes.Get(id);
            if (!record.TriangleCount) continue;
            std::vector<uint32_t> indices{std::from_range, meshes.Arenas().PrimitiveMaterials.Get(record.PrimitiveMaterials)};
            SortUnique(indices);
            for (const auto index : indices) {
                const auto material = index < materials.size() ? materials[index] : state::Null;
                const auto *program = material != state::Null ? r.try_get<const MaterialProgram>(material) : nullptr;
                if (program && program->Program) visit(mesh_entity, id, material);
            }
        }
    };
    // The programs of every drawn material link in one batch, then each input that bakes over the unit square bakes once.
    each_drawn([&](state::Entity, uint32_t, state::Entity material) { compile(material); });
    ops::LinkCompiled(r, drivers);
    for (auto &[material, baking] : bakings) {
        for (auto &input : baking.Inputs) {
            if (input.Square && input.Square->Program) input.Shared = bake_input(input.Socket, input.Graph, state::Null, [&](auto &chain, const auto &images, auto output) {
                ops::RecordGrid(chain, drivers, *input.Square, {buffers.OpValues.Buffer.Slot, baking.Values + input.Graph.Base}, images, {size, size, 0u}, {output, 0u}, float(frame), time);
            });
        }
    }
    each_drawn([&](state::Entity mesh_entity, uint32_t id, state::Entity material) {
        const auto &baking = bakings.at(material);
        if (baking.Inputs.empty()) return;
        // A texture reads a mesh's first texture coordinates, which a mesh without them lacks.
        if (!meshes.Get(id).CornerUvs[0].Count) {
            bake.Unbaked.push_back(std::format("Mesh '{}' has no texture coordinates, so it exports material '{}' with its values", GetName(r, mesh_entity), r.get<const Material>(material).Name));
            return;
        }
        auto &material_baked = bake.Materials.emplace_back(gltf::BakedMaterial{.Material = material, .Mesh = mesh_entity});
        for (const auto &input : baking.Inputs) {
            if (input.Shared) {
                material_baked.Inputs.push_back(input.Shared);
                continue;
            }
            const auto &compiled = ops::CompileShared(drivers, input.Graph.Graph, ops::Domain::Texel);
            material_baked.Inputs.push_back(bake_input(input.Socket, input.Graph, mesh_entity, [&](auto &chain, const auto &images, auto output) {
                ops::RecordTexels(chain, drivers, meshes, id, compiled, {buffers.OpValues.Buffer.Slot, baking.Values + input.Graph.Base}, MaterialAttributes(meshes, id, ops::Domain::Texel, compiled.Program->Attributes), images, {0u, size, size, size}, {output, 0u}, float(frame), time);
            }));
        }
    });
    return bake;
}

void RegisterMaterialGraphHandlers(state::Registry &r) {
    using state::On;
    r.Resources.emplace<MaterialPatterns>();
    auto &changes = r.Resources.emplace<MaterialGraphChanges>();
    changes.Changed.bind(r)
        .on<MaterialGraph>(On::Create | On::Update | On::Destroy)
        .on<ProcedureNode>(On::Create | On::Update)
        .on<NetworkNodes>(On::Create | On::Update)
        .on<Subnet>(On::Update)
        .on<SubnetInterface>(On::Create | On::Update)
        .on<NodeInputs>(On::Update)
        .on<Op>(On::Update)
        .on<OpValueBlock>(On::Create | On::Update)
        .on<Material>(On::Create | On::Update | On::Destroy)
        .on<MaterializedTexture>(On::Create | On::Update | On::Destroy);
    changes.Meshes.bind(r).on<MeshGeometryDirty>(On::Create | On::Update).on<EvaluatedMesh>(On::Create | On::Update).on<MeshHandle>(On::Create | On::Update).on<MeshMaterialAssignment>(On::Create | On::Update);
}

void NoteMaterialValues(state::Registry &r) { ++r.Resources.get<MaterialGraphChanges>().Revision; }

void SettleMaterialGraphs(state::Registry &r) {
    auto &all_changes = r.Resources.get<MaterialGraphChanges>();
    auto &changes = all_changes.Changed;
    // A mesh's change reaches the attribute tables of the patterns that read named attributes.
    const bool attributes = !all_changes.Meshes.empty() && std::ranges::any_of(r.view<const MaterialProgram>(), [&](state::Entity e) { return !r.get<const MaterialProgram>(e).Attributes.empty(); });
    all_changes.Meshes.clear();
    // A graph, a material record or an attribute table a preview shows changed.
    if (!changes.empty() || attributes || AnyChanged(r.Resources.get<const RenderChanges>().Materials)) ++all_changes.Revision;
    if (changes.empty() && !attributes) return;
    std::vector<state::Entity> materials;
    bool patterns = r.Resources.get<const MaterialPatterns>().Records || !r.view<const MaterialProgram>().empty();
    bool definitions = false;
    for (const auto e : changes) {
        const auto *node = r.valid(e) ? r.try_get<const ProcedureNode>(e) : nullptr;
        const auto network = node ? node->Network : e;
        definitions |= r.valid(network) && r.all_of<SubnetDefinition>(network);
        if (r.valid(network) && r.all_of<MaterialGraph>(network)) materials.push_back(network);
        // A material whose graph is gone computes nothing.
        else if (r.valid(e) && r.all_of<MaterialProgram>(e)) r.remove<MaterialProgram>(e);
    }
    // A change inside a definition reaches every graph that may instance it.
    if (definitions) materials.append_range(r.view<const MaterialGraph>());
    SortUnique(materials);
    CompileMaterials(r, materials);
    patterns |= !materials.empty() || attributes;
    changes.clear();
    if (!patterns) return;
    if (WritePatterns(r)) r.Resources.get<RenderChanges>().Materials.emplace(state::Document);
}

void ResetPrograms(state::Registry &r) {
    r.Resources.erase<ops::Drivers>();
    r.Resources.erase<mtl::ProgramRegistry>();
    auto &patterns = r.Resources.get<MaterialPatterns>();
    patterns.Pipelines.reset();
    patterns.Previews.reset();
    std::vector<state::Entity> materials{std::from_range, r.view<const MaterialGraph>()};
    SortUnique(materials);
    CompileMaterials(r, materials);
    if (WritePatterns(r)) r.Resources.get<RenderChanges>().Materials.emplace(state::Document);
}

std::optional<ImageTextures> MaterialPreviewOf(state::Registry &r, state::Entity material, std::span<const state::Entity> path) {
    auto &changes = r.Resources.get<MaterialGraphChanges>();
    std::erase_if(changes.Previews, [](const auto &preview) { return preview.Image.expired(); });
    const auto held = std::ranges::find_if(changes.Previews, [&](const auto &preview) { return preview.Material == material && std::ranges::equal(preview.Path, path); });
    const auto texture_of = [](std::shared_ptr<const PreviewImage> image) {
        return ImageTextures{.Image = *image->Display, .Shown = *image->Display, .Width = image->Display.Extent.Width, .Height = image->Display.Extent.Height, .Hold = std::move(image)};
    };
    if (held != changes.Previews.end() && held->Revision == changes.Revision) return texture_of(held->Image.lock());
    const auto viewports = r.view<const Viewport>();
    if (viewports.empty() || !r.valid(material) || !r.all_of<Material>(material)) return {};
    constexpr mtl::Extent2D Extent{256u, 256u};
    const auto view = PreviewView(r, r.get<const Viewport>(*viewports.begin()).SceneViewUBO.GetSpan<SceneViewUBO>()[0], Extent);
    const auto index = r.get<const Material>(material).Index;
    std::optional<PreviewValue> value;
    std::optional<mtl::Buffer> tables;
    if (!path.empty()) {
        // The node's first output, evaluated by a program of its own at the sphere's points.
        const auto plan = Plan(r, material);
        const auto node = IndexOf(plan, path);
        if (node == NoNode || !IsInlineOp(r, plan, node)) return {};
        const auto layout = LayoutOf(r, plan);
        const auto socket = uint16_t(DefinitionOf(Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity).Node)->Inputs);
        const auto type = OutputType(r, plan, layout, node, socket);
        if (!type || mx::Words(*type) == 0u || mx::Words(*type) > 4u) return {};
        auto graph = GraphOf(r, plan, layout, std::array{std::pair{node, socket}}, FiltersOf(r, plan));
        const bool vector = *type == mx::Type::Vector2 || *type == mx::Type::Vector3 || *type == mx::Type::Vector4;
        // An integer or boolean shows as the float it converts to.
        if (*type == mx::Type::Integer || *type == mx::Type::Boolean) ConvertOutput(graph, 0u, mx::Type::Float, node);
        auto &drivers = ops::GetDrivers(r);
        const auto &compiled = ops::CompileShared(drivers, graph.Graph, ops::Domain::Surface);
        ops::LinkCompiled(r, drivers);
        if (!compiled.Program) return {};
        auto &buffers = r.Resources.get<GpuBuffers>();
        const auto *block = r.try_get<const OpValueBlock>(material);
        MaterialPattern pattern{
            .Program = {.Index = *compiled.Index, .Needs = compiled.Program->Needs, .Values = {buffers.OpValues.Buffer.Slot, (block ? block->Block.Offset : 0u) + graph.Base}},
            .Words = mx::Words(compiled.Program->Outputs.front()),
        };
        // The program's tables, where an attribute directory of no mesh record reads its table holding no attribute.
        std::vector<uint32_t> words;
        AppendTables(r, words, pattern, graph.Images | std::views::transform(&OpImage::Texture) | std::ranges::to<std::vector>(), compiled.Program->Attributes, {});
        if (!words.empty()) {
            tables.emplace(buffers.Ctx, std::as_bytes(std::span{words}), SlotType::Buffer, mtl::BufferLifetime::Workspace);
            pattern.Program.Images.Slot = tables->Slot;
            if (!compiled.Program->Attributes.empty()) pattern.Program.Attributes.Slot = tables->Slot;
        }
        value = PreviewValue{.Program = pattern, .Display = vector ? PreviewDisplay::Vector : PreviewDisplay::Color};
    }
    auto image = RenderMaterialPreview(r, view, Extent, index, value);
    if (held != changes.Previews.end()) *held = {material, NodePath{path.begin(), path.end()}, changes.Revision, image};
    else changes.Previews.push_back({material, NodePath{path.begin(), path.end()}, changes.Revision, image});
    return texture_of(std::move(image));
}
} // namespace procedure
