#include "Paths.h"
#include "RunSuites.h"
#include "TestMeshOrdinals.h"
#include "TopologyCases.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshCreate.h"
#include "mesh/Primitives.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/SpatialFaceWork.h"
#include "mesh/Unsubdivide.h"
#include "state/Registry.h"
#include <numbers>
#include <print>

using namespace boost::ut;

namespace {
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCase, ::test::TopologyCases;

struct GeometryFixture {
    NS::SharedPtr<NS::AutoreleasePool> Pool{NS::TransferPtr(NS::AutoreleasePool::alloc()->init())};
    mtl::Context Context;
    mtl::BindlessSet Slots{Context};
    mtl::BufferContext Buffers{Context, Slots};
    state::Registry Scene;
    GeometryFixture() {
        Scene.Resources.emplace<mtl::LibraryCache>(Context, Paths::Shaders());
        Scene.Resources.emplace<MeshStore>(Buffers);
    }
    MeshStore &Store() { return Scene.Resources.get<MeshStore>(); }
    Mesh Get(uint32_t id) { return Mesh{Store(), id}; }
    void Audit() {
        std::string why;
        const bool valid = Store().Audit(why);
        if (!valid) std::println("audit: {}", why);
        expect(valid);
    }
    uint32_t Create(MeshData data) {
        const auto id = CreateMesh(Scene, {.Data = std::move(data)}).StoreId;
        ShadeNewMeshes(Scene);
        Audit();
        return id;
    }
    GeometryTopologyResult Run(MeshTopologyTask task) {
        auto result = ExecuteGeometryTopology(Scene, std::span{&task, 1u}).front();
        Audit();
        return result;
    }
};

MeshData Quad() {
    return MeshData{{{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}}, {{0u, 1u, 2u, 3u}}};
}

// Writes selections as mask blocks into an arena the test owns, one block run per domain with room for the blocks an edit inserts.
struct MaskWriter {
    BufferArena<SelectionBlock> Blocks;
    explicit MaskWriter(mtl::BufferContext &buffers) : Blocks{buffers, SlotType::Buffer} {}
    SelectionMasks Write(std::span<const uint32_t> handles, uint32_t capacity) {
        const auto range = Blocks.Allocate((capacity + MeshElementBlockSize - 1u) / MeshElementBlockSize + 64u);
        auto blocks = Blocks.GetMutable(range);
        std::ranges::fill(blocks, SelectionBlock{});
        for (const auto h : handles) blocks[h / MeshElementBlockSize][(h % MeshElementBlockSize) / 32u] |= 1u << (h % 32u);
        return {&Blocks, range.Offset};
    }
    // Masks mirror arena blocks, so each handle of record `id` sets the bit at its arena address.
    GeometryMasks Write(const MeshStore &meshes, uint32_t id, const GeometrySelection &selection) {
        const auto &a = meshes.Arenas();
        const auto addresses = [&](MeshStore::ElementDomain domain, std::span<const uint32_t> handles) {
            std::vector<uint32_t> result;
            for (const auto h : handles) result.push_back(meshes.HandleAddress(id, domain, ElementHandle{h}).Value());
            return result;
        };
        using Domain = MeshStore::ElementDomain;
        return {Write(addresses(Domain::Vertex, selection.Vertices), a.Vertices.Capacity()), Write(addresses(Domain::Edge, selection.Edges), a.EdgeHalfedges.Capacity()), Write(addresses(Domain::Face, selection.Faces), a.FaceTriangles.Capacity())};
    }
};

// Positions rounded so that results compare across meshes whose handles differ.
using Point = std::array<int64_t, 3>;
Point Key(vec3 p) { return {std::llround(p.x * 1e4f), std::llround(p.y * 1e4f), std::llround(p.z * 1e4f)}; }
std::vector<Point> Points(const Mesh &mesh) {
    std::vector<Point> points;
    for (const auto v : mesh.vertices()) points.push_back(Key(mesh.GetPosition(v)));
    return points;
}
// Each element's sorted corner points, sorted.
std::vector<std::vector<Point>> Shapes(const Mesh &mesh, Element element, std::span<const uint32_t> handles) {
    std::vector<std::vector<Point>> shapes;
    for (const auto handle : handles) {
        auto &shape = shapes.emplace_back();
        if (element == Element::Vertex) shape.push_back(Key(mesh.GetPosition(he::VH{handle})));
        else if (element == Element::Edge) {
            const auto h = mesh.GetHalfedge(he::EH{handle}, 0u);
            shape = {Key(mesh.GetPosition(mesh.GetFromVertex(h))), Key(mesh.GetPosition(mesh.GetToVertex(h)))};
        } else
            for (const auto v : mesh.fv_range(he::FH{handle})) shape.push_back(Key(mesh.GetPosition(v)));
        std::ranges::sort(shape);
    }
    std::ranges::sort(shapes);
    return shapes;
}
struct Outcome {
    bool Changed, Copied;
    std::array<std::vector<std::vector<Point>>, 9> Elements;
    bool operator==(const Outcome &) const = default;
};
Outcome Describe(const MeshStore &meshes, uint32_t source, const GeometryTopologyResult &result) {
    const Mesh mesh{meshes, result.GeometryId};
    Outcome outcome{result.Changed, result.GeometryId != source};
    const auto all = AllHandles(mesh);
    for (uint32_t d = 0u; const auto element : {Element::Vertex, Element::Edge, Element::Face}) {
        outcome.Elements[d] = Shapes(mesh, element, result.Created.Get(element));
        outcome.Elements[3u + d] = Shapes(mesh, element, result.Retained.Get(element));
        outcome.Elements[6u + d++] = Shapes(mesh, element, all.Get(element));
    }
    return outcome;
}

// Named attributes of every type on every domain, each following fixed channels the gathers carry by the same rule, so outputs check against those channels.
// Encode vertex identity in morph targets and skin, and corner identity in UVs and blue channel.
// Encode face identity in primitive and sharpness; add 1000 to sharp-edge identifiers.
constexpr std::array AttributeDomains{AttributeDomain::Vertex, AttributeDomain::Edge, AttributeDomain::Face, AttributeDomain::Corner};
constexpr std::array AttributeKinds{AttributeType::Float, AttributeType::Vec3, AttributeType::Int, AttributeType::Bool};
constexpr std::array<std::array<std::string_view, 4>, 4> AttributeNames{{
    {"vertex_float", "vertex_vec3", "vertex_int", "vertex_bool"},
    {"edge_float", "edge_vec3", "edge_int", "edge_bool"},
    {"face_float", "face_vec3", "face_int", "face_bool"},
    {"corner_float", "corner_vec3", "corner_int", "corner_bool"},
}};
constexpr std::array Blues{1.f, 1000.f, 1000000.f};

AttributeValue Bits(AttributeScalar auto value) {
    AttributeValue bits{};
    std::memcpy(bits.data(), &value, sizeof(value));
    return bits;
}

vec3 FaceVector(int32_t p) { return {float(p), -float(p), 2.f * float(p)}; }
float FaceFloat(int32_t p) { return 1.5f * float(p) + .25f; }
vec3 EdgeVector(int32_t v) { return {float(v), -float(v), .5f * float(v)}; }
float EdgeFloat(int32_t v) { return .5f * float(v); }

template<AttributeScalar T> void Fill(MeshStore &meshes, uint32_t id, std::string_view name, uint32_t count, auto &&value) {
    const auto values = std::make_unique<T[]>(count);
    for (uint32_t i = 0u; i < count; ++i) values[i] = value(i);
    meshes.SetAttribute<T>(meshes.FindAttribute(id, name), 0u, std::span<const T>{values.get(), count});
}

// The shape with a morph target, skin, UVs, colors, a primitive per face and every third face and edge sharp, and the named attributes over them.
// Returns the mesh and its edge integers.
std::pair<uint32_t, std::set<int32_t>> CreateAttributed(GeometryFixture &f, MeshData data) {
    const auto vertex_count = uint32_t(data.Positions.size()), face_count = data.FaceCount();
    MeshSource source{.Data = std::move(data)};
    ArmatureDeformData deform;
    MorphTargetData morph{.TargetCount = 1u, .DefaultWeights = {0.f}};
    std::vector<vec2> uvs;
    std::vector<vec4> colors;
    for (uint32_t i = 0u; i < vertex_count; ++i) {
        uvs.push_back({.5f * float(i), 1.f - .25f * float(i)});
        colors.push_back({.1f * float(i), 2.f - .3f * float(i), Blues[i % 3u], 1.f});
        deform.Joints.push_back({i * 3u % 17u, i % 2u, 0u, 0u});
        deform.Weights.push_back({1.f, 0.f, 0.f, 0.f});
        morph.PositionDeltas.push_back({.25f * float(i), .5f * float(i) + .25f, -.25f * float(i)});
        morph.NormalDeltas.push_back({3.f * float(i) - 7.f, 0.f, 0.f});
    }
    source.Attrs.TexCoords0 = std::move(uvs);
    source.Attrs.Colors0 = std::move(colors);
    source.Attrs.Colors0ComponentCount = 4u;
    std::vector<uint32_t> primitives(face_count);
    std::iota(primitives.begin(), primitives.end(), 0u);
    source.Primitives = {.ElementPrimitiveIndices = std::move(primitives), .MaterialIndices = std::vector<uint32_t>(face_count, 0u)};
    source.Deform = std::move(deform);
    source.Morph = std::move(morph);
    const auto id = CreateMesh(f.Scene, std::move(source)).StoreId;
    ShadeNewMeshes(f.Scene);
    auto &meshes = f.Store();
    {
        const auto mesh = f.Get(id);
        GeometrySelection faces, edges;
        for (const auto face : mesh.faces())
            if (mesh.FaceOrdinal(face) % 3u == 0u) faces.Faces.push_back(*face);
        for (uint32_t e = 0u; e < mesh.EdgeCount(); e += 3u) edges.Edges.push_back(*mesh.EdgeAt(e));
        mtl::ComputeChain chain{meshes.BufferContext()};
        const SharpnessOperationTarget sharp_faces{id, faces}, sharp_edges{id, edges};
        ExecuteGeometrySharpness(f.Scene, chain, std::span{&sharp_faces, 1u}, EditSharpnessOperation::SetSelectedFaces, true);
        ExecuteGeometrySharpness(f.Scene, chain, std::span{&sharp_edges, 1u}, EditSharpnessOperation::SetSelectedEdges, true);
    }
    f.Audit();
    const std::array<std::array<AttributeValue, 4>, 4> defaults{{
        {},
        {Bits(EdgeFloat(-1)), Bits(EdgeVector(-1)), Bits(int32_t{-1}), {}},
        {Bits(FaceFloat(0)), Bits(FaceVector(0)), {}, {}},
        {Bits(1.f), Bits(vec3{0.f, 0.f, 1.f}), Bits(int32_t{1}), {}},
    }};
    for (uint32_t d = 0u; d < 4u; ++d)
        for (uint32_t k = 0u; k < 4u; ++k) expect(fatal(meshes.AddAttribute(id, AttributeDomains[d], AttributeKinds[k], AttributeNames[d][k], defaults[d][k]).has_value()));
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    // A dense domain's addresses run from its range's offset while its handles run from zero.
    const auto vertices = a.Vertices.Dense(record.Vertices), edges = a.EdgeHalfedges.Dense(record.EdgeData), faces = a.FaceTriangles.Dense(record.FaceData), corners = a.FaceCorners.Dense(record.FaceCorners);
    Fill<float>(meshes, id, "vertex_float", vertices.Count, [&](uint32_t v) { return a.Morph.Get(record.Morph, v).NormalDelta.x; });
    Fill<vec3>(meshes, id, "vertex_vec3", vertices.Count, [&](uint32_t v) { return a.Morph.Get(record.Morph, v).PositionDelta; });
    Fill<int32_t>(meshes, id, "vertex_int", vertices.Count, [&](uint32_t v) { return int32_t(a.Skin.Get(record.Skin, v).Joints.x); });
    Fill<bool>(meshes, id, "vertex_bool", vertices.Count, [&](uint32_t v) { return a.Skin.Get(record.Skin, v).Joints.y != 0u; });
    const auto edge_value = [&](uint32_t e) { return int32_t(e) + (a.EdgeSharpness.Get({edges.Offset + e, 1u})[0] ? 1000 : 0); };
    std::set<int32_t> edge_values;
    for (uint32_t e = 0u; e < edges.Count; ++e) edge_values.insert(edge_value(e));
    Fill<float>(meshes, id, "edge_float", edges.Count, [&](uint32_t e) { return EdgeFloat(edge_value(e)); });
    Fill<vec3>(meshes, id, "edge_vec3", edges.Count, [&](uint32_t e) { return EdgeVector(edge_value(e)); });
    Fill<int32_t>(meshes, id, "edge_int", edges.Count, edge_value);
    Fill<bool>(meshes, id, "edge_bool", edges.Count, [&](uint32_t e) { return a.EdgeSharpness.Get({edges.Offset + e, 1u})[0] != 0u; });
    const auto primitive = [&](uint32_t face) { return int32_t(a.FacePrimitives.Get(record.FacePrimitives, face)); };
    Fill<float>(meshes, id, "face_float", faces.Count, [&](uint32_t face) { return FaceFloat(primitive(face)); });
    Fill<vec3>(meshes, id, "face_vec3", faces.Count, [&](uint32_t face) { return FaceVector(primitive(face)); });
    Fill<int32_t>(meshes, id, "face_int", faces.Count, primitive);
    Fill<bool>(meshes, id, "face_bool", faces.Count, [&](uint32_t face) { return a.FaceSharpness.Get({faces.Offset + face, 1u})[0] != 0u; });
    const auto uv = [&](uint32_t h) { return a.CornerUvs[0].Get(record.CornerUvs[0], h); };
    const auto color = [&](uint32_t h) { return a.CornerColors.Get(record.CornerColors, h); };
    Fill<float>(meshes, id, "corner_float", corners.Count, [&](uint32_t h) { return color(h).y; });
    Fill<vec3>(meshes, id, "corner_vec3", corners.Count, [&](uint32_t h) { return vec3{uv(h).x, uv(h).y, color(h).x}; });
    Fill<int32_t>(meshes, id, "corner_int", corners.Count, [&](uint32_t h) { return int32_t(color(h).z); });
    Fill<bool>(meshes, id, "corner_bool", corners.Count, [&](uint32_t h) { return color(h).z >= 1000.f; });
    return {id, std::move(edge_values)};
}

// Checks every element of the record against the fixed channels its attributes follow.
// A copied output renumbers its primitives, so its face integers only come from the source.
void CheckAttributed(const MeshStore &meshes, uint32_t id, const std::set<int32_t> &edge_values, uint32_t face_count, bool copied) {
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    // The attributes of a domain in AttributeKinds order.
    const auto views = [&](uint32_t d) {
        const auto find = [&](uint32_t k) { return meshes.FindAttribute(id, AttributeNames[d][k]); };
        return std::tuple{meshes.AttributeView<float>(find(0u)), meshes.AttributeView<vec3>(find(1u)), meshes.AttributeView<int32_t>(find(2u)), meshes.AttributeView<bool>(find(3u))};
    };
    const auto [vf, vv, vi, vb] = views(0u);
    a.Vertices.ForEach(record.Vertices, [&](ElementHandle handle, ElementAddress) {
        const auto v = handle.Value;
        const auto morph = a.Morph.Get(record.Morph, v, 0u, record.MorphTargetCount);
        const auto skin = a.Skin.Get(record.Skin, v);
        expect(vf[v] == morph.NormalDelta.x);
        expect(vv[v] == morph.PositionDelta);
        expect(vi[v] == int32_t(skin.Joints.x));
        expect(vb[v] == (skin.Joints.y != 0u));
    });
    const auto [ef, ev, ei, eb] = views(1u);
    a.EdgeHalfedges.ForEach(record.EdgeData, [&](ElementHandle handle, ElementAddress address) {
        const auto e = handle.Value;
        const auto value = ei[e];
        const bool sharp = a.EdgeSharpness.Get({address.Value(), 1u})[0] != 0u;
        expect(edge_values.contains(value) || value == -1) << "edge" << e << "holds" << value;
        expect((value >= 1000) == sharp);
        expect(eb[e] == sharp);
        expect(ef[e] == EdgeFloat(value));
        expect(ev[e] == EdgeVector(value));
    });
    const auto [ff, fv, fi, fb] = views(2u);
    a.FaceTriangles.ForEach(record.FaceData, [&](ElementHandle handle, ElementAddress address) {
        const auto face = handle.Value;
        const auto value = fi[face];
        if (copied) expect(value >= 0 && uint32_t(value) < face_count);
        else expect(value == int32_t(a.FacePrimitives.Get(record.FacePrimitives, face)));
        expect(ff[face] == FaceFloat(value));
        expect(fv[face] == FaceVector(value));
        expect(fb[face] == (a.FaceSharpness.Get({address.Value(), 1u})[0] != 0u));
    });
    const auto [cf, cv, ci, cb] = views(3u);
    const auto blue = [](float value) { return std::ranges::contains(Blues, value); };
    a.FaceCorners.ForEach(record.FaceCorners, [&](ElementHandle handle, ElementAddress) {
        const auto h = handle.Value;
        const auto uv = a.CornerUvs[0].Get(record.CornerUvs[0], h);
        const auto color = a.CornerColors.Get(record.CornerColors, h);
        expect(cf[h] == color.y);
        expect(cv[h] == vec3{uv.x, uv.y, color.x});
        // Blue mixes where corner sources differ, and the integer takes the dominant source.
        expect(blue(float(ci[h])));
        if (blue(color.z)) expect(ci[h] == int32_t(color.z));
        expect(cb[h] == (ci[h] >= 1000));
    });
}
uint32_t Place(GeometryFixture &f, const auto &shape, vec3 offset) {
    auto data = primitive::CreateMesh(shape);
    for (auto &p : data.Positions) p += offset;
    return f.Create(std::move(data));
}
uint32_t Box(GeometryFixture &f, vec3 center, vec3 half = vec3{1.f}) { return Place(f, primitive::Cube{.HalfExtents = half}, center); }
std::optional<uint32_t> RunBoolean(GeometryFixture &f, uint32_t first, uint32_t second, BooleanOperation operation) {
    const auto task = BooleanTask(f.Get(first), f.Get(second), operation, mat3{1.f}, vec3{0.f});
    const auto id = task ? ExecuteGeometryBoolean(f.Scene, *task) : std::nullopt;
    f.Audit();
    return id;
}
bool ClosedManifold(const Mesh &mesh) {
    if (!mesh.HasClosedSurface()) return false;
    const auto &c = mesh.GetConnectivity();
    for (const auto face : mesh.faces())
        for (const auto h : mesh.fh_range(face)) {
            const auto opposite = c.Opposite(h);
            if (!opposite || c.Opposite(opposite) != h || !c.FaceOf(opposite) || c.FaceOf(opposite) == face) return false;
        }
    return true;
}
int64_t EulerCharacteristic(const Mesh &mesh) { return int64_t(mesh.VertexCount()) - int64_t(mesh.EdgeCount()) + int64_t(mesh.FaceCount()); }
bool Solid(const Mesh &mesh, double volume) { return ClosedManifold(mesh) && std::abs(::test::EnclosedVolume(mesh).value_or(-1.0) - volume) < 1e-4; }
} // namespace

void RegisterGeometryOperationsTests() {
    "headless topology uses explicit masks and returns created geometry"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selected = AllHandles(f.Get(id));
        expect(!f.Store().Get(id).MeshletData);
        for (const auto op : {MeshTopologyOp::DuplicateGeometry, MeshTopologyOp::ExtrudeRegion, MeshTopologyOp::DeleteLoose}) {
            const auto empty = f.Run({.SourceId = id, .Op = op});
            expect(!empty.Changed);
            expect(f.Get(id).VertexCount() == 4u);
            expect(f.Get(id).FaceCount() == 1u);
        }
        const auto duplicated = f.Run({.SourceId = id, .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies, .CopyRotation = mat3{1.f}, .CopyTranslation = {4.f, 0.f, 0.f}, .Selection = selected});
        expect(duplicated.Changed);
        expect(duplicated.GeometryId == id);
        expect(duplicated.Created.Vertices.size() == 4u);
        expect(duplicated.Created.Faces.size() == 1u);
        expect(duplicated.Retained.Vertices.size() == 4u);
        expect(f.Get(id).FaceCount() == 2u);
        const auto &bounds = f.Store().GetSelectionRoot(id, Element::Vertex).Bounds;
        expect(bounds.Min.x == -1.f);
        expect(bounds.Max.x == 5.f);
        expect(f.Store().GetBoundaryEdges(id).Count() == 8u);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        expect(!f.Store().Get(id).MeshletData);
        const auto extrude_id = f.Create(Quad());
        const auto extruded = f.Run({.SourceId = extrude_id, .Op = MeshTopologyOp::ExtrudeRegion, .Selection = AllHandles(f.Get(extrude_id))});
        expect(extruded.Changed);
        expect(extruded.Created.Vertices.size() == 4u);
        expect(f.Get(extrude_id).FaceCount() == 6u);
        expect(f.Get(extrude_id).HasClosedSurface());
        const auto extracted = f.Run({.SourceId = id, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = Element::Face, .Selection = GeometrySelection{.Faces = duplicated.Created.Faces}});
        expect(extracted.Changed);
        expect(extracted.GeometryId != id);
        expect(f.Get(extracted.GeometryId).VertexCount() == 4u);
        expect(f.Get(extracted.GeometryId).FaceCount() == 1u);
        expect(f.Store().GetBoundaryEdges(extracted.GeometryId).Count() == 4u);
        expect(f.Store().GetSelectionRoot(extracted.GeometryId, Element::Vertex).Bounds.Min.x == 3.f);
    };
    "headless loose deletion ignores document visibility and selection"_test = [] {
        GeometryFixture f;
        MeshData data{{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {3.f, 0.f, 0.f}}};
        data.Edges.push_back({0u, 1u});
        const auto id = f.Create(std::move(data));
        const auto selected = AllHandles(f.Get(id));
        const std::array blocks{f.Store().Arenas().Vertices.Entries(f.Store().Get(id).Vertices)[selected.Vertices.front() / MeshElementBlockSize]};
        f.Store().EditHiddenBlocks(Element::Vertex, blocks, [](uint32_t, auto &words) { std::ranges::fill(words, ~0u); });
        const auto removed = f.Run({.SourceId = id, .Op = MeshTopologyOp::DeleteLoose, .Selection = selected});
        expect(removed.Changed);
        expect(f.Get(id).VertexCount() == 0u);
        expect(f.Get(id).EdgeCount() == 0u);
    };
    "extracted geometry chains edits and preserves source attributes"_test = [] {
        GeometryFixture f;
        MeshVertexAttributes attributes;
        const std::array texcoords{vec2{.2f, .7f}, vec2{.3f, .6f}, vec2{.4f, .5f}, vec2{.5f, .4f}};
        attributes.TexCoords0 = std::vector<vec2>(texcoords.begin(), texcoords.end());
        attributes.Colors0 = std::vector<vec4>(4u, {.1f, .3f, .5f, 1.f});
        attributes.Colors0ComponentCount = 4u;
        const auto source = CreateMesh(f.Scene, {.Data = Quad(), .Attrs = std::move(attributes)}).StoreId;
        ShadeNewMeshes(f.Scene);
        const auto extracted = f.Run({.SourceId = source, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = Element::Face, .Selection = AllHandles(f.Get(source))});
        const auto output_id = extracted.GeometryId;
        const auto selection = extracted.Created;
        expect(extracted.Changed);
        expect(output_id != source);
        const auto original = f.Get(source).GetPosition(*f.Get(source).vertices().begin());
        const PositionOperationTarget target{.StoreId = output_id, .Selection = selection};
        const auto changes = ExecuteGeometryPositions(f.Scene, std::span{&target, 1u}, PositionEditOp::PushPull, .25f, 1u, {.Center = {0.f, 0.f, 2.f}});
        f.Audit();
        expect(changes.size() == 1u);
        expect(f.Get(source).GetPosition(*f.Get(source).vertices().begin()) == original);
        expect(f.Get(output_id).GetPosition(*f.Get(output_id).vertices().begin()).z > 0.f);
        {
            mtl::ComputeChain chain{f.Store().BufferContext()};
            const auto permuted = EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&target, 1u}, false, 0u, 0u);
            expect(permuted.size() == 1u);
            chain.Submit();
        }
        for (const auto corner : f.Get(source).fh_range(f.Get(source).FaceAt(0u)))
            expect(f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(f.Get(source).GetStoreId()).CornerUvs[0], *corner) == texcoords[f.Get(source).VertexOrdinal(f.Get(source).GetToVertex(corner))]);
        for (const auto corner : f.Get(output_id).fh_range(f.Get(output_id).FaceAt(0u)))
            expect(f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(f.Get(output_id).GetStoreId()).CornerUvs[0], *corner) == texcoords[(f.Get(output_id).VertexOrdinal(f.Get(output_id).GetToVertex(corner)) + 3u) % 4u]);
        const auto triangulated = f.Run({.SourceId = output_id, .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = selection});
        expect(triangulated.Changed);
        expect(triangulated.Created.Faces.size() == 1u);
        expect(triangulated.Retained.Faces.size() == 1u);
        expect(f.Get(output_id).FaceCount() == 2u);
        expect(f.Get(source).FaceCount() == 1u);
        for (const auto face : f.Get(output_id).faces())
            for (const auto corner : f.Get(output_id).fh_range(face)) {
                expect(f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(f.Get(output_id).GetStoreId()).CornerUvs[0], *corner) == texcoords[(f.Get(output_id).VertexOrdinal(f.Get(output_id).GetToVertex(corner)) + 3u) % 4u]);
                expect(f.Store().Arenas().CornerColors.Get(f.Store().Get(f.Get(output_id).GetStoreId()).CornerColors, *corner) == vec4{.1f, .3f, .5f, 1.f});
            }
        expect(!f.Store().Get(output_id).MeshletData);
        expect(f.Scene.Living.size() == 0u);
    };
    "headless affine transforms share editor frame math"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selection = AllHandles(f.Get(id));
        const auto first = selection.Vertices.front();
        const auto untouched = f.Get(id).GetPosition(he::VH{selection.Vertices.back()});
        const quat quarter_turn{.7071067811865475f, 0.f, 0.f, .7071067811865475f};
        const PositionOperationTarget target{.StoreId = id, .Selection = GeometrySelection{.Vertices = {first}}, .World = {.P = {10.f, 20.f, 3.f}, .R = quarter_turn, .S = {2.f, 3.f, 4.f}}};
        const PositionOperationOptions options{.Transform = PositionTransform{.Delta = {.P = {1.f, 2.f, 3.f}, .R = quarter_turn, .S = {2.f, .5f, 1.f}}, .Pivot = {10.f, 20.f, 3.f}}};
        const auto changed = ExecuteGeometryPositions(f.Scene, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, options);
        f.Audit();
        expect(changed.size() == 1u);
        const auto position = f.Get(id).GetPosition(he::VH{first});
        expect(std::abs(position.x - 4.f) < 1e-5f);
        expect(std::abs(position.y + 2.f / 3.f) < 1e-5f);
        expect(std::abs(position.z - .75f) < 1e-5f);
        expect(f.Get(id).GetPosition(he::VH{selection.Vertices.back()}) == untouched);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        const auto triangulated = f.Run({.SourceId = id, .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = selection});
        expect(triangulated.Changed);
        expect(f.Get(id).FaceCount() == 2u);
        expect(!f.Store().Get(id).MeshletData);
    };
    "geometry planners reject foreign masks and invalid planes before reading topology"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto mesh = f.Get(id);
        // Handles in a local block the mesh does not contain name no element of it.
        const auto selected = AllHandles(mesh);
        const GeometrySelection foreign{.Vertices = {MeshElementBlockSize}, .Edges = {MeshElementBlockSize}, .Faces = {MeshElementBlockSize}};
        const auto rejects = [](auto &&plan) {
            try {
                plan();
            } catch (const std::invalid_argument &) { return true; }
            return false;
        };
        expect(rejects([&] { FillTask(f.Store(), mesh, foreign); }));
        expect(rejects([&] { BeautifyFaceTask(mesh, foreign, false); }));
        expect(rejects([&] { UnsubdivideTask(mesh, foreign); }));
        expect(rejects([&] { DecimateTask(f.Store(), mesh, selected, .5f, foreign.Faces); }));
        expect(rejects([&] { BisectTasks(mesh, {}, {}, false, false); }));
        expect(rejects([&] { SymmetrizeTasks(mesh, 3u, false); }));
        expect(rejects([&] { KnifeTask(mesh, selected, mat4{1.f}, {}, {}, {}); }));
        expect(rejects([&] { MergeTask(f.Store(), mesh, selected, GeometryMergeMode::First, 0.f, foreign.Vertices.front()); }));
        expect(mesh.FaceCount() == 1u);
    };

    "headless sharpness uses explicit masks and refreshes canonical normals"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto all = AllHandles(f.Get(id));
        mtl::ComputeChain chain{f.Store().BufferContext()};
        const auto apply = [&](EditSharpnessOperation operation, GeometrySelection selection = {}, bool value = true) {
            const SharpnessOperationTarget target{id, std::move(selection)};
            auto changes = ExecuteGeometrySharpness(f.Scene, chain, std::span{&target, 1u}, operation, value, .1f);
            f.Audit();
            return changes;
        };
        expect(apply(EditSharpnessOperation::SetSelectedFaces).empty());
        expect(apply(EditSharpnessOperation::SetSelectedFaces, {.Faces = all.Faces}).size() == 1u);
        expect(f.Store().Arenas().FaceSharpness.Get({test::Address(f.Get(id), he::FH{all.Faces.front()}), 1u})[0] == 1u);
        // A local sharpness edit may retain mixed classification without scanning the mesh.
        expect(f.Store().GetCornerClassMode(id) != uint32_t(CornerClassMode::UniformVertex));
        expect((f.Store().GetSelectionRoot(id, Element::Face).Flags & SelectionLiveSharp) != 0u);
        apply(EditSharpnessOperation::SmoothAll);
        apply(EditSharpnessOperation::SetSelectedEdges, {.Edges = {all.Edges.front()}});
        expect(f.Store().Arenas().EdgeSharpness.Get({test::Address(f.Get(id), he::EH{all.Edges.front()}), 1u})[0] == 1u);
        apply(EditSharpnessOperation::SmoothAll);
        apply(EditSharpnessOperation::SetVertexEdges, {.Vertices = {all.Vertices[0], all.Vertices[1]}});
        uint32_t sharp = 0u;
        for (const auto edge : all.Edges) sharp += f.Store().Arenas().EdgeSharpness.Get({test::Address(f.Get(id), he::EH{edge}), 1u})[0] != 0u;
        expect(sharp == 3u);
        apply(EditSharpnessOperation::SetAllFaces);
        expect(f.Store().Arenas().FaceSharpness.Get({test::Address(f.Get(id), he::FH{all.Faces.front()}), 1u})[0] == 1u);
        apply(EditSharpnessOperation::SmoothByAngle);
        for (const auto edge : all.Edges) expect(f.Store().Arenas().EdgeSharpness.Get({test::Address(f.Get(id), he::EH{edge}), 1u})[0] == 0u);
        expect(f.Get(id).GetNormal(he::FH{all.Faces.front()}).z > .99f);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        expect(!f.Store().Get(id).MeshletData);
        expect(f.Scene.Living.empty());
    };
    "headless knife restricts canonical candidates without render caches"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selected = AllHandles(f.Get(id));
        const auto cut = f.Run({.SourceId = id, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagScreenCuts, .ScreenTransform = mat4{1.f}, .Extent = {100.f, 100.f}, .KnifeStart = {50.f, 0.f}, .KnifeEnd = {50.f, 100.f}, .SelectionElement = Element::Face, .Selection = GeometrySelection{.Faces = selected.Faces}});
        expect(cut.Changed);
        expect(f.Get(id).VertexCount() == 6u);
        expect(f.Get(id).FaceCount() == 2u);
        expect(cut.Created.Vertices.size() == 2u);
        expect(!f.Store().Get(id).MeshletData);
        MeshData separated;
        for (uint32_t face = 0u; face < 512u; ++face) {
            const auto first = uint32_t(separated.Positions.size());
            const float x = face < 256u ? 0.f : 100.f;
            separated.Positions.insert(separated.Positions.end(), {{x - 1.f, -1.f, 0.f}, {x + 1.f, -1.f, 0.f}, {x + 1.f, 1.f, 0.f}, {x - 1.f, 1.f, 0.f}});
            separated.AddFace(std::array{first, first + 1u, first + 2u, first + 3u});
        }
        const auto separated_id = f.Create(std::move(separated));
        for (const auto mode : {TopologyFlagPlaneCuts, TopologyFlagScreenCuts}) {
            const MeshTopologyTask query{.SourceId = separated_id, .Op = MeshTopologyOp::Subdivide, .Flags = mode | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}, .ScreenTransform = mat4{1.f}, .Extent = {100.f, 100.f}, .KnifeStart = {50.f, 0.f}, .KnifeEnd = {50.f, 100.f}};
            mtl::ComputeChain chain{f.Buffers};
            SpatialFaceWork work{f.Scene, chain, query};
            chain.Submit();
            expect(work.CandidateCount == 256u);
            expect(work.CandidateBlocks == 1u);
            work.RecordFaces(f.Scene, chain);
            chain.Submit();
            expect(work.Count == 256u);
        }
        auto moved = AllHandles(f.Get(separated_id));
        moved.Vertices.resize(1024u);
        const PositionOperationTarget moving{.StoreId = separated_id, .Selection = GeometrySelection{.Vertices = moved.Vertices}};
        ExecuteGeometryPositions(f.Scene, std::span{&moving, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {200.f, 0.f, 0.f}}}});
        f.Audit();
        const MeshTopologyTask query{.SourceId = separated_id, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}};
        mtl::ComputeChain chain{f.Buffers};
        SpatialFaceWork moved_work{f.Scene, chain, query};
        chain.Submit();
        expect(moved_work.CandidateCount == 0u);
        expect(f.Store().GetSelectionRoot(separated_id, Element::Face).Bounds.Min.x == 99.f);
        for (const auto input : {id, f.Create(Quad())}) {
            const auto before = f.Get(input).FaceCount();
            const auto stages = BisectTasks(f.Get(input), vec3{}, vec3{1.f, 0.f, 0.f}, true, false);
            expect(throws<std::invalid_argument>([&] { ExecuteGeometryTopology(f.Scene, stages); }));
            expect(f.Get(input).FaceCount() == before);
            const auto staged = ExecuteGeometryTopologyStages(f.Scene, stages);
            f.Audit();
            expect(staged.size() == stages.size());
            expect(staged.back().Changed);
            expect(f.Get(input).FaceCount() == 1u);
            for (const auto v : f.Get(input).vertices()) expect(f.Get(input).GetPosition(v).x >= 0.f);
        }
        const auto through_vertex = f.Create(MeshData{{{-1.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, -1.f, 0.f}}, {{0u, 1u, 2u}}});
        const MeshTopologyTask vertex_query{.SourceId = through_vertex, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}};
        SpatialFaceWork vertex_work{f.Scene, chain, vertex_query};
        chain.Submit();
        vertex_work.RecordFaces(f.Scene, chain);
        chain.Submit();
        expect(vertex_work.Count == 1u);
    };
    "topology operators read mask selections like handle lists"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        for (const auto &c : TopologyCases) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto listed = PlanCase(c, f.Store(), f.Get(listed_id));
                auto masked = PlanCase(c, f.Store(), f.Get(masked_id));
                expect(fatal(listed.has_value() && masked.has_value()));
                masked->Selection = masks.Write(f.Store(), masked_id, std::get<GeometrySelection>(masked->Selection));
                const auto by_list = Describe(f.Store(), listed_id, f.Run(*listed));
                const auto by_masks = Describe(f.Store(), masked_id, f.Run(*masked));
                expect(by_list.Changed);
                expect(by_list == by_masks);
            };
        }
    };
    "named attributes carry through topology operators on every domain"_test = [] {
        using enum MeshTopologyOp;
        const std::array<TopologyCase, 7> cases{{
            {"ExtrudeRegion", Op(ExtrudeRegion)},
            {"Subdivide", Op(Subdivide, 1.f)},
            {"DuplicateGeometry", Op(DuplicateGeometry)},
            {"DeleteFaces", Op(DeleteFaces)},
            {"InsetRegion", Op(InsetRegion, .1f)},
            {"MergeByDistance", {}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(meshes, mesh, selection, GeometryMergeMode::ByDistance, .1f, InvalidOffset); }},
            {"KeepSelectedFaces", Op(KeepSelectedFaces, 0.f, 0u, 1u, Element::Face)},
        }};
        GeometryFixture f;
        // Runs the task planned over the attributed shape and checks every element before and after against the channels.
        const auto carry = [&](std::string_view name, MeshData shape, auto &&plan) {
            boost::ut::test(std::string{name}) = [&] {
                const auto [id, edge_values] = CreateAttributed(f, std::move(shape));
                const auto face_count = f.Get(id).FaceCount();
                CheckAttributed(f.Store(), id, edge_values, face_count, false);
                const auto task = plan(id);
                expect(fatal(task.has_value()));
                const auto result = f.Run(*task);
                expect(result.Changed);
                CheckAttributed(f.Store(), result.GeometryId, edge_values, face_count, result.GeometryId != id);
            };
        };
        for (const auto &c : cases) carry(c.Name, Patchwork(), [&](uint32_t id) { return PlanCase(c, f.Store(), f.Get(id)); });
        // The union keeps the attributed box's elements with their channels, and the plain box's elements take the defaults matching its absent channels.
        carry("Boolean", primitive::CreateMesh(primitive::Cube{}), [&](uint32_t id) {
            const auto second = Box(f, vec3{1.f});
            return BooleanTask(f.Get(id), f.Get(second), BooleanOperation::Union, mat3{1.f}, vec3{0.f});
        });
    };
    "named attributes clone independently and release their payloads"_test = [] {
        GeometryFixture f;
        auto &meshes = f.Store();
        const auto id = f.Create(Quad());
        expect(meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "") == std::unexpected{MeshStore::AttributeError::InvalidName});
        expect(meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, std::string(64u, 'x')) == std::unexpected{MeshStore::AttributeError::InvalidName});
        const auto weight = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "weight");
        expect(fatal(weight.has_value()));
        expect(meshes.AddAttribute(id, AttributeDomain::Face, AttributeType::Int, "weight") == std::unexpected{MeshStore::AttributeError::Duplicate});
        const std::array values{1.f, 2.f, 3.f, 4.f};
        meshes.SetAttribute<float>(*weight, 0u, values);
        CloneCopies copies;
        const auto clone = meshes.CloneMeshes(copies, std::array{id}, true).front();
        {
            mtl::ComputeChain chain{meshes.BufferContext()};
            copies.Encode(chain, GetMeshPipelines(f.Scene));
            chain.Submit();
        }
        f.Audit();
        const auto copy = meshes.FindAttribute(clone, "weight");
        expect(fatal(copy != InvalidOffset && copy != *weight));
        for (uint32_t i = 0u; i < 4u; ++i) expect(meshes.AttributeView<float>(copy)[i] == values[i]);
        const std::array changed{9.f, 9.f, 9.f, 9.f};
        meshes.SetAttribute<float>(copy, 0u, changed);
        for (uint32_t i = 0u; i < 4u; ++i) {
            expect(meshes.AttributeView<float>(*weight)[i] == values[i]);
            expect(meshes.AttributeView<float>(copy)[i] == 9.f);
        }
        // A released range is free, so the next attribute takes it.
        const auto released = meshes.GetAttribute(copy).Values;
        expect(released.Count == 1u);
        meshes.Release(clone);
        f.Audit();
        const auto again = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "again");
        expect(fatal(again.has_value()));
        expect(meshes.GetAttribute(*again).Values.Offset == released.Offset);
        meshes.ReleaseAttribute(*again);
        expect(meshes.FindAttribute(id, "again") == InvalidOffset);
        expect(meshes.FindAttribute(id, "weight") == *weight);
    };
    "topology writes emitted selection into explicit destination masks"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        const auto id = f.Create(Quad());
        const auto output = masks.Write(f.Store(), id, {});
        const auto duplicated = f.Run({.SourceId = id, .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies, .CopyRotation = mat3{1.f}, .CopyTranslation = {4.f, 0.f, 0.f}, .Selection = AllHandles(f.Get(id)), .SelectionOutput = output});
        expect(duplicated.Changed);
        for (const auto element : {Element::Vertex, Element::Edge, Element::Face})
            expect(ReadMasks(f.Store(), id, element, output.Get(element)) == duplicated.Created.Get(element));
        for (const auto element : {Element::Vertex, Element::Edge, Element::Face})
            expect(f.Store().GetSelectedElements(id, element).Count() == 0u);
    };
    "position and sharpness operations read mask selections like handle lists"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        struct PositionCase {
            std::string_view Name;
            PositionEditOp Op;
            float Factor;
            PositionOperationOptions Options;
        };
        const std::array<PositionCase, 6> positions{{
            {"Smooth", PositionEditOp::Smooth, .5f, {}},
            {"PushPull", PositionEditOp::PushPull, .25f, {.Center = {0.f, 0.f, 2.f}}},
            {"Flatten", PositionEditOp::Copy, 1.f, {}},
            {"Circularize", PositionEditOp::Copy, 1.f, {.Circle = PositionCircle{}}},
            {"ShrinkFatten", PositionEditOp::ShrinkFatten, .1f, {.Flags = PositionEditSelectedFaceNormals}},
            {"VertexSlide", PositionEditOp::VertexSlide, .5f, {.Slide = PositionSlide{}}},
        }};
        // Exclusions name the triangulated cell's faces.
        const auto excluded = [](const Mesh &mesh) {
            GeometrySelection result;
            for (const auto face : mesh.faces())
                if (mesh.GetValence(face) == 3u) result.Faces.push_back(*face);
            return result;
        };
        for (const auto &c : positions) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto listed_selection = Region(f.Get(listed_id)), masked_selection = Region(f.Get(masked_id));
                const PositionOperationTarget listed{.StoreId = listed_id, .Selection = listed_selection, .Excluded = excluded(f.Get(listed_id)), .Reference = listed_selection.Vertices[4]};
                const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), masked_id, masked_selection), .Excluded = masks.Write(f.Store(), masked_id, excluded(f.Get(masked_id))), .Reference = masked_selection.Vertices[4]};
                const auto before = Points(f.Get(listed_id));
                const auto listed_changes = ExecuteGeometryPositions(f.Scene, std::span{&listed, 1u}, c.Op, c.Factor, 1u, c.Options);
                const auto masked_changes = ExecuteGeometryPositions(f.Scene, std::span{&masked, 1u}, c.Op, c.Factor, 1u, c.Options);
                f.Audit();
                expect(listed_changes.size() == masked_changes.size());
                expect(!std::ranges::equal(Points(f.Get(listed_id)), before));
                expect(std::ranges::equal(Points(f.Get(listed_id)), Points(f.Get(masked_id))));
            };
        }
        "face attributes"_test = [&] {
            const auto uvs = [] {
                MeshVertexAttributes attributes;
                std::vector<vec2> texcoords;
                for (uint32_t i = 0u; i < 28u; ++i) texcoords.push_back({float(i) / 28.f, 1.f - float(i) / 28.f});
                attributes.TexCoords0 = std::move(texcoords);
                return attributes;
            };
            const auto listed_id = CreateMesh(f.Scene, {.Data = Patchwork(), .Attrs = uvs()}).StoreId, masked_id = CreateMesh(f.Scene, {.Data = Patchwork(), .Attrs = uvs()}).StoreId;
            ShadeNewMeshes(f.Scene);
            const PositionOperationTarget listed{.StoreId = listed_id, .Selection = Region(f.Get(listed_id))};
            const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), masked_id, Region(f.Get(masked_id)))};
            mtl::ComputeChain chain{f.Store().BufferContext()};
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&listed, 1u}, false, 0u, 1u).size() == 1u);
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&masked, 1u}, false, 0u, 1u).size() == 1u);
            chain.Submit();
            f.Audit();
            const auto corner_uvs = [&](uint32_t id) {
                std::vector<vec2> result;
                for (const auto face : f.Get(id).faces())
                    for (const auto corner : f.Get(id).fh_range(face)) result.push_back(f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(f.Get(id).GetStoreId()).CornerUvs[0], *corner));
                return result;
            };
            expect(std::ranges::equal(corner_uvs(listed_id), corner_uvs(masked_id)));
        };
        for (const auto operation : {EditSharpnessOperation::SetSelectedFaces, EditSharpnessOperation::SetSelectedEdges, EditSharpnessOperation::SetVertexEdges, EditSharpnessOperation::SetAllFaces, EditSharpnessOperation::SmoothByAngle}) {
            boost::ut::test("sharpness " + std::to_string(uint32_t(operation))) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const SharpnessOperationTarget listed{listed_id, Region(f.Get(listed_id))};
                const SharpnessOperationTarget masked{masked_id, masks.Write(f.Store(), masked_id, Region(f.Get(masked_id)))};
                mtl::ComputeChain chain{f.Store().BufferContext()};
                const auto listed_changes = ExecuteGeometrySharpness(f.Scene, chain, std::span{&listed, 1u}, operation, true, .1f);
                const auto masked_changes = ExecuteGeometrySharpness(f.Scene, chain, std::span{&masked, 1u}, operation, true, .1f);
                f.Audit();
                expect(listed_changes.size() == 1u);
                expect(masked_changes.size() == 1u);
                const auto sharpness = [&](uint32_t id) {
                    std::vector<uint8_t> result;
                    for (const auto face : f.Get(id).faces()) result.push_back(f.Store().Arenas().FaceSharpness.Get({test::Address(f.Get(id), face), 1u})[0]);
                    for (const auto edge : f.Get(id).edges()) result.push_back(f.Store().Arenas().EdgeSharpness.Get({test::Address(f.Get(id), edge), 1u})[0]);
                    return result;
                };
                expect(std::ranges::equal(sharpness(listed_id), sharpness(masked_id)));
            };
        }
    };
    "booleans of boxes and a sphere yield the expected solids and keep their inputs"_test = [] {
        // Expected boolean counts and volume; nullopt skips an assertion.
        // A row without an Euler characteristic expects no task, since an open input plans nothing, and every planned task yields a mesh.
        struct BooleanCase {
            std::string_view Name;
            uint32_t (*First)(GeometryFixture &), (*Second)(GeometryFixture &);
            BooleanOperation Operation;
            std::optional<uint32_t> Vertices{}, Faces{};
            std::optional<double> Volume{};
            std::optional<int64_t> Euler{};
            // The share of the second input's measured volume added to the expected volume.
            double SecondShare{0.0};
        };
        const auto cube = [](GeometryFixture &f) { return Box(f, vec3{0.f}); };
        const auto overlapping = [](GeometryFixture &f) { return Box(f, vec3{1.f}); };
        // A sphere centered on the cube's +x face, whose plane halves it.
        const auto sphere = [](GeometryFixture &f) { return Place(f, primitive::UVSphere{.Radius = .75f}, vec3{1.f, 0.f, 0.f}); };
        // Boxes flush with the cube's +x face share the whole face or part of it, where naive booleans fail since faces coincide.
        const auto flush = [](GeometryFixture &f) { return Box(f, vec3{2.f, 0.f, 0.f}); };
        const auto offset = [](GeometryFixture &f) { return Box(f, vec3{2.f, .5f, .5f}); };
        const auto far = [](GeometryFixture &f) { return Box(f, vec3{5.f, 0.f, 0.f}); };
        const auto inner = [](GeometryFixture &f) { return Box(f, vec3{0.f}, vec3{.5f}); };
        const auto quad = [](GeometryFixture &f) { return f.Create(Quad()); };
        using enum BooleanOperation;
        const std::array<BooleanCase, 23> cases{{
            {"overlapping union", cube, overlapping, Union, 20u, 36u, 15.0, 2},
            {"overlapping difference", cube, overlapping, Difference, 14u, 24u, 7.0, 2},
            {"overlapping intersection", cube, overlapping, Intersection, 8u, 12u, 1.0, 2},
            {"sphere union", cube, sphere, Union, {}, {}, 8.0, 2, .5},
            {"sphere difference", cube, sphere, Difference, {}, {}, 8.0, 2, -.5},
            {"sphere intersection", cube, sphere, Intersection, {}, {}, 0.0, 2, .5},
            {"flush union", cube, flush, Union, 12u, 20u, 16.0, 2},
            {"flush difference", cube, flush, Difference, 8u, 12u, 8.0, 2},
            // Solids that only touch share no volume.
            {"flush intersection", cube, flush, Intersection, 0u, 0u, {}, 0},
            {"offset union", cube, offset, Union, {}, {}, 16.0, 2},
            {"offset difference", cube, offset, Difference, {}, {}, 8.0, 2},
            {"offset intersection", cube, offset, Intersection, 0u, 0u, {}, 0},
            {"disjoint union", cube, far, Union, 16u, 24u, 16.0, 4},
            {"disjoint difference", cube, far, Difference, 8u, 12u, 8.0, 2},
            {"disjoint intersection", cube, far, Intersection, 0u, 0u, {}, 0},
            {"nested union", cube, inner, Union, 8u, 12u, 8.0, 2},
            // A box with a box-shaped cavity, two shells facing apart.
            {"nested difference", cube, inner, Difference, 16u, 24u, 7.0, 4},
            {"nested intersection", cube, inner, Intersection, 8u, 12u, 1.0, 2},
            {"inner minus outer", inner, cube, Difference, 0u, 0u, {}, 0},
            {"open second", cube, quad, Union},
            {"open first", quad, cube, Union},
            {"open difference", cube, quad, Difference},
            {"open intersection", quad, cube, Intersection},
        }};
        GeometryFixture f;
        for (const auto &c : cases) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto first = c.First(f), second = c.Second(f);
                const auto first_vertices = f.Get(first).VertexCount(), second_vertices = f.Get(second).VertexCount();
                const auto task = BooleanTask(f.Get(first), f.Get(second), c.Operation, mat3{1.f}, vec3{0.f});
                expect(task.has_value() == c.Euler.has_value());
                const auto id = task ? ExecuteGeometryBoolean(f.Scene, *task) : std::nullopt;
                f.Audit();
                expect(id.has_value() == task.has_value());
                expect(f.Get(first).VertexCount() == first_vertices);
                expect(f.Get(second).VertexCount() == second_vertices);
                if (!id || !c.Euler) return;
                const auto mesh = f.Get(*id);
                if (c.Vertices) expect(mesh.VertexCount() == *c.Vertices);
                if (c.Faces) expect(mesh.FaceCount() == *c.Faces);
                if (c.Volume) expect(Solid(mesh, *c.Volume + c.SecondShare * ::test::EnclosedVolume(f.Get(second)).value_or(-1.0)));
                expect(EulerCharacteristic(mesh) == *c.Euler);
            };
        }
        "named attribute"_test = [&] {
            const auto a = cube(f), b = overlapping(f);
            auto &meshes = f.Store();
            const auto weight = meshes.AddAttribute(a, AttributeDomain::Face, AttributeType::Float, "weight", AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u}).value();
            for (const auto face : f.Get(a).faces()) {
                const float value = 5.f;
                meshes.SetAttribute<float>(weight, *face, std::span{&value, 1u});
            }
            const auto id = RunBoolean(f, a, b, Union).value();
            const auto output = meshes.FindAttribute(id, "weight");
            expect(output != InvalidOffset);
            // Each box keeps all of its surface but the three unit squares inside the other, the first's faces holding its value and the second's the default.
            const auto mesh = f.Get(id);
            const auto view = meshes.AttributeView<float>(output);
            std::array<double, 2> areas{};
            for (const auto face : mesh.faces()) {
                std::vector<vec3> corners;
                for (const auto v : mesh.fv_range(face)) corners.push_back(mesh.GetPosition(v));
                const float area = .5f * Length(Cross(corners[1] - corners[0], corners[2] - corners[0]));
                expect(view[*face] == 5.f || view[*face] == -1.f);
                areas[view[*face] == 5.f ? 0u : 1u] += area;
            }
            expect(std::abs(areas[0] - 21.0) < 1e-4);
            expect(std::abs(areas[1] - 21.0) < 1e-4);
        };
    };
    "degenerate dissolve merges a run of coincident vertices in one pass"_test = [] {
        GeometryFixture f;
        // Vertices 0, 1 and 2 coincide, joined only through 2, so 1 reaches 0 through 2 alone.
        const auto id = f.Create(MeshData{{{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}, {{0u, 2u, 3u}, {1u, 2u, 4u}}});
        f.Run({.SourceId = id, .Op = MeshTopologyOp::DissolveDegenerate, .Param0 = 1e-6f, .Flags = TopologyFlagSelectAll});
        expect(f.Get(id).VertexCount() == 3u);
        expect(f.Get(id).FaceCount() == 0u);
    };
    "boolean of two 100k triangle spheres"_test = [] {
        GeometryFixture f;
        const primitive::UVSphere sphere{.Radius = 1.f, .Slices = 320u, .Stacks = 160u};
        const auto a = Place(f, sphere, vec3{0.f}), b = Place(f, sphere, vec3{.5f, .3f, .2f});
        expect(f.Get(a).DerivedTriangles().size() > 100000u);
        // A first run compiles the pipelines.
        RunBoolean(f, a, b, BooleanOperation::Union);
        const auto start = std::chrono::steady_clock::now();
        const auto id = RunBoolean(f, a, b, BooleanOperation::Union).value();
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::println("union of two {}-triangle spheres: {:.1f} ms", f.Get(a).DerivedTriangles().size(), 1e3 * elapsed);
        expect(elapsed < 1.0);
        const auto mesh = f.Get(id);
        expect(ClosedManifold(mesh));
        expect(EulerCharacteristic(mesh) == 2);
        // Two unit spheres whose centers are d apart overlap in a lens of volume pi (4 + d)(2 - d)^2 / 12.
        const double d = std::sqrt(.5 * .5 + .3 * .3 + .2 * .2), lens = std::numbers::pi * (4.0 + d) * (2.0 - d) * (2.0 - d) / 12.0;
        expect(std::abs(::test::EnclosedVolume(mesh).value_or(-1.0) - (2.0 * ::test::EnclosedVolume(f.Get(a)).value_or(-1.0) - lens)) < 2e-2);
    };
}
