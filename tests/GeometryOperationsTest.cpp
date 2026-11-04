#include "GeometryFixture.h"
#include "ManifoldChecks.h"
#include "RunSuites.h"
#include "TopologyCases.h"
#include "mesh/MeshClone.h"
#include "mesh/Primitives.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/Simplify.h"

using namespace boost::ut;

namespace {
using ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCases;

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

// An element's vertex handles with their positions, in edge halfedge or face cycle order.
using ElementCorners = std::vector<std::pair<uint32_t, std::array<float, 3>>>;
// Each live element's corners by handle, for the vertex, edge and face domains.
using ElementSnapshot = std::array<std::map<uint32_t, ElementCorners>, 3>;
ElementSnapshot Snapshot(const Mesh &mesh) {
    const auto corner = [&mesh](Mesh::VH v) {
        const auto &p = test::Position(mesh, v);
        return std::pair{*v, std::array{p.x, p.y, p.z}};
    };
    ElementSnapshot snapshot;
    for (const auto v : mesh.vertices()) snapshot[0][*v] = {corner(v)};
    for (const auto e : mesh.edges()) {
        const auto [from, to] = test::EdgeVertices(mesh, e);
        snapshot[1][*e] = {corner(from), corner(to)};
    }
    for (const auto f : mesh.faces()) snapshot[2][*f] = std::ranges::to<ElementCorners>(test::FaceVertices(mesh, f) | std::views::transform(corner));
    return snapshot;
}

// Created holds the handles live after the edit and not before.
// Retained holds the handles live before and after whose corners differ.
struct ElementDiff {
    GeometrySelection Created, Retained;
};
ElementDiff Diff(const ElementSnapshot &before, const ElementSnapshot &after) {
    ElementDiff diff;
    const std::array created{&diff.Created.Vertices, &diff.Created.Edges, &diff.Created.Faces}, retained{&diff.Retained.Vertices, &diff.Retained.Edges, &diff.Retained.Faces};
    for (uint32_t d = 0u; d < 3u; ++d) {
        for (const auto &[handle, corners] : after[d]) {
            const auto it = before[d].find(handle);
            if (it == before[d].end()) created[d]->push_back(handle);
            else if (it->second != corners) retained[d]->push_back(handle);
        }
    }
    return diff;
}
std::pair<GeometryTopologyResult, ElementDiff> RunDiff(GeometryFixture &f, MeshTopologyTask task) {
    const auto source = task.SourceId;
    const auto before = Snapshot(f.Get(source));
    const auto result = f.Run(std::move(task));
    return {result, Diff(result.GeometryId == source ? before : ElementSnapshot{}, Snapshot(f.Get(result.GeometryId)))};
}

// Positions rounded so that results compare across meshes whose handles differ.
using Point = std::array<int64_t, 3>;
Point Key(vec3 p) { return {std::llround(p.x * 1e4f), std::llround(p.y * 1e4f), std::llround(p.z * 1e4f)}; }
std::vector<Point> Points(const Mesh &mesh) {
    std::vector<Point> points;
    for (const auto v : mesh.vertices()) points.push_back(Key(test::Position(mesh, v)));
    return points;
}
std::vector<std::vector<Point>> Shapes(const Mesh &mesh, Element element, std::span<const uint32_t> handles) {
    std::vector<std::vector<Point>> shapes;
    for (const auto handle : handles) {
        auto &shape = shapes.emplace_back();
        if (element == Element::Vertex) shape.push_back(Key(test::Position(mesh, he::VH{handle})));
        else if (element == Element::Edge) {
            const auto [from, to] = test::EdgeVertices(mesh, he::EH{handle});
            shape = {Key(test::Position(mesh, from)), Key(test::Position(mesh, to))};
        } else
            for (const auto v : test::FaceVertices(mesh, he::FH{handle})) shape.push_back(Key(test::Position(mesh, v)));
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
Outcome Describe(const MeshStore &meshes, uint32_t source, const std::pair<GeometryTopologyResult, ElementDiff> &run) {
    const auto &[result, diff] = run;
    const Mesh mesh{meshes, result.GeometryId};
    Outcome outcome{result.Changed, result.GeometryId != source};
    const auto all = AllHandles(mesh);
    for (uint32_t d = 0u; const auto element : {Element::Vertex, Element::Edge, Element::Face}) {
        outcome.Elements[d] = Shapes(mesh, element, diff.Created.Get(element));
        outcome.Elements[3u + d] = Shapes(mesh, element, diff.Retained.Get(element));
        outcome.Elements[6u + d++] = Shapes(mesh, element, all.Get(element));
    }
    return outcome;
}

// Named attributes of every type on every domain, each following a fixed identity channel.
// Morph targets and skin hold vertex identity, UVs and the blue color channel hold corner identity, and primitives and sharpness hold face identity.
// A sharp edge's identifier has 1000 added.
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
    const auto id = f.Create(std::move(source));
    auto &meshes = f.Store();
    {
        const auto mesh = f.Get(id);
        GeometrySelection faces, edges;
        for (const auto face : mesh.faces())
            if (mesh.FaceOrdinal(face) % 3u == 0u) faces.Faces.push_back(*face);
        for (uint32_t e = 0u; e < mesh.EdgeCount(); e += 3u) edges.Edges.push_back(*mesh.EdgeAt(e));
        mtl::ComputeChain chain{meshes.BufferContext()};
        const SharpnessOperationTarget sharp_faces{id, faces}, sharp_edges{id, edges};
        ExecuteGeometrySharpness(f.Scene, chain, sharp_faces, EditSharpnessOperation::SetSelectedFaces, true);
        ExecuteGeometrySharpness(f.Scene, chain, sharp_edges, EditSharpnessOperation::SetSelectedEdges, true);
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

// A copied output renumbers its primitives, so its face integers only come from the source.
void CheckAttributed(const MeshStore &meshes, uint32_t id, const std::set<int32_t> &edge_values, uint32_t face_count, bool copied) {
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
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
        else expect(value == int32_t(a.FacePrimitives.Get(record.FacePrimitives, face))) << "face" << face << "int" << value << "primitive" << a.FacePrimitives.Get(record.FacePrimitives, face);
        expect(ff[face] == FaceFloat(value)) << "face" << face << "int" << value << "float" << ff[face];
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
        expect(blue(float(ci[h])));
        if (blue(color.z)) expect(ci[h] == int32_t(color.z));
        expect(cb[h] == (ci[h] >= 1000));
    });
}
} // namespace

void RegisterManifoldOracleTests();
void RegisterGeometryOperationsTests() {
    "simplify toward a ratio keeps that fraction of the triangles"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(primitive::CreateMesh(primitive::IcoSphere{.Subdivisions = 2}));
        const auto triangles = [&] {
            uint32_t count = 0u;
            for (const auto face : f.Get(id).faces()) count += test::Valence(f.Get(id), face) - 2u;
            return count;
        };
        const auto before = triangles();
        expect(SimplifyGeometry(f.Scene, id, AllHandles(f.Get(id)), 0.f, .5f, [&](const MeshTopologyTask &task) { f.Run(task); })) << "collapsed";
        const auto after = triangles();
        expect(after <= before / 2u && after >= before / 2u - 8u) << "triangles" << after << "of" << before;
        expect(f.Get(id).HasClosedSurface()) << "closed";
    };
    "topology operators preserve unselected vertices and agree on masks and lists"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        const auto patchwork = f.Get(f.Create(Patchwork()));
        expect(::test::TriangulatesLikeManifold(patchwork, patchwork.FaceAt(11u)));
        for (auto polygon : ::test::TriangulationPolygons()) {
            const auto id = f.Create(std::move(polygon));
            expect(::test::TriangulatesLikeManifold(f.Get(id), f.Get(id).FaceAt(0u)));
            const PositionOperationTarget moving{.StoreId = id, .Selection = GeometrySelection{.Vertices = AllHandles(f.Get(id)).Vertices}};
            ExecuteGeometryPositions(f.Scene, moving, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {3.f, -1.f, 2.f}, .R = quat{.9238795f, .3826834f, 0.f, 0.f}, .S = {1.f, 2.5f, .6f}}}});
            expect(::test::TriangulatesLikeManifold(f.Get(id), f.Get(id).FaceAt(0u)));
            f.Run({.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .Flags = TopologyFlagSelectAll});
            expect(::test::TriangulatesLikeManifold(f.Get(id), f.Get(id).FaceAt(0u)));
        }
        for (const auto &c : TopologyCases) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto source = f.Get(listed_id);
                const auto selected = Region(source);
                auto untouched = AllHandles(source).Vertices;
                std::erase_if(untouched, [&](uint32_t v) { return std::ranges::binary_search(selected.Vertices, v); });
                const auto positions = Shapes(source, Element::Vertex, untouched);
                auto listed = PlanCase(c, f.Scene, source);
                auto masked = PlanCase(c, f.Scene, f.Get(masked_id));
                expect(fatal(listed.has_value() && masked.has_value()));
                PublishStoreSelection(f.Store(), *listed);
                masked->Selection = masks.Write(f.Store(), masked_id, std::get<GeometrySelection>(masked->Selection));
                masked->SelectionOutput = masks.Write(f.Store(), masked_id, {});
                const auto listed_run = RunDiff(f, *listed);
                const auto masked_run = RunDiff(f, *masked);
                const auto by_list = Describe(f.Store(), listed_id, listed_run);
                const auto by_masks = Describe(f.Store(), masked_id, masked_run);
                expect(by_list.Changed);
                expect(by_list == by_masks);
                const auto listed_result = f.Get(listed_run.first.GeometryId), masked_result = f.Get(masked_run.first.GeometryId);
                for (const auto element : {Element::Vertex, Element::Edge, Element::Face}) {
                    const auto into_store = ReadMasks(f.Store(), listed_result.GetStoreId(), element, listed->SelectionOutput->Get(element));
                    const auto into_masks = ReadMasks(f.Store(), masked_result.GetStoreId(), element, masked->SelectionOutput->Get(element));
                    expect(std::ranges::equal(Shapes(listed_result, element, into_store), Shapes(masked_result, element, into_masks))) << "element" << uint32_t(element);
                    expect(f.Store().GetSelectedElements(masked_result.GetStoreId(), element).Count() == 0u);
                }
                if (listed->Op != MeshTopologyOp::KeepSelectedFaces)
                    expect(std::ranges::all_of(positions, [&](const auto &point) { return std::ranges::contains(by_list.Elements[6], point); }));
                expect(!f.Store().Get(listed_result.GetStoreId()).MeshletData);
                if (c.Holds) expect(c.Holds(listed_result));
            };
        }
    };
    "named attributes carry through topology operators on every domain"_test = [] {
        GeometryFixture f;
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
        // A convex hull's new faces take a source face's primitive while their named attributes take the defaults, so the hull does not follow the channels.
        for (const auto &c : TopologyCases)
            if (c.Task.Op != MeshTopologyOp::AddPrimitives) carry(c.Name, Patchwork(), [&](uint32_t id) { return PlanCase(c, f.Scene, f.Get(id)); });
        carry("Boolean", primitive::CreateMesh(primitive::Cube{}), [&](uint32_t id) {
            auto box = primitive::CreateMesh(primitive::Cube{});
            for (auto &p : box.Positions) p += vec3{1.f};
            const auto second = f.Create(std::move(box));
            return BooleanTask(f.Get(id), f.Get(second), BooleanOperation::Union, mat3{1.f}, vec3{0.f});
        });
    };
    "named attributes clone independently and release their payloads"_test = [] {
        GeometryFixture f;
        auto &meshes = f.Store();
        const auto id = f.Create(UnitQuad());
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
        const auto excluded = [](const Mesh &mesh) {
            GeometrySelection result;
            for (const auto face : mesh.faces())
                if (test::Valence(mesh, face) == 3u) result.Faces.push_back(*face);
            return result;
        };
        for (const auto &c : positions) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto listed_selection = Region(f.Get(listed_id)), masked_selection = Region(f.Get(masked_id));
                const PositionOperationTarget listed{.StoreId = listed_id, .Selection = listed_selection, .Excluded = excluded(f.Get(listed_id)), .Reference = listed_selection.Vertices[4]};
                const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), masked_id, masked_selection), .Excluded = masks.Write(f.Store(), masked_id, excluded(f.Get(masked_id))), .Reference = masked_selection.Vertices[4]};
                const auto before = Points(f.Get(listed_id));
                const auto listed_change = ExecuteGeometryPositions(f.Scene, listed, c.Op, c.Factor, 1u, c.Options);
                const auto masked_change = ExecuteGeometryPositions(f.Scene, masked, c.Op, c.Factor, 1u, c.Options);
                f.Audit();
                expect(listed_change.has_value() == masked_change.has_value());
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
            const auto listed_id = f.Create({.Data = Patchwork(), .Attrs = uvs()}), masked_id = f.Create({.Data = Patchwork(), .Attrs = uvs()});
            const PositionOperationTarget listed{.StoreId = listed_id, .Selection = Region(f.Get(listed_id))};
            const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), masked_id, Region(f.Get(masked_id)))};
            mtl::ComputeChain chain{f.Store().BufferContext()};
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, listed, false, 0u, 1u).has_value());
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, masked, false, 0u, 1u).has_value());
            chain.Submit();
            f.Audit();
            const auto corner_uvs = [&](uint32_t id) {
                std::vector<vec2> result;
                for (const auto face : f.Get(id).faces())
                    for (const auto corner : test::FaceHalfedges(f.Get(id), face)) result.push_back(f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(f.Get(id).GetStoreId()).CornerUvs[0], *corner));
                return result;
            };
            expect(std::ranges::equal(corner_uvs(listed_id), corner_uvs(masked_id)));
        };
        struct SharpnessCase {
            EditSharpnessOperation Op;
            Element Domain;
            bool (*Sharp)(const Mesh &, const GeometrySelection &, uint32_t);
        };
        const std::array<SharpnessCase, 5> sharpness_cases{{
            {EditSharpnessOperation::SetSelectedFaces, Element::Face, [](const Mesh &, const GeometrySelection &s, uint32_t h) { return std::ranges::binary_search(s.Faces, h); }},
            {EditSharpnessOperation::SetSelectedEdges, Element::Edge, [](const Mesh &, const GeometrySelection &s, uint32_t h) { return std::ranges::binary_search(s.Edges, h); }},
            {EditSharpnessOperation::SetVertexEdges, Element::Edge, [](const Mesh &mesh, const GeometrySelection &s, uint32_t h) {
                 return std::ranges::any_of(test::EdgeVertices(mesh, he::EH{h}), [&](he::VH v) { return std::ranges::binary_search(s.Vertices, *v); });
             }},
            {EditSharpnessOperation::SetAllFaces, Element::Face, [](const Mesh &, const GeometrySelection &, uint32_t) { return true; }},
            {EditSharpnessOperation::SmoothByAngle, Element::Edge, [](const Mesh &mesh, const GeometrySelection &, uint32_t h) {
                 const auto half = test::Halfedge(mesh, he::EH{h}), opposite = test::Opposite(mesh, half);
                 const auto face = test::Face(mesh, half), other = opposite ? test::Face(mesh, opposite) : he::FH{};
                 return face && other && Dot(test::Normal(mesh, face), test::Normal(mesh, other)) < std::cos(.1f);
             }},
        }};
        for (const auto &c : sharpness_cases) {
            boost::ut::test("sharpness " + std::to_string(uint32_t(c.Op))) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto region = Region(f.Get(listed_id));
                const SharpnessOperationTarget listed{listed_id, region};
                const SharpnessOperationTarget masked{masked_id, masks.Write(f.Store(), masked_id, Region(f.Get(masked_id)))};
                const auto face_normals = [&](uint32_t id) {
                    std::vector<vec3> result;
                    for (const auto face : f.Get(id).faces()) result.push_back(test::Normal(f.Get(id), face));
                    return result;
                };
                const auto normals = face_normals(listed_id);
                mtl::ComputeChain chain{f.Store().BufferContext()};
                const auto listed_change = ExecuteGeometrySharpness(f.Scene, chain, listed, c.Op, true, .1f);
                const auto masked_change = ExecuteGeometrySharpness(f.Scene, chain, masked, c.Op, true, .1f);
                f.Audit();
                expect(listed_change.has_value());
                expect(masked_change.has_value());
                const auto sharpness = [&](uint32_t id) {
                    std::vector<uint8_t> result;
                    for (const auto face : f.Get(id).faces()) result.push_back(f.Store().Arenas().FaceSharpness.Get({test::Address(f.Get(id), face), 1u})[0]);
                    for (const auto edge : f.Get(id).edges()) result.push_back(f.Store().Arenas().EdgeSharpness.Get({test::Address(f.Get(id), edge), 1u})[0]);
                    return result;
                };
                expect(std::ranges::equal(sharpness(listed_id), sharpness(masked_id)));
                const auto mesh = f.Get(listed_id);
                bool sharp_face = false, sharp_edge = false;
                for (const auto face : mesh.faces()) {
                    const bool sharp = f.Store().Arenas().FaceSharpness.Get({test::Address(mesh, face), 1u})[0] != 0u;
                    expect((c.Domain == Element::Face && c.Sharp(mesh, region, *face)) == sharp) << "face" << *face;
                    sharp_face |= sharp;
                }
                for (const auto edge : mesh.edges()) {
                    const bool sharp = f.Store().Arenas().EdgeSharpness.Get({test::Address(mesh, edge), 1u})[0] != 0u;
                    expect((c.Domain == Element::Edge && c.Sharp(mesh, region, *edge)) == sharp) << "edge" << *edge;
                    sharp_edge |= sharp;
                }
                expect(!(sharp_face || sharp_edge) || f.Store().GetCornerClassMode(listed_id) != uint32_t(CornerClassMode::UniformVertex));
                expect(!sharp_face || (f.Store().GetSelectionRoot(listed_id, Element::Face).Flags & SelectionLiveSharp) != 0u);
                expect(face_normals(listed_id) == normals);
                expect(f.Store().GetSelectedElements(listed_id, Element::Vertex).Count() == 0u);
                expect(!f.Store().Get(listed_id).MeshletData);
                expect(f.Scene.Living.empty());
                const SharpnessOperationTarget unselected{listed_id, GeometrySelection{}};
                expect(ExecuteGeometrySharpness(f.Scene, chain, unselected, c.Op, true, .1f).has_value() == (c.Op < EditSharpnessOperation::SetSelectedFaces));
            };
        }
    };
    "mixed sources weld wire endpoints onto a surface and keep their corner attributes"_test = [] {
        GeometryFixture f;
        for (const bool weld : {false, true}) {
            MeshSource source{.Weld = weld};
            source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 0, 0}, {3, 0, 0}, {0, 0, 0}};
            source.Data.AddFace(std::array{0u, 1u, 2u});
            source.Data.Edges = {{5, 3}, {3, 4}};
            auto &uvs = source.Attrs.TexCoords0.emplace();
            for (uint32_t i = 0u; i < source.Data.Positions.size(); ++i) uvs.push_back({float(i), .25f});
            const auto id = f.Create(std::move(source));
            const auto mesh = f.Get(id);
            expect(mesh.VertexCount() == (weld ? 5u : 6u) && mesh.EdgeCount() == 5u && mesh.FaceCount() == 1u);
            const auto uv = [&](he::HH h) { return f.Store().Arenas().CornerUvs[0].Get(f.Store().Get(id).CornerUvs[0], *h); };
            for (const auto face : mesh.faces())
                for (const auto h : test::FaceHalfedges(mesh, face)) expect(uv(h) == vec2{float(mesh.VertexOrdinal(test::ToVertex(mesh, h))), .25f});
            for (const auto edge : mesh.edges()) {
                const auto h = test::Halfedge(mesh, edge);
                if (test::Face(mesh, h)) continue;
                for (const auto corner : {h, test::Opposite(mesh, h)}) {
                    const auto p = test::Position(mesh, test::ToVertex(mesh, corner));
                    expect(uv(corner).x == (p.x == 0.f ? 5.f : p.x + 1.f));
                }
            }
        }
    };
}

int main(int argc, char **argv) {
    if (argc > 1) cfg<override> = {.filter = argv[1]};
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    RegisterGeometryOperationsTests();
    RegisterManifoldOracleTests();
    return RunSuites();
}
