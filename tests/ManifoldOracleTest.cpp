#include "GeometryFixture.h"
#include "ManifoldChecks.h"
#include "MeshProperties.h"
#include "mesh/AttributeFields.h"
#include "mesh/ConvexHull.h"
#include "mesh/LevelSet.h"
#include "mesh/Minkowski.h"
#include "mesh/Primitives.h"
#include "mesh/Region.h"
#include "mesh/Simplify.h"
#include "mesh/TopologyOperations.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"

#include <manifold/cross_section.h>
#include <manifold/manifold.h>
#include <random>

using boost::ut::expect;
using namespace boost::ut::literals;
using numeric::dvec3;

namespace {
// Cases per row, each a seeded placement of random primitives.
constexpr uint32_t Seeds = 8;

// A closed surface in a form indifferent to representation and precision: its face vertices, surface area, volume and Euler characteristic.
struct Shape {
    std::vector<dvec3> Points;
    double Area{}, Volume{};
    int64_t Euler{};
    double Extent{};
};
double TriangleArea(dvec3 a, dvec3 b, dvec3 c) { return Length(Cross(b - a, c - a)) / 2.0; }
double Extent(std::span<const dvec3> points) {
    dvec3 lo{1e300, 1e300, 1e300}, hi = -lo;
    for (const auto p : points) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    return points.empty() ? 0.0 : Length(hi - lo);
}

Shape ShapeOf(const Mesh &mesh) {
    Shape shape;
    std::vector<bool> used(mesh.VertexCount());
    for (const auto face : mesh.faces()) {
        for (const auto v : test::FaceVertices(mesh, face)) used[mesh.VertexOrdinal(v)] = true;
    }
    for (uint32_t i = 0; i < used.size(); ++i) {
        if (used[i]) shape.Points.emplace_back(test::Position(mesh, mesh.VertexAt(i)));
    }
    shape.Extent = Extent(shape.Points);
    // The rendered triangles, which are also what the reference receives, so a non-planar face measures the same on both sides.
    for (const auto triangle : mesh.DerivedTriangles()) {
        const auto [a, b, c] = test::TrianglePositions(mesh, triangle);
        shape.Area += TriangleArea(dvec3{a}, dvec3{b}, dvec3{c});
    }
    shape.Volume = test::EnclosedVolume(mesh).value_or(0.0);
    shape.Euler = int64_t(shape.Points.size()) - int64_t(mesh.EdgeCount()) + int64_t(mesh.FaceCount());
    return shape;
}
std::vector<std::pair<uint64_t, uint64_t>> UniqueEdges(const manifold::MeshGL64 &gl) {
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    for (size_t t = 0; t < gl.NumTri(); ++t) {
        for (uint32_t k = 0; k < 3; ++k) edges.push_back(std::minmax(gl.triVerts[3 * t + k], gl.triVerts[3 * t + (k + 1) % 3]));
    }
    std::ranges::sort(edges);
    edges.erase(std::ranges::unique(edges).begin(), edges.end());
    return edges;
}
Shape ShapeOf(const manifold::Manifold &solid) {
    const auto gl = solid.GetMeshGL64();
    Shape shape;
    for (size_t v = 0; v < gl.NumVert(); ++v) shape.Points.emplace_back(gl.vertProperties[v * gl.numProp], gl.vertProperties[v * gl.numProp + 1], gl.vertProperties[v * gl.numProp + 2]);
    shape.Extent = Extent(shape.Points);
    for (size_t t = 0; t < gl.NumTri(); ++t) shape.Area += TriangleArea(shape.Points[gl.triVerts[3 * t]], shape.Points[gl.triVerts[3 * t + 1]], shape.Points[gl.triVerts[3 * t + 2]]);
    shape.Volume = solid.Volume();
    shape.Euler = int64_t(gl.NumVert()) - int64_t(UniqueEdges(gl).size()) + int64_t(gl.NumTri());
    return shape;
}

manifold::Manifold ToManifold(const Mesh &mesh) {
    manifold::MeshGL64 gl;
    gl.numProp = 3;
    for (uint32_t i = 0; i < mesh.VertexCount(); ++i) {
        const auto p = test::Position(mesh, mesh.VertexAt(i));
        gl.vertProperties.insert(gl.vertProperties.end(), {p.x, p.y, p.z});
    }
    for (const auto corners : mesh.DerivedTriangles()) {
        for (const auto c : {corners.x, corners.y, corners.z}) gl.triVerts.push_back(mesh.VertexOrdinal(test::ToVertex(mesh, Mesh::HH{c})));
    }
    manifold::Manifold solid{gl};
    expect(solid.Status() == manifold::Manifold::Error::NoError) << "manifold input status" << int(solid.Status());
    return solid;
}

void ExpectSameMeasures(const Shape &actual, const Shape &reference, std::string_view label) {
    expect(std::abs(actual.Area - reference.Area) <= 1e-3 * std::max(1.0, reference.Area)) << label << "area" << actual.Area << "vs" << reference.Area;
    expect(std::abs(actual.Volume - reference.Volume) <= 1e-3 * std::max(1.0, reference.Volume)) << label << "volume" << actual.Volume << "vs" << reference.Volume;
    expect(actual.Euler == reference.Euler) << label << "euler" << actual.Euler << "vs" << reference.Euler;
}
std::vector<bool> MatchPoints(std::span<const dvec3> points, std::span<const dvec3> candidates, double tolerance, std::string_view label, std::string_view role) {
    std::vector<bool> matched(candidates.size());
    for (const auto p : points) {
        size_t nearest = candidates.size();
        double distance = 1e300;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (const auto d = Length(candidates[i] - p); !matched[i] && d < distance) {
                nearest = i;
                distance = d;
            }
        }
        if (nearest < matched.size() && distance <= tolerance) matched[nearest] = true;
        else expect(false) << label << role << p.x << p.y << p.z << "is" << distance << "from its nearest unmatched counterpart";
    }
    return matched;
}
// A reference cut on triangles carries vertices on the diagonals of the faces a polygon cut leaves whole, which `subset` admits.
void ExpectSame(const Shape &actual, const Shape &reference, std::string_view label, bool subset = false) {
    const double tolerance = 2e-4 * std::max(1.0, reference.Extent);
    if (!subset) expect(actual.Points.size() == reference.Points.size()) << label << "points" << actual.Points.size() << "vs" << reference.Points.size();
    MatchPoints(actual.Points, reference.Points, tolerance, label, "point");
    ExpectSameMeasures(actual, reference, label);
}

// Equality as solids within the precision of a float result against a double reference, up to the collinear vertices the result keeps on the reference's edges.
void ExpectSameSolid(const Mesh &actual_mesh, const manifold::Manifold &reference_solid, std::string_view label) {
    const auto actual = ShapeOf(actual_mesh), reference = ShapeOf(reference_solid);
    const double tolerance = 2e-4 * std::max(1.0, reference.Extent);
    const auto matched = MatchPoints(reference.Points, actual.Points, tolerance, label, "reference point");
    const auto edges = UniqueEdges(reference_solid.GetMeshGL64());
    const auto on_edge = [&](dvec3 p) {
        return std::ranges::any_of(edges, [&](const auto edge) {
            const auto a = reference.Points[edge.first], b = reference.Points[edge.second];
            const auto ab = b - a;
            const double t = std::clamp(Dot(p - a, ab) / Dot(ab, ab), 0.0, 1.0);
            return Length(p - (a + ab * t)) <= tolerance;
        });
    };
    for (size_t i = 0; i < actual.Points.size(); ++i) {
        const auto p = actual.Points[i];
        if (!matched[i] && !on_edge(p)) expect(false) << label << "point" << p.x << p.y << p.z << "matches no reference point and lies on no reference edge";
    }
    ExpectSameMeasures(actual, reference, label);
}

struct Random {
    std::mt19937 Gen;
    float Uniform(float a, float b) { return std::uniform_real_distribution<float>{a, b}(Gen); }
    uint32_t Below(uint32_t n) { return std::uniform_int_distribution<uint32_t>{0u, n - 1u}(Gen); }
    quat Rotation() {
        std::normal_distribution<float> normal;
        return Normalize(quat{normal(Gen), normal(Gen), normal(Gen), normal(Gen)});
    }
    Transform Placement() {
        return {.P = {Uniform(-1.2f, 1.2f), Uniform(-1.2f, 1.2f), Uniform(-1.2f, 1.2f)}, .R = Rotation(), .S = {Uniform(.6f, 1.6f), Uniform(.6f, 1.6f), Uniform(.6f, 1.6f)}};
    }
    // A random primitive, with the ico sphere in place of the quad sphere when every face must be planar.
    PrimitiveParameters Shape(bool planar = false) {
        switch (Below(5)) {
            case 0: return primitive::Cube{};
            case 1: return planar ? PrimitiveParameters{primitive::IcoSphere{.Subdivisions = 2}} : PrimitiveParameters{primitive::UVSphere{.Slices = 16, .Stacks = 8}};
            case 2: return primitive::Cylinder{.Slices = 12};
            case 3: return primitive::Cylinder{.TopRadius = .4f, .Slices = 10};
            default: return planar ? PrimitiveParameters{primitive::IcoSphere{.Subdivisions = 1}} : PrimitiveParameters{primitive::QuadSphere{.Segments = 3}};
        }
    }
};
MeshData Placed(const PrimitiveParameters &shape, const Transform &t) {
    auto data = primitive::CreateMesh(shape);
    for (auto &p : data.Positions) p = t.P + t.R * (t.S * p);
    return data;
}
std::string Label(std::string_view row, uint32_t seed) { return std::format("{} seed {}", row, seed); }
// The signed area of the loop's XY projection, positive where it winds counterclockwise.
double SignedArea(const auto &loop) {
    double area = 0.0;
    for (size_t i = 0; i < loop.size(); ++i) {
        const auto p = loop[i], q = loop[(i + 1) % loop.size()];
        area += double(p.x) * double(q.y) - double(q.x) * double(p.y);
    }
    return area / 2.0;
}
// A reference sliced from triangles carries vertices on the diagonals of faces the plane fill leaves whole, which `subset` admits.
void ExpectSamePolygons(std::span<const std::vector<vec2>> loops, const manifold::Polygons &reference, std::string_view label, bool subset = false) {
    std::vector<dvec3> actual_points, reference_points;
    double actual_area = 0.0, reference_area = 0.0;
    for (const auto &loop : loops) {
        for (const auto p : loop) actual_points.emplace_back(p.x, p.y, 0.0);
        actual_area += SignedArea(loop);
    }
    for (const auto &polygon : reference) {
        for (const auto p : polygon) reference_points.emplace_back(p.x, p.y, 0.0);
        reference_area += SignedArea(polygon);
    }
    expect(loops.size() == reference.size()) << label << "loops" << loops.size() << "vs" << reference.size();
    if (!subset) expect(actual_points.size() == reference_points.size()) << label << "points" << actual_points.size() << "vs" << reference_points.size();
    const double tolerance = 2e-4 * std::max(1.0, Extent(reference_points));
    for (const auto p : actual_points) {
        double nearest = 1e300;
        for (const auto q : reference_points) nearest = std::min(nearest, Length(q - p));
        expect(nearest <= tolerance) << label << "point" << p.x << p.y << "is" << nearest << "from the reference";
    }
    expect(std::abs(actual_area - reference_area) <= 1e-3 * std::max(1.0, std::abs(reference_area))) << label << "area" << actual_area << "vs" << reference_area;
}
std::vector<std::vector<vec2>> PlanarLoops(const Mesh &mesh) {
    std::vector<std::vector<vec2>> loops;
    for (const auto face : mesh.faces()) {
        auto &loop = loops.emplace_back();
        for (const auto v : test::FaceVertices(mesh, face)) loop.emplace_back(test::Position(mesh, v).x, test::Position(mesh, v).y);
    }
    return loops;
}
// A regular polygon in the XY plane, counterclockwise, as a region record and as manifold's polygon.
std::pair<MeshData, manifold::SimplePolygon> RegularPolygon(uint32_t sides, vec2 center, float radius, float phase, bool star = false) {
    MeshData data;
    manifold::SimplePolygon reference;
    std::vector<uint32_t> face;
    for (uint32_t i = 0u; i < sides; ++i) {
        const float angle = phase + 2.f * std::numbers::pi_v<float> * float(i) / float(sides);
        const vec2 p = center + (star && i % 2u ? .5f * radius : radius) * vec2{std::cos(angle), std::sin(angle)};
        face.push_back(i);
        data.Positions.emplace_back(p.x, p.y, 0.f);
        reference.emplace_back(p.x, p.y);
    }
    data.AddFace(face);
    return std::pair{std::move(data), std::move(reference)};
}
// Bisect's run: the plane cut, then the vertices on the cleared side deleted.
uint32_t Bisected(GeometryFixture &f, uint32_t id, vec3 point, vec3 normal, PlaneSideKind cleared) {
    id = f.Run(PlaneCutTask(f.Get(id), point, normal)).GeometryId;
    if (const auto clear = PlaneSideDeleteTask(f.Scene, f.Get(id), point, normal, cleared, PlaneSideTolerance)) id = f.Run(*clear).GeometryId;
    return id;
}
// The macros' cap: Fill over the boundary edges whose vertices lie on the plane, with the vertices a linked edge selection derives.
uint32_t Capped(GeometryFixture &f, uint32_t id, vec3 point, vec3 normal) {
    auto &meshes = f.Store();
    const auto on = PlaneSideField(f.Scene, id, AttributeDomain::Edge, point, normal, PlaneSideKind::On, PlaneSideTolerance, "on").value();
    const auto boundary = BoundaryField(f.Scene, id, AttributeDomain::Edge, "boundary").value();
    const auto mesh = f.Get(id);
    GeometrySelection selection;
    for (const auto e : mesh.edges()) {
        if (!meshes.AttributeView<bool>(on)[*e] || !meshes.AttributeView<bool>(boundary)[*e]) continue;
        selection.Edges.push_back(*e);
        for (const auto v : test::EdgeVertices(mesh, e)) selection.Vertices.push_back(*v);
    }
    meshes.ReleaseAttribute(on);
    meshes.ReleaseAttribute(boundary);
    std::ranges::sort(selection.Edges);
    std::ranges::sort(selection.Vertices);
    selection.Vertices.erase(std::ranges::unique(selection.Vertices).begin(), selection.Vertices.end());
    const auto fill = FillTask(f.Scene, mesh, selection, 0u);
    return fill ? f.Run(*fill).GeometryId : id;
}

void Append(MeshData &data, const MeshData &other) {
    const auto base = uint32_t(data.Positions.size());
    data.Positions.insert(data.Positions.end(), other.Positions.begin(), other.Positions.end());
    for (uint32_t face = 0u; face < other.FaceCount(); ++face) {
        std::vector<uint32_t> loop;
        for (const auto v : other.Face(face)) loop.push_back(base + v);
        data.AddFace(loop);
    }
    for (const auto &[a, b] : other.Edges) data.Edges.push_back({base + a, base + b});
}
std::optional<uint32_t> RunBoolean(GeometryFixture &f, uint32_t first, uint32_t second, BooleanOperation operation) {
    const auto task = BooleanTask(f.Get(first), f.Get(second), operation, mat3{1.f}, vec3{0.f});
    const auto id = task ? ExecuteGeometryBoolean(f.Scene, *task) : std::nullopt;
    f.Audit();
    return id;
}

std::array<double, 2> AreasHolding(const Mesh &mesh, uint32_t attribute, float value) {
    std::array<double, 2> areas{};
    const auto &meshes = mesh.GetStore();
    const auto view = meshes.AttributeView<float>(attribute);
    for (const auto face : mesh.faces()) {
        for (const auto t : test::RenderTriangles(meshes, mesh, face, test::Valence(mesh, face) - 2u)) {
            const auto [a, b, c] = test::TrianglePositions(mesh, t);
            areas[view[*face] == value ? 0u : 1u] += TriangleArea(dvec3{a}, dvec3{b}, dvec3{c});
        }
    }
    return areas;
}
std::array<double, 2> AreasFrom(const manifold::Manifold &solid, int original) {
    const auto gl = solid.GetMeshGL64();
    std::array<double, 2> areas{};
    for (size_t run = 0; run < gl.NumRun(); ++run) {
        for (auto t = gl.runIndex[run] / 3; t < gl.runIndex[run + 1] / 3; ++t) {
            const auto corner = [&](uint32_t k) {
                const auto v = gl.triVerts[3 * t + k] * gl.numProp;
                return dvec3{gl.vertProperties[v], gl.vertProperties[v + 1], gl.vertProperties[v + 2]};
            };
            areas[int(gl.runOriginalID[run]) == original ? 0u : 1u] += TriangleArea(corner(0), corner(1), corner(2));
        }
    }
    return areas;
}
// The inputs stay as they were, and the first's face attribute marks the result's faces from it, which cover the area of manifold's triangles from it.
std::pair<std::optional<uint32_t>, manifold::Manifold> MatchBoolean(GeometryFixture &f, uint32_t a, uint32_t b, BooleanOperation operation, manifold::OpType reference_operation, std::string_view label) {
    auto &meshes = f.Store();
    // An original ID marks the first solid's triangles in manifold's result.
    const auto ma = ToManifold(f.Get(a)).AsOriginal(), mb = ToManifold(f.Get(b));
    const auto reference = ma.Boolean(mb, reference_operation);
    const auto weight = meshes.AddAttribute(a, AttributeDomain::Face, AttributeType::Float, "weight", AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u}).value();
    for (const auto face : f.Get(a).faces()) {
        const float value = 5.f;
        meshes.SetAttribute<float>(weight, *face, std::span{&value, 1u});
    }
    const auto a_vertices = f.Get(a).VertexCount(), b_vertices = f.Get(b).VertexCount();
    const auto id = RunBoolean(f, a, b, operation);
    expect(f.Get(a).VertexCount() == a_vertices && f.Get(b).VertexCount() == b_vertices) << label << "inputs changed";
    if (id && !reference.IsEmpty()) {
        const auto output = meshes.FindAttribute(*id, "weight");
        expect(output != InvalidOffset) << label << "weight";
        if (output != InvalidOffset) {
            const auto actual = AreasHolding(f.Get(*id), output, 5.f), expected = AreasFrom(reference, ma.OriginalID());
            for (uint32_t k = 0u; k < 2u; ++k) expect(std::abs(actual[k] - expected[k]) <= 1e-3 * std::max(1.0, expected[k])) << label << (k ? "second's" : "first's") << "area" << actual[k] << "vs" << expected[k];
        }
    }
    return {id, reference};
}
void ExpectSameBoolean(const Mesh &actual, const manifold::Manifold &reference, std::string_view label) {
    if (reference.IsEmpty()) expect(actual.FaceCount() == 0u) << label << "expected empty";
    else ExpectSameSolid(actual, reference, label);
}

// How a row's result compares with manifold's: as the same shape, as a shape whose points are a subset of manifold's, or as the same solid.
enum class Match { Same,
                   Subset,
                   Solid };
// Run draws the operator's parameters after the input and returns our result record and manifold's result on the same input.
struct SolidRow {
    std::string_view Name;
    MeshData (*Input)(Random &);
    std::pair<std::optional<uint32_t>, manifold::Manifold> (*Run)(GeometryFixture &, Random &, uint32_t);
    Match Mode;
};
MeshData RandomSolid(Random &rng) { return Placed(rng.Shape(), rng.Placement()); }
MeshData RandomIcoSphere(Random &rng) { return Placed(primitive::IcoSphere{.Subdivisions = 1}, rng.Placement()); }
template<BooleanOperation Operation, manifold::OpType ReferenceOperation> std::pair<std::optional<uint32_t>, manifold::Manifold> RandomBoolean(GeometryFixture &f, Random &rng, uint32_t a) {
    return MatchBoolean(f, a, f.Create(RandomSolid(rng)), Operation, ReferenceOperation, "boolean");
}
const std::array<SolidRow, 12> SolidRows{{
    {"union", RandomSolid, RandomBoolean<BooleanOperation::Union, manifold::OpType::Add>, Match::Solid},
    {"difference", RandomSolid, RandomBoolean<BooleanOperation::Difference, manifold::OpType::Subtract>, Match::Solid},
    {"intersection", RandomSolid, RandomBoolean<BooleanOperation::Intersection, manifold::OpType::Intersect>, Match::Solid},
    {"hull", [](Random &rng) {
         std::vector<vec3> points(8 + rng.Below(33));
         for (auto &p : points) p = {rng.Uniform(-1.f, 1.f), rng.Uniform(-1.f, 1.f), rng.Uniform(-1.f, 1.f)};
         return MeshData{points}; },
     [](GeometryFixture &f, Random &, uint32_t id) {
         std::vector<manifold::vec3> points;
         for (const auto v : f.Get(id).vertices()) {
             const auto p = test::Position(f.Get(id), v);
             points.emplace_back(p.x, p.y, p.z);
         }
         for (const auto &task : ConvexHullTasks(f.Scene, f.Get(id), AllHandles(f.Get(id)))) f.Run(task);
         return std::pair{std::optional{id}, manifold::Manifold::Hull(points)};
     },
     Match::Same},
    {"subdivide", RandomIcoSphere, [](GeometryFixture &f, Random &rng, uint32_t id) {
         const auto reference = ToManifold(f.Get(id));
         const auto cuts = 1u + rng.Below(3);
         f.Run({.SourceId = id, .Op = MeshTopologyOp::Subdivide, .Param0 = float(cuts), .Flags = TopologyFlagSelectAll});
         return std::pair{std::optional{id}, reference.Refine(int(cuts) + 1)};
     },
     Match::Same},
    // Both divide each edge into pieces no longer than the length, by the truncated ratio, and fill triangles whose counts differ with manifold's partition tables.
    {"subdivide to length", RandomIcoSphere, [](GeometryFixture &f, Random &rng, uint32_t id) {
         const auto reference = ToManifold(f.Get(id));
         const auto length = rng.Uniform(.25f, .6f);
         const auto task = SubdivideLengthTask(f.Scene, f.Get(id), AllHandles(f.Get(id)), length);
         if (task) f.Run(*task);
         return std::pair{task ? std::optional{id} : std::nullopt, reference.RefineToLength(length)};
     },
     Match::Same},
    // A coarse ico sphere with uniform scale leaves every triangle unpaired, so both sides evaluate the same tangents on the same triangles.
    {"smooth surface", [](Random &rng) {
         auto placement = rng.Placement();
         placement.S = vec3{placement.S.x};
         return Placed(primitive::IcoSphere{.Subdivisions = 1}, placement); }, [](GeometryFixture &f, Random &rng, uint32_t id) {
         const auto cuts = 1u + rng.Below(3);
         const auto reference = ToManifold(f.Get(id)).CalculateNormals(0, 180.0).SmoothByNormals(0).Refine(int(cuts) + 1);
         ExecuteGeometryTopologyStages(f.Scene, SmoothSurfaceTasks(f.Scene, f.Get(id), cuts));
         f.Audit();
         return std::pair{std::optional{id}, reference}; }, Match::Same},
    {"transform", RandomSolid, [](GeometryFixture &f, Random &rng, uint32_t id) {
         const auto delta = rng.Placement();
         const auto reference = ToManifold(f.Get(id));
         const PositionOperationTarget target{.StoreId = id, .Selection = AllHandles(f.Get(id))};
         const bool changed = ExecuteGeometryPositions(f.Scene, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = delta}}).has_value();
         f.Audit();
         const auto column = [&](vec3 axis) {
             const auto c = delta.R * (delta.S * axis);
             return manifold::vec3{c.x, c.y, c.z};
         };
         const manifold::mat3x4 matrix{column({1, 0, 0}), column({0, 1, 0}), column({0, 0, 1}), {delta.P.x, delta.P.y, delta.P.z}};
         return std::pair{changed ? std::optional{id} : std::nullopt, reference.Transform(matrix)};
     },
     Match::Same},
    // Planar-faced solids are convex to both sides, so each answers with one hull of the vertex sums, keeping nearly coplanar vertices by its own tolerance.
    {"minkowski sum", [](Random &rng) { return Placed(rng.Shape(true), rng.Placement()); }, [](GeometryFixture &f, Random &rng, uint32_t a) {
         // The brush contains the origin, which keeps manifold's union with the first operand inside the hull of the sums.
         auto brush = rng.Placement();
         brush.P = {};
         brush.S *= .3f;
         const auto b = f.Create(Placed(rng.Shape(true), brush));
         return std::pair{MinkowskiRecord(f.Scene, f.Get(a), f.Get(b), MinkowskiOperation::Sum), ToManifold(f.Get(a)).MinkowskiSum(ToManifold(f.Get(b)))}; }, Match::Subset},
    // Every triangle of the solid with its own three vertices welds back into the solid.
    {"merge by distance", RandomSolid, [](GeometryFixture &f, Random &, uint32_t whole_id) {
         const auto whole = f.Get(whole_id);
         MeshData soup;
         manifold::MeshGL64 gl;
         gl.numProp = 3;
         for (const auto triangle : whole.DerivedTriangles()) {
             const auto first = uint32_t(soup.Positions.size());
             for (const auto p : test::TrianglePositions(whole, triangle)) {
                 soup.Positions.push_back(p);
                 gl.vertProperties.insert(gl.vertProperties.end(), {p.x, p.y, p.z});
                 gl.triVerts.push_back(uint32_t(gl.triVerts.size()));
             }
             soup.AddFace(std::array{first, first + 1u, first + 2u});
         }
         const auto id = f.Create(std::move(soup));
         f.Run(*MergeTask(f.Scene, f.Get(id), AllHandles(f.Get(id)), GeometryMergeMode::ByDistance, 1e-5f, InvalidOffset));
         expect(gl.Merge()) << "reference merged nothing";
         const manifold::Manifold reference{gl};
         expect(reference.Status() == manifold::Manifold::Error::NoError) << int(reference.Status());
         return std::pair{std::optional{id}, reference};
     },
     Match::Same},
    // A lone face extrudes into a closed prism with its original as the far cap, which is manifold's extrusion.
    // The circle lies in the XZ plane, and the disc is its polygon in the XY plane, counterclockwise.
    {"extrusion", [](Random &rng) {
         const auto circle = primitive::CreateMesh(primitive::Circle{.Segments = 5u + rng.Below(8), .Fill = primitive::FillType::Face});
         MeshData data;
         for (const auto p : circle.Positions) data.Positions.emplace_back(p.x, p.z, 0.f);
         if (SignedArea(data.Positions) < 0.0) std::ranges::reverse(data.Positions);
         std::vector<uint32_t> loop(data.Positions.size());
         std::ranges::iota(loop, 0u);
         data.AddFace(loop);
         return data; }, [](GeometryFixture &f, Random &rng, uint32_t id) {
         const auto height = rng.Uniform(.3f, 2.f);
         const auto disc = f.Get(id);
         manifold::SimplePolygon polygon;
         for (const auto v : test::FaceVertices(disc, disc.FaceAt(0))) polygon.emplace_back(test::Position(disc, v).x, test::Position(disc, v).y);
         f.Run(ExtrudeStepsTask(disc, AllHandles(disc), Element::Face, 1u, mat3{1.f}, {0.f, 0.f, height}, {}));
         return std::pair{std::optional{id}, manifold::Manifold::Extrude({polygon}, height)}; }, Match::Same},
    // Manifold also removes the flat vertices a second cut leaves on the first cut's edges, which ours keep as collinear vertices.
    {"chained booleans", RandomSolid, [](GeometryFixture &f, Random &rng, uint32_t a) {
         const auto b = f.Create(RandomSolid(rng)), c = f.Create(RandomSolid(rng));
         const auto reference = ToManifold(f.Get(a)).Boolean(ToManifold(f.Get(b)), manifold::OpType::Add).Boolean(ToManifold(f.Get(c)), manifold::OpType::Subtract);
         const auto joined = RunBoolean(f, a, b, BooleanOperation::Union);
         return std::pair{joined ? RunBoolean(f, *joined, c, BooleanOperation::Difference) : std::nullopt, reference};
     },
     Match::Solid},
}};

// A planar operation on seeded inputs, run by ours and by manifold, whose results compare as polygons: the same loops, points and area.
// Run returns our result's loops in the XY plane and manifold's polygons, and Subset admits reference points ours lacks.
struct PolygonRow {
    std::string_view Name;
    std::pair<std::optional<std::vector<std::vector<vec2>>>, manifold::Polygons> (*Run)(GeometryFixture &, Random &);
    bool Subset;
};
std::array<std::pair<MeshData, manifold::SimplePolygon>, 2> RandomPolygonPair(Random &rng) {
    const auto polygon = [&](vec2 center, float radius) {
        const auto sides = 3u + rng.Below(6u);
        return RegularPolygon(sides, center, radius, rng.Uniform(0.f, 6.28f));
    };
    auto a = polygon({rng.Uniform(-.3f, .3f), rng.Uniform(-.3f, .3f)}, rng.Uniform(.6f, 1.2f));
    auto b = polygon({rng.Uniform(-.8f, .8f), rng.Uniform(-.8f, .8f)}, rng.Uniform(.4f, 1.f));
    return {std::move(a), std::move(b)};
}
// A region is a coplanar face set whose holes wind against their outer loops, which is manifold's polygon model.
template<RegionFill Fill, manifold::OpType ReferenceOperation> std::pair<std::optional<std::vector<std::vector<vec2>>>, manifold::Polygons> RegionBoolean(GeometryFixture &f, Random &rng) {
    const auto [a, b] = RandomPolygonPair(rng);
    const auto a_id = f.Create(a.first), b_id = f.Create(b.first);
    const manifold::CrossSection section_a{a.second}, section_b{b.second};
    return {RegionLoops(f.Scene, PlaneAt({}, {0.f, 0.f, 1.f}), std::array{RegionOperand{a_id}, RegionOperand{b_id}}, Fill), section_a.Boolean(section_b, ReferenceOperation).ToPolygons()};
}
// A convex erosion is the intersection of the region's translates against each brush vertex, which a brush off the origin shifts.
std::pair<std::optional<std::vector<std::vector<vec2>>>, manifold::Polygons> Eroded(GeometryFixture &f, Random &rng, const std::pair<MeshData, manifold::SimplePolygon> &brush) {
    const auto [c_data, c_polygon] = RegularPolygon(5u + rng.Below(4u), {rng.Uniform(-.3f, .3f), rng.Uniform(-.3f, .3f)}, rng.Uniform(.8f, 1.2f), rng.Uniform(0.f, 6.28f));
    const auto c = f.Create(c_data), brush_id = f.Create(brush.first);
    const auto difference = MinkowskiRecord(f.Scene, f.Get(c), f.Get(brush_id), MinkowskiOperation::Difference);
    std::vector<manifold::CrossSection> translates;
    for (const auto v : brush.second) translates.emplace_back(manifold::CrossSection{c_polygon}.Translate(-v));
    return {difference ? std::optional{PlanarLoops(f.Get(*difference))} : std::nullopt, manifold::CrossSection::BatchBoolean(translates, manifold::OpType::Intersect).ToPolygons()};
}
const std::array<PolygonRow, 10> PolygonRows{{
    // Manifold slices the triangulation, so its loops also carry the crossings of face diagonals, which makes ours a subset.
    {"cross section", [](GeometryFixture &f, Random &rng) {
         const auto placement = rng.Placement();
         const auto id = f.Create(Placed(rng.Shape(true), placement));
         const auto polygons = ToManifold(f.Get(id)).Slice(placement.P.z);
         const vec3 point{0.f, 0.f, placement.P.z}, normal{0.f, 0.f, 1.f};
         // Cross Section's run: the side in front cleared, the cut capped, then the vertices behind the plane deleted.
         auto section_id = Capped(f, Bisected(f, id, point, normal, PlaneSideKind::Positive), point, normal);
         if (const auto clear = PlaneSideDeleteTask(f.Scene, f.Get(section_id), point, normal, PlaneSideKind::Negative, PlaneSideTolerance)) section_id = f.Run(*clear).GeometryId;
         return std::pair{std::optional{PlanarLoops(f.Get(section_id))}, polygons};
     },
     true},
    {"union", RegionBoolean<RegionFill::Union, manifold::OpType::Add>, false},
    {"difference", RegionBoolean<RegionFill::Difference, manifold::OpType::Subtract>, false},
    {"intersection", RegionBoolean<RegionFill::Intersection, manifold::OpType::Intersect>, false},
    // Both polygons in one record fill where either winds positive.
    {"positive fill", [](GeometryFixture &f, Random &rng) {
         const auto [a, b] = RandomPolygonPair(rng);
         MeshData joined = a.first;
         Append(joined, b.first);
         const auto both = f.Create(std::move(joined));
         return std::pair{RegionLoops(f.Scene, PlaneAt({}, {0.f, 0.f, 1.f}), std::array{RegionOperand{both}}, RegionFill::Positive), manifold::CrossSection{manifold::Polygons{a.second, b.second}}.ToPolygons()};
     },
     false},
    // The silhouette along z is the loops of halfedges whose face faces +z and whose opposite face does not, filled where they wind positive, which is manifold's projection.
    {"projection", [](GeometryFixture &f, Random &rng) {
         const auto id = f.Create(Placed(rng.Shape(true), rng.Placement()));
         return std::pair{RegionLoops(f.Scene, PlaneAt({}, {0.f, 0.f, 1.f}), std::array{RegionOperand{.Record = id, .Silhouette = true}}, RegionFill::Positive), manifold::CrossSection{ToManifold(f.Get(id)).Project()}.ToPolygons()};
     },
     false},
    {"region hull", [](GeometryFixture &f, Random &rng) {
         MeshData data;
         manifold::Polygons polygons;
         for (uint32_t k = 0u; k < 2u; ++k) {
             const auto sides = 3u + rng.Below(6u);
             const vec2 center{rng.Uniform(-.8f, .8f), rng.Uniform(-.8f, .8f)};
             const auto radius = rng.Uniform(.3f, 1.f), phase = rng.Uniform(0.f, 6.28f);
             auto [polygon_data, polygon] = RegularPolygon(sides, center, radius, phase);
             Append(data, polygon_data);
             polygons.push_back(std::move(polygon));
         }
         const auto id = f.Create(std::move(data));
         const auto tasks = ConvexHullTasks(f.Scene, f.Get(id), AllHandles(f.Get(id)));
         for (const auto &task : tasks) f.Run(task);
         return std::pair{tasks.empty() ? std::nullopt : std::optional{PlanarLoops(f.Get(id))}, manifold::CrossSection::Hull(polygons).ToPolygons()};
     },
     false},
    // Prisms over two regions sum to a prism over their plane sum, whose middle slice is that sum.
    {"region sum", [](GeometryFixture &f, Random &rng) {
         const auto [a_data, a_polygon] = RegularPolygon(6u + 2u * rng.Below(4u), {rng.Uniform(-.3f, .3f), rng.Uniform(-.3f, .3f)}, rng.Uniform(.8f, 1.2f), rng.Uniform(0.f, 6.28f), true);
         const auto [b_data, b_polygon] = RegularPolygon(4u + 2u * rng.Below(3u), {}, rng.Uniform(.1f, .25f), rng.Uniform(0.f, 6.28f));
         const auto a = f.Create(a_data), b = f.Create(b_data);
         const auto sum = MinkowskiRecord(f.Scene, f.Get(a), f.Get(b), MinkowskiOperation::Sum);
         const auto prism = manifold::Manifold::Extrude({a_polygon}, 1.0).MinkowskiSum(manifold::Manifold::Extrude({b_polygon}, 1.0));
         return std::pair{sum ? std::optional{PlanarLoops(f.Get(*sum))} : std::nullopt, manifold::CrossSection{prism.Slice(1.0)}.ToPolygons()};
     },
     true},
    // The brush is symmetric about the origin, so eroding by it or by its reflection is the same.
    {"region difference", [](GeometryFixture &f, Random &rng) { return Eroded(f, rng, RegularPolygon(4u + 2u * rng.Below(3u), {}, rng.Uniform(.1f, .25f), rng.Uniform(0.f, 6.28f))); }, false},
    {"region difference off origin", [](GeometryFixture &f, Random &rng) { return Eroded(f, rng, RegularPolygon(4u + 2u * rng.Below(3u), {rng.Uniform(.1f, .2f), rng.Uniform(-.2f, -.1f)}, rng.Uniform(.1f, .2f), rng.Uniform(0.f, 6.28f))); }, false},
}};
} // namespace

void RegisterManifoldOracleTests() {
    "operators on seeded inputs match manifold"_test = [] {
        GeometryFixture f;
        for (const auto &row : SolidRows) {
            boost::ut::test(std::string{row.Name}) = [&] {
                for (uint32_t seed = 0; seed < Seeds; ++seed) {
                    Random rng{std::mt19937{seed}};
                    const auto label = Label(row.Name, seed);
                    const auto [id, reference] = row.Run(f, rng, f.Create(row.Input(rng)));
                    expect(id.has_value()) << label;
                    if (!id) continue;
                    const auto mesh = f.Get(*id);
                    if (row.Mode == Match::Solid) {
                        ExpectSameBoolean(mesh, reference, label);
                        continue;
                    }
                    const auto actual = ShapeOf(mesh);
                    expect(mesh.HasClosedSurface()) << label << "closed";
                    expect(mesh.VertexCount() == actual.Points.size()) << label << "vertices" << mesh.VertexCount() << "on faces" << actual.Points.size();
                    ExpectSame(actual, ShapeOf(reference), label, row.Mode == Match::Subset);
                }
            };
        }
    };
    // Coincident placements put faces and vertices on each other's planes, which random placements never do.
    // A boolean with an open input plans nothing.
    "booleans of coincident solids match manifold"_test = [] {
        GeometryFixture f;
        const std::array<std::pair<std::string_view, MeshData>, 7> seconds{{
            {"overlapping", Placed(primitive::Cube{}, {.P = vec3{1.f}})},
            {"sphere", Placed(primitive::UVSphere{.Radius = .75f}, {.P = {1.f, 0.f, 0.f}})},
            {"flush", Placed(primitive::Cube{}, {.P = {2.f, 0.f, 0.f}})},
            {"offset", Placed(primitive::Cube{}, {.P = {2.f, .5f, .5f}})},
            {"disjoint", Placed(primitive::Cube{}, {.P = {5.f, 0.f, 0.f}})},
            {"nested", Placed(primitive::Cube{}, {.S = vec3{.5f}})},
            {"enclosing", Placed(primitive::Cube{}, {.S = vec3{2.f}})},
        }};
        for (const auto &[name, second] : seconds) {
            for (const auto [operation, reference_operation, operation_name] : {std::tuple{BooleanOperation::Union, manifold::OpType::Add, "union"}, std::tuple{BooleanOperation::Difference, manifold::OpType::Subtract, "difference"}, std::tuple{BooleanOperation::Intersection, manifold::OpType::Intersect, "intersection"}}) {
                const auto label = std::format("{} {}", name, operation_name);
                const auto [id, reference] = MatchBoolean(f, f.Create(primitive::CreateMesh(primitive::Cube{})), f.Create(second), operation, reference_operation, label);
                expect(id.has_value()) << label;
                if (id) ExpectSameBoolean(f.Get(*id), reference, label);
            }
        }
        const auto cube = f.Create(primitive::CreateMesh(primitive::Cube{})), quad = f.Create(UnitQuad());
        expect(!BooleanTask(f.Get(cube), f.Get(quad), BooleanOperation::Union, mat3{1.f}, vec3{0.f}).has_value());
        expect(!BooleanTask(f.Get(quad), f.Get(cube), BooleanOperation::Intersection, mat3{1.f}, vec3{0.f}).has_value());
    };
    // The hulls of two adjacent triangles swept by a cube share the swept edge's faces exactly, which only the moved-vertex ties resolve.
    "unions of adjacent sweep hulls match manifold"_test = [] {
        GeometryFixture f;
        const auto brush = primitive::CreateMesh(primitive::Cube{});
        std::vector<vec3> b_points;
        for (const auto p : brush.Positions) b_points.push_back(p * .15f);
        const auto hull_record = [&](std::span<const vec3> tri) {
            std::vector<vec3> sums;
            for (const auto p : tri) {
                for (const auto q : b_points) sums.push_back(p + q);
            }
            mtl::ComputeChain chain{f.Store().BufferContext()};
            const auto positions = chain.Scratch.Allocate(3u * uint32_t(sums.size() + 1u));
            chain.Scratch.Buffer.Update(std::as_bytes(std::span{sums}), uint64_t(positions.Offset) * sizeof(uint32_t));
            MeshData data;
            std::vector<uint32_t> index(sums.size(), InvalidOffset);
            std::vector<uint32_t> loop;
            for (const auto &face : ConvexHulls(f.Scene, chain, positions.Offset, std::array{0u, uint32_t(sums.size())}).front()) {
                loop.clear();
                for (const auto v : face) {
                    if (index[v] == InvalidOffset) {
                        index[v] = uint32_t(data.Positions.size());
                        data.Positions.push_back(sums[v]);
                    }
                    loop.push_back(index[v]);
                }
                data.AddFace(loop);
            }
            return f.Create(std::move(data));
        };
        const auto torus = f.Get(f.Create(primitive::CreateMesh(primitive::Torus{.MajorSegments = 8, .MinorSegments = 4})));
        std::vector<std::array<vec3, 3>> tris;
        for (const auto c : torus.DerivedTriangles()) tris.push_back(test::TrianglePositions(torus, c));
        for (uint32_t i = 0; i + 1 < 12; i += 2) {
            const auto label = Label("sweep pair", i);
            const auto h1 = hull_record(tris[i]), h2 = hull_record(tris[i + 1]);
            const auto reference = ToManifold(f.Get(h1)).Boolean(ToManifold(f.Get(h2)), manifold::OpType::Add);
            const auto id = RunBoolean(f, h1, h2, BooleanOperation::Union);
            expect(id.has_value()) << label;
            if (!id) continue;
            expect(f.Get(*id).HasClosedSurface()) << label << "closed";
            ExpectSameMeasures(ShapeOf(f.Get(*id)), ShapeOf(reference), label);
        }
    };
    "join and separate match compose and decompose"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            auto first = Placed(rng.Shape(), rng.Placement());
            auto second = Placed(rng.Shape(), rng.Placement());
            for (auto &p : second.Positions) p.x += 5.f;
            const auto reference = ToManifold(f.Get(f.Create(first))).Boolean(ToManifold(f.Get(f.Create(second))), manifold::OpType::Add).Decompose();
            Append(first, second);
            const auto joined = f.Create(std::move(first));
            expect(test::ClosedComponentsOutward(f.Get(joined))) << Label("outward", seed);
            const auto tasks = SeparateGeometryTasks(f.Scene, f.Get(joined), AllHandles(f.Get(joined)), GeometrySeparateMode::LooseParts);
            const auto results = ExecuteGeometryTopology(f.Scene, tasks);
            f.Audit();
            std::vector<Shape> actual;
            for (const auto &result : results) actual.push_back(ShapeOf(f.Get(result.GeometryId)));
            std::vector<Shape> expected;
            for (const auto &part : reference) expected.push_back(ShapeOf(part));
            const auto by_volume = [](const Shape &a, const Shape &b) { return a.Volume < b.Volume; };
            std::ranges::sort(actual, by_volume);
            std::ranges::sort(expected, by_volume);
            expect(actual.size() == expected.size()) << Label("parts", seed) << actual.size() << "vs" << expected.size();
            for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) ExpectSame(actual[i], expected[i], Label("part", seed));
        }
    };
    // Bisect Fill clearing either side keeps the half manifold splits off on the other.
    "filled bisects match split by plane"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto placement = rng.Placement();
            // A plane cut through a non-planar face splits its volume by the face's triangulation, so every face is planar here.
            const auto data = Placed(rng.Shape(true), placement);
            const auto normal = Normalize(vec3{rng.Uniform(-1.f, 1.f), rng.Uniform(-1.f, 1.f), rng.Uniform(-1.f, 1.f)});
            const auto point = placement.P + vec3{rng.Uniform(-.3f, .3f), rng.Uniform(-.3f, .3f), rng.Uniform(-.3f, .3f)};
            const auto [front, back] = ToManifold(f.Get(f.Create(data))).SplitByPlane({normal.x, normal.y, normal.z}, Dot(normal, point));
            for (const auto [cleared, half, label] : {std::tuple{PlaneSideKind::Negative, &front, "front"}, std::tuple{PlaneSideKind::Positive, &back, "back"}}) {
                const auto kept = Capped(f, Bisected(f, f.Create(data), point, normal, cleared), point, normal);
                if (half->IsEmpty()) expect(f.Get(kept).FaceCount() == 0u) << Label(label, seed) << "expected empty";
                else ExpectSame(ShapeOf(f.Get(kept)), ShapeOf(*half), Label(label, seed), true);
            }
        }
    };
    "simplify within a tolerance removes refinement as manifold does"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto data = Placed(rng.Shape(true), rng.Placement());
            const auto id = f.Create(data), before = f.Create(data);
            const auto cuts = 1u + rng.Below(2);
            for (const auto record : {id, before}) f.Run({.SourceId = record, .Op = MeshTopologyOp::Subdivide, .Param0 = float(cuts), .Flags = TopologyFlagSelectAll});
            const auto reference = ToManifold(f.Get(id));
            SimplifyGeometry(f.Scene, id, AllHandles(f.Get(id)), 1e-3f, 1.f, [&](const MeshTopologyTask &task) { f.Run(task); });
            // Manifold promises a vertex subset within the tolerance rather than the smallest one, and its greedy pass leaves some cap vertices that the quadric pass removes.
            const auto actual = ShapeOf(f.Get(id)), simplified = ShapeOf(reference.Simplify(1e-3));
            const auto label = Label(std::format("simplify after {} cuts", cuts), seed);
            expect(test::SimplifiedWithin(f.Get(id), f.Get(before), f.Get(before).FaceCount() - 1u, 1e-3f)) << label << "faces" << f.Get(id).FaceCount() << "of" << f.Get(before).FaceCount();
            expect(actual.Points.size() <= simplified.Points.size()) << label << "points" << actual.Points.size() << "vs" << simplified.Points.size();
            ExpectSame(actual, simplified, label, true);
        }
    };
    "planar operations on seeded inputs match manifold's polygons"_test = [] {
        GeometryFixture f;
        for (const auto &row : PolygonRows) {
            boost::ut::test(std::string{row.Name}) = [&] {
                for (uint32_t seed = 0; seed < Seeds; ++seed) {
                    Random rng{std::mt19937{seed}};
                    const auto label = Label(row.Name, seed);
                    const auto [loops, reference] = row.Run(f, rng);
                    expect(loops.has_value()) << label;
                    if (loops) ExpectSamePolygons(*loops, reference, label, row.Subset);
                }
            };
        }
    };
    // Region simplification keeps a subset of the vertices and the area within the tolerance times the perimeter.
    "simplified regions stay within tolerance"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            constexpr uint32_t Sides = 64u;
            constexpr float Tolerance = .05f;
            MeshData data;
            manifold::SimplePolygon polygon;
            std::vector<uint32_t> face;
            std::vector<dvec3> input;
            double perimeter = 0.0;
            for (uint32_t i = 0u; i < Sides; ++i) {
                const float angle = 2.f * std::numbers::pi_v<float> * float(i) / float(Sides), radius = 1.f + rng.Uniform(-.004f, .004f);
                const vec2 p = radius * vec2{std::cos(angle), std::sin(angle)};
                face.push_back(i);
                data.Positions.emplace_back(p.x, p.y, 0.f);
                polygon.emplace_back(p.x, p.y);
                input.emplace_back(p.x, p.y, 0.0);
            }
            for (uint32_t i = 0u; i < Sides; ++i) perimeter += Length(input[(i + 1) % Sides] - input[i]);
            const auto input_area = SignedArea(input);
            data.AddFace(face);
            const auto id = f.Create(std::move(data));
            const auto label = Label("simplified region", seed);
            expect(SimplifyGeometry(f.Scene, id, AllHandles(f.Get(id)), Tolerance, 1.f, [&](const MeshTopologyTask &task) { f.Run(task); })) << label << "collapsed";
            const auto mesh = f.Get(id);
            expect(mesh.FaceCount() == 1u) << label << "faces" << mesh.FaceCount();
            std::vector<dvec3> output;
            for (const auto face : mesh.faces()) {
                for (const auto v : test::FaceVertices(mesh, face)) output.emplace_back(test::Position(mesh, v));
            }
            for (const auto p : output) expect(std::ranges::any_of(input, [&](dvec3 point) { return Length(point - p) < 1e-6; })) << label << "vertex" << p.x << p.y << "is new";
            const auto area = SignedArea(output);
            const auto reference = manifold::CrossSection{polygon}.Simplify(Tolerance).ToPolygons();
            expect(output.size() < Sides && !reference.empty() && output.size() >= reference.front().size()) << label << "vertices" << output.size() << "of" << Sides << "vs" << (reference.empty() ? 0 : reference.front().size());
            expect(std::abs(area - input_area) <= Tolerance * perimeter) << label << "area" << area << "vs" << input_area;
        }
    };
    "split faces triangulates polygons as triangulate does"_test = [] {
        GeometryFixture f;
        for (auto polygon : test::TriangulationPolygons()) {
            const auto loop = polygon.Positions;
            const auto id = f.Create(std::move(polygon));
            f.Run({.SourceId = id, .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = AllHandles(f.Get(id))});
            const auto mesh = f.Get(id);
            test::Triangles actual;
            for (const auto face : mesh.faces()) {
                std::array<uint32_t, 3> triangle{};
                uint32_t k = 0u;
                for (const auto v : test::FaceVertices(mesh, face)) triangle[k++] = uint32_t(std::ranges::find(loop, test::Position(mesh, v)) - loop.begin());
                actual.push_back(test::Canonical(triangle));
            }
            std::ranges::sort(actual);
            expect(actual == test::ManifoldTriangles(loop)) << loop.size() << "corners";
        }
    };
    // Welding a full revolve's seam leaves one coincident cap as a membrane across the ring, and deleting it gives manifold's revolve.
    "full revolves with a welded seam match revolve"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto slices = 5u + rng.Below(8), steps = 6u + rng.Below(12);
            const auto offset = rng.Uniform(1.5f, 3.f);
            // The circle lies in the XZ plane, which is the half plane manifold revolves around Z.
            auto data = primitive::CreateMesh(primitive::Circle{.Segments = slices, .Fill = primitive::FillType::Face});
            for (auto &p : data.Positions) p.x += offset;
            const auto id = f.Create(std::move(data));
            const auto disc = f.Get(id);
            manifold::SimplePolygon polygon;
            for (const auto v : test::FaceVertices(disc, disc.FaceAt(0))) polygon.emplace_back(test::Position(disc, v).x, test::Position(disc, v).z);
            if (SignedArea(polygon) < 0.0) std::ranges::reverse(polygon);
            const auto reference = manifold::Manifold::Revolve({polygon}, int(steps));
            f.Run(ExtrudeStepsTask(disc, AllHandles(disc), Element::Face, steps, ToMat3(AngleAxis(2.f * std::numbers::pi_v<float> / float(steps), vec3{0.f, 0.f, 1.f})), {}, {}));
            f.Run(*MergeTask(f.Scene, f.Get(id), AllHandles(f.Get(id)), GeometryMergeMode::ByDistance, 1e-5f, InvalidOffset));
            const auto welded = f.Get(id);
            std::vector<std::pair<uint32_t, std::vector<uint32_t>>> loops;
            for (const auto face : welded.faces()) {
                auto &vertices = loops.emplace_back(*face, std::vector<uint32_t>{}).second;
                for (const auto v : test::FaceVertices(welded, face)) vertices.push_back(*v);
            }
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_faces;
            for (const auto &[_, vertices] : loops) {
                for (size_t i = 0; i < vertices.size(); ++i) ++edge_faces[std::minmax(vertices[i], vertices[(i + 1) % vertices.size()])];
            }
            GeometrySelection membrane;
            for (const auto &[face, vertices] : loops) {
                bool interior = true;
                for (size_t i = 0; i < vertices.size(); ++i) interior &= edge_faces.at(std::minmax(vertices[i], vertices[(i + 1) % vertices.size()])) > 2u;
                if (interior) membrane.Faces.push_back(face);
            }
            expect(membrane.Faces.size() == 1u) << Label("revolve", seed) << "membranes" << membrane.Faces.size();
            f.Run({.SourceId = id, .Op = MeshTopologyOp::DeleteOnlyFaces, .SelectionElement = Element::Face, .Selection = membrane});
            ExpectSame(ShapeOf(f.Get(id)), ShapeOf(reference), Label("revolve", seed));
        }
    };
    // Both march the same lattice with linearly interpolated crossings, and manifold also snaps lattice points near the surface onto it, so the two agree in measure.
    "level sets match manifold in measure"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto label = Label("level set", seed);
            const auto radius = rng.Uniform(.5f, .9f);
            const vec3 center{rng.Uniform(-.2f, .2f), rng.Uniform(-.2f, .2f), rng.Uniform(-.2f, .2f)};
            const auto edge = rng.Uniform(.1f, .2f);
            const vec3 min{-1.2f, -1.2f, -1.2f}, max{1.2f, 1.2f, 1.2f};
            const auto lattice = LatticeOver(min, max, edge);
            expect(lattice.has_value()) << label;
            if (!lattice) continue;
            std::vector<float> values;
            for (const auto p : lattice->Points()) values.push_back(radius - Length(p - center));
            const auto id = f.Create(MeshData{lattice->Points()});
            const auto attribute = f.Store().AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "field").value();
            f.Store().SetAttribute<float>(attribute, 0u, values);
            const auto result = f.Run(LevelSetTask(f.Store(), id, *lattice, attribute, 0.f));
            expect(result.Changed) << label << "changed";
            const auto surface = f.Get(result.GeometryId);
            expect(surface.HasClosedSurface()) << label << "closed";
            const auto actual = ShapeOf(surface);
            const auto reference = ShapeOf(manifold::Manifold::LevelSet([&](manifold::vec3 p) { return double(radius) - Length(vec3{float(p.x), float(p.y), float(p.z)} - center); }, manifold::Box{{min.x, min.y, min.z}, {max.x, max.y, max.z}}, edge));
            expect(std::abs(actual.Volume - reference.Volume) <= 2e-2 * reference.Volume) << label << "volume" << actual.Volume << "vs" << reference.Volume;
            expect(std::abs(actual.Area - reference.Area) <= 2e-2 * reference.Area) << label << "area" << actual.Area << "vs" << reference.Area;
            expect(actual.Euler == reference.Euler) << label << "euler" << actual.Euler << "vs" << reference.Euler;
        }
    };
    // Booleans of the same hulls place their crossing vertices by their own arithmetic, so the swept results agree in measure.
    "minkowski sweeps of faces match manifold in measure"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto torus = f.Create(Placed(primitive::Torus{.MajorSegments = 8, .MinorSegments = 4}, rng.Placement()));
            const auto cube = f.Create(Placed(primitive::Cube{}, {.P = {}, .R = rng.Rotation(), .S = vec3{.15f}}));
            const auto sum = MinkowskiRecord(f.Scene, f.Get(torus), f.Get(cube), MinkowskiOperation::Sum);
            expect(sum.has_value()) << Label("torus sum", seed);
            if (sum) ExpectSameMeasures(ShapeOf(f.Get(*sum)), ShapeOf(ToManifold(f.Get(torus)).MinkowskiSum(ToManifold(f.Get(cube)))), Label("torus sum", seed));
            const auto block = f.Create(Placed(primitive::Cube{}, rng.Placement()));
            // Manifold's own Minkowski difference returns empty for some placements, so the reference is that intersection.
            const auto offset_cube = f.Create(Placed(primitive::Cube{}, {.P = {.2f, .1f, -.15f}, .R = rng.Rotation(), .S = vec3{.15f}}));
            const auto block_solid = ToManifold(f.Get(block));
            for (const auto [brush, label] : {std::pair{cube, "block difference"}, std::pair{offset_cube, "block difference off origin"}}) {
                const auto difference = MinkowskiRecord(f.Scene, f.Get(block), f.Get(brush), MinkowskiOperation::Difference);
                expect(difference.has_value()) << Label(label, seed);
                const auto brush_mesh = f.Get(brush);
                std::vector<manifold::Manifold> translates;
                for (const auto v : brush_mesh.vertices()) {
                    const auto p = test::Position(brush_mesh, v);
                    translates.push_back(block_solid.Translate({-p.x, -p.y, -p.z}));
                }
                const auto erosion = manifold::Manifold::BatchBoolean(translates, manifold::OpType::Intersect);
                if (difference) ExpectSameMeasures(ShapeOf(f.Get(*difference)), ShapeOf(erosion), Label(label, seed));
            }
        }
    };
    "curvature matches manifold"_test = [] {
        GeometryFixture f;
        for (uint32_t seed = 0; seed < Seeds; ++seed) {
            Random rng{std::mt19937{seed}};
            const auto id = f.Create(Placed(rng.Shape(), rng.Placement()));
            const auto mesh = f.Get(id);
            const auto gl = ToManifold(mesh).CalculateCurvature(0, 1).GetMeshGL64();
            expect(gl.numProp == 5u) << Label("curvature", seed) << "properties" << gl.numProp;
            std::vector<dvec3> points;
            for (size_t v = 0; v < gl.NumVert(); ++v) points.emplace_back(gl.vertProperties[v * gl.numProp], gl.vertProperties[v * gl.numProp + 1], gl.vertProperties[v * gl.numProp + 2]);
            for (const auto kind : {CurvatureKind::Gaussian, CurvatureKind::Mean}) {
                const auto label = Label(kind == CurvatureKind::Mean ? "mean curvature" : "gaussian curvature", seed);
                const auto actual = f.Store().AttributeView<float>(CurvatureField(f.Scene, id, kind, label).value());
                const auto column = kind == CurvatureKind::Mean ? 4u : 3u;
                double scale = 1.0;
                for (size_t v = 0; v < gl.NumVert(); ++v) scale = std::max(scale, std::abs(gl.vertProperties[v * gl.numProp + column]));
                for (const auto v : mesh.vertices()) {
                    const dvec3 p{test::Position(mesh, v)};
                    size_t nearest = 0;
                    for (size_t i = 1; i < points.size(); ++i) {
                        if (Length(points[i] - p) < Length(points[nearest] - p)) nearest = i;
                    }
                    const double reference = gl.vertProperties[nearest * gl.numProp + column];
                    expect(std::abs(double(actual[*v]) - reference) <= 2e-3 * scale) << label << "vertex" << *v << actual[*v] << "vs" << reference;
                }
            }
        }
    };
}
