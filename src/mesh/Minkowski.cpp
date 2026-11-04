#include "mesh/Minkowski.h"
#include "gpu/ConvexHull.h"
#include "mesh/ConvexHull.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshCreate.h"
#include "mesh/Neighborhood.h"
#include "mesh/Region.h"
#include "mesh/ScratchPasses.h"
#include "mesh/TopologyOperations.h"
#include "state/Registry.h"

namespace {
struct Geometry : FaceLoops {
    std::vector<vec3> Normals;

    Geometry(state::Registry &r, const Mesh &mesh)
        : FaceLoops{ReadFaceLoops(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), mesh)},
          Normals{FaceNormals(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), mesh.GetStoreId(), Faces)} {}
    bool Convex(const Mesh &mesh) const {
        if (int64_t(mesh.VertexCount()) - int64_t(mesh.EdgeCount()) + int64_t(mesh.FaceCount()) != 2) return false;
        float extent = 0.f;
        for (const auto v : mesh.vertices()) extent = std::max(extent, Length(Positions[*v]));
        const float tolerance = 1e-5f * std::max(1.f, extent);
        for (uint32_t f = 0u; f < Faces.size(); ++f) {
            const auto origin = Positions[Corners.Of(f).front().Vertex];
            for (const auto v : mesh.vertices()) {
                if (Dot(Normals[f], Positions[*v] - origin) > tolerance) return false;
            }
        }
        return true;
    }
    bool ConvexRegion(const RegionPlane &plane) const {
        if (Faces.size() != 1u) return false;
        const auto corners = Corners.Of(0u);
        for (size_t i = 0; i < corners.size(); ++i) {
            const auto a = Positions[corners[i].Vertex], b = Positions[corners[(i + 1) % corners.size()].Vertex], c = Positions[corners[(i + 2) % corners.size()].Vertex];
            if (Dot(Cross(b - a, c - b), plane.Normal) < 0.f) return false;
        }
        return true;
    }
    std::vector<uint32_t> EdgeHandles() const {
        std::vector<uint32_t> handles;
        for (uint32_t f = 0u; f < Faces.size(); ++f) {
            const auto corners = Corners.Of(f);
            for (size_t i = 0; i < corners.size(); ++i) handles.insert(handles.end(), {corners[i].Vertex, corners[(i + 1) % corners.size()].Vertex});
        }
        return handles;
    }
    vec3 TriangleNormal(std::span<const uint32_t> handles) const {
        const auto a = Positions[handles[0]];
        return Normalize(Cross(Positions[handles[1]] - a, Positions[handles[2]] - a));
    }
    vec3 Centroid(const Mesh &mesh) const {
        vec3 sum{};
        for (const auto v : mesh.vertices()) sum += Positions[*v];
        return mesh.VertexCount() ? sum / float(mesh.VertexCount()) : sum;
    }
};
void Translate(state::Registry &r, uint32_t id, vec3 offset) {
    const PositionOperationTarget target{.StoreId = id, .Selection = AllHandles(Mesh{r.Resources.get<const MeshStore>(), id})};
    ExecuteGeometryPositions(r, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = offset}}});
}
// With `reflect_about`, the sums take the second mesh reflected through that point, which is the brush an erosion subtracts.
std::vector<uint32_t> HullRecords(state::Registry &r, const Mesh &first, const Mesh &second, std::span<const uint32_t> first_handles, std::span<const uint32_t> second_handles, std::span<const HullSumSet> sets, std::span<const uint32_t> first_point, std::optional<vec3> reflect_about, vec3 facing) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto points = first_point.back();
    const auto positions = chain.Scratch.Allocate(3u * (points + uint32_t(sets.size())));
    const HullPointsPushConstants pc{
        .Positions = {chain.Scratch.Buffer.Slot, positions.Offset},
        .Count = points,
        .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
        .First = meshes.GetConnectivityRef(first.GetStoreId()),
        .Second = meshes.GetConnectivityRef(second.GetStoreId()),
        .FirstHandles = chain.Upload(std::as_bytes(first_handles)),
        .SecondHandles = chain.Upload(std::as_bytes(second_handles)),
        .Sets = chain.Upload(std::as_bytes(sets)),
        .SetCount = uint32_t(sets.size()),
        .SecondSign = reflect_about ? -1.f : 1.f,
        .SecondOffset = reflect_about.value_or(vec3{}),
    };
    chain.Threads(GetMeshPipelines(r)[MeshPass::HullSums], pc, points);
    const auto hulls = ConvexHulls(r, chain, positions.Offset, first_point);
    const auto sums = ReadScratch<vec3>(chain, positions.Offset, points);
    std::vector<MeshSource> sources;
    for (uint32_t s = 0u; s < hulls.size(); ++s) {
        if (hulls[s].empty()) continue;
        MeshData data;
        std::vector<uint32_t> index(first_point[s + 1u] - first_point[s], InvalidOffset);
        std::vector<uint32_t> loop;
        for (const auto &face : hulls[s]) {
            loop.clear();
            for (const auto v : face) {
                if (index[v] == InvalidOffset) {
                    index[v] = uint32_t(data.Positions.size());
                    data.Positions.push_back(sums[first_point[s] + v]);
                }
                loop.push_back(index[v]);
            }
            if (hulls[s].size() == 1u && Dot(facing, facing) > 0.f) {
                vec3 normal{};
                for (size_t i = 1; i + 1 < loop.size(); ++i) normal += Cross(data.Positions[loop[i]] - data.Positions[loop[0]], data.Positions[loop[i + 1]] - data.Positions[loop[0]]);
                if (Dot(normal, facing) < 0.f) std::ranges::reverse(loop);
            }
            data.AddFace(loop);
        }
        sources.push_back({.Data = std::move(data)});
    }
    std::vector<uint32_t> records;
    for (const auto &created : CreateMeshes(r, sources)) records.push_back(created.StoreId);
    ShadeNewMeshes(r);
    return records;
}
std::optional<uint32_t> ConvexSum(state::Registry &r, const Mesh &a, const Mesh &b, vec3 facing) {
    const auto a_handles = HandleList(a.vertices()), b_handles = HandleList(b.vertices());
    const auto count = uint32_t(a_handles.size()), b_count = uint32_t(b_handles.size());
    const auto hulls = HullRecords(r, a, b, a_handles, b_handles, std::array{HullSumSet{.CountB = b_count}}, std::array{0u, count * b_count}, {}, facing);
    return hulls.empty() ? std::nullopt : std::optional{hulls.front()};
}
std::optional<uint32_t> Combine(state::Registry &r, const std::optional<RegionPlane> &plane, uint32_t first, uint32_t second, bool difference) {
    if (!plane) {
        const auto &meshes = r.Resources.get<const MeshStore>();
        const auto task = BooleanTask(Mesh{meshes, first}, Mesh{meshes, second}, difference ? BooleanOperation::Difference : BooleanOperation::Union, mat3{1.f}, vec3{});
        return task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
    }
    const auto loops = RegionLoops(r, *plane, std::array{RegionOperand{.Record = first}, RegionOperand{.Record = second}}, difference ? RegionFill::Difference : RegionFill::Union);
    if (!loops || loops->empty()) return {};
    return CreateShadedMesh(r, {.Data = RegionData(*plane, *loops)});
}
std::optional<uint32_t> FoldHulls(state::Registry &r, uint32_t first, std::span<const uint32_t> hulls, bool erosion, vec3 center, const std::optional<RegionPlane> &plane) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::optional<uint32_t> swept;
    for (const auto hull : hulls) {
        if (!swept) {
            swept = hull;
            continue;
        }
        const auto joined = Combine(r, plane, *swept, hull, false);
        meshes.Release(hull);
        if (joined) {
            meshes.Release(*swept);
            swept = joined;
        }
    }
    if (!swept) return {};
    const auto result = Combine(r, plane, first, *swept, erosion);
    meshes.Release(*swept);
    if (result && erosion) Translate(r, *result, center * -1.f);
    return result;
}
} // namespace

std::optional<uint32_t> MinkowskiRecord(state::Registry &r, const Mesh &first, const Mesh &second, MinkowskiOperation operation) {
    const bool solids = first.HasClosedSurface() && second.HasClosedSurface();
    if (!first.FaceCount() || !second.FaceCount()) return {};
    const auto plane = solids ? std::nullopt : FacePlane(r, first);
    const auto facing = plane ? plane->Normal : vec3{};
    const uint32_t k = plane ? 2u : 3u;
    const Mesh *a = &first, *b = &second;
    Geometry first_geometry{r, first}, second_geometry{r, second};
    const Geometry *a_geometry = &first_geometry, *b_geometry = &second_geometry;
    bool a_convex = plane ? first_geometry.ConvexRegion(*plane) : first_geometry.Convex(first);
    bool b_convex = plane ? second_geometry.ConvexRegion(*plane) : second_geometry.Convex(second);
    // A sum commutes, so the convex operand is the one swept along the other's faces.
    if (operation == MinkowskiOperation::Sum && a_convex && !b_convex) {
        std::swap(a, b);
        std::swap(a_geometry, b_geometry);
        std::swap(a_convex, b_convex);
    }
    if (operation == MinkowskiOperation::Sum && a_convex && b_convex) return ConvexSum(r, *a, *b, facing);
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto a_simplices = plane ? a_geometry->EdgeHandles() : TriangleVertices(meshes, pipelines, a->GetStoreId());
    std::vector<HullSumSet> sets;
    std::vector<uint32_t> first_point{0u};
    std::vector<uint32_t> b_handles;
    if (b_convex) {
        b_handles = HandleList(b->vertices());
        const auto b_count = uint32_t(b_handles.size());
        for (uint32_t sa = 0u; sa < a_simplices.size(); sa += k) {
            sets.push_back({.First = first_point.back(), .FirstA = sa, .CountB = b_count});
            first_point.push_back(first_point.back() + k * b_count);
        }
    } else {
        b_handles = plane ? b_geometry->EdgeHandles() : TriangleVertices(meshes, pipelines, b->GetStoreId());
        for (uint32_t sa = 0u; sa < a_simplices.size(); sa += k) {
            const auto na = plane ? vec3{} : a_geometry->TriangleNormal(std::span{a_simplices}.subspan(sa, 3u));
            for (uint32_t sb = 0u; sb < b_handles.size(); sb += k) {
                // Parallel faces sweep a flat hull, which adds nothing.
                if (!plane && std::abs(std::abs(Dot(na, b_geometry->TriangleNormal(std::span{b_handles}.subspan(sb, 3u)))) - 1.f) < 1e-6f) continue;
                sets.push_back({.First = first_point.back(), .FirstA = sa, .FirstB = sb, .CountB = k});
                first_point.push_back(first_point.back() + k * k);
            }
        }
    }
    const auto erosion = operation == MinkowskiOperation::Difference;
    const auto center = b_geometry->Centroid(*b);
    const auto hulls = HullRecords(r, *a, *b, a_simplices, b_handles, sets, first_point, erosion ? std::optional{center} : std::nullopt, facing);
    return FoldHulls(r, a->GetStoreId(), hulls, erosion, center, plane);
}
