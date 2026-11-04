#include "mesh/Region.h"

#include "Profile.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

RegionPlane PlaneAt(vec3 point, vec3 normal) {
    const auto n = Normalize(normal);
    const vec3 axis = std::abs(n.x) < std::abs(n.y) ? (std::abs(n.x) < std::abs(n.z) ? vec3{1.f, 0.f, 0.f} : vec3{0.f, 0.f, 1.f}) : (std::abs(n.y) < std::abs(n.z) ? vec3{0.f, 1.f, 0.f} : vec3{0.f, 0.f, 1.f});
    const auto u = Normalize(Cross(axis, n));
    return {point, n, u, Cross(n, u)};
}

std::optional<RegionPlane> FacePlane(state::Registry &r, const Mesh &mesh) {
    if (!mesh.FaceCount()) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const std::array face{**mesh.faces().begin()};
    const std::array vertex{FaceCorners(meshes, pipelines, mesh.GetStoreId(), face).Items.front().Vertex};
    return PlaneAt(VertexPositions(meshes, pipelines, mesh.GetStoreId(), vertex).front(), FaceNormals(meshes, pipelines, mesh.GetStoreId(), face).front());
}

namespace {
// Float bits whose unsigned order is the floats' order, matching the kernels.
float OrderedFloat(uint32_t bits) { return std::bit_cast<float>((bits & 0x80000000u) ? bits & 0x7fffffffu : ~bits); }
} // namespace

std::optional<std::vector<std::vector<vec2>>> RegionLoops(state::Registry &r, const RegionPlane &plane, std::span<const RegionOperand> operands, RegionFill fill) {
    const profile::CpuScope scope{"RegionLoops"};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto &scratch = chain.Scratch;
    const auto record = TakeScratch(chain, sizeof(RegionJob) / sizeof(uint32_t)).Offset;
    RegionJob job{.Fill = fill, .Bounds = TakeScratch(chain, 8u).Offset, .Fault = TakeScratch(chain, 1u).Offset};
    const auto deviation = TakeScratch(chain, 1u).Offset;
    {
        const auto bounds = scratch.GetMutable({job.Bounds, 8u});
        for (uint32_t side = 0u; side < 2u; ++side) {
            std::ranges::fill(bounds.subspan(4u * side, 2u), ~0u);
            std::ranges::fill(bounds.subspan(4u * side + 2u, 2u), 0u);
        }
        scratch.GetMutable({deviation, 1u})[0] = 0u;
        scratch.GetMutable({job.Fault, 1u})[0] = 0u;
    }
    const auto count = uint32_t(operands.size());
    std::array<RegionSegmentsPushConstants, 2> segments{};
    std::array<ClosureSeed, 2> faces{};
    std::array<uint32_t, 2> counts{};
    for (uint32_t side = 0u; side < count; ++side) {
        job.Vertices[side] = a.Vertices.HandleBound(meshes.Get(operands[side].Record).Vertices);
        faces[side] = EncodeAllSeed(r, chain, operands[side].Record, Element::Face);
    }
    const uint32_t vertices = job.Vertices[0] + job.Vertices[1];
    job.Positions = TakeScratch(chain, 2ull * vertices).Offset;
    job.Moves = TakeScratch(chain, 2ull * vertices).Offset;
    const auto aliases = TakeScratch(chain, vertices).Offset;
    std::ranges::iota(scratch.GetMutable({aliases, vertices}), 0u);
    const JobPushConstants pc{.Job = {scratch.Buffer.Slot, record}};
    const auto upload = [&] { scratch.Buffer.Update(std::as_bytes(std::span{&job, 1u}), uint64_t(record) * sizeof(uint32_t)); };
    for (uint32_t side = 0u; side < count; ++side) {
        counts[side] = TakeScratch(chain, uint64_t(faces[side].Count) + 1u).Offset;
        segments[side] = {
            .Connectivity = meshes.GetConnectivityRef(operands[side].Record),
            .VertexSlot = a.Vertices.Buffer.Slot,
            .FaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
            .Faces = faces[side].Work,
            .FaceCount = faces[side].Count,
            .Operand = side,
            .VertexBase = side ? job.Vertices[0] : 0u,
            .Cusps = operands[side].Silhouette ? 1u : 0u,
            .Rotation = operands[side].Rotation,
            .Translation = operands[side].Translation,
            .Point = plane.Point,
            .Normal = plane.Normal,
            .U = plane.U,
            .V = plane.V,
            .Aliases = {scratch.Buffer.Slot, aliases},
            .Counts = {scratch.Buffer.Slot, counts[side]},
            .Positions = {scratch.Buffer.Slot, job.Positions},
            .Bounds = {scratch.Buffer.Slot, job.Bounds},
            .Deviation = {scratch.Buffer.Slot, deviation},
        };
        chain.Threads(pipelines[MeshPass::RegionSegmentAlias], segments[side], faces[side].Count);
        chain.Threads(pipelines[MeshPass::RegionSegmentCount], segments[side], faces[side].Count + 1u);
        ScanScratch(chain, pipelines, counts[side], faces[side].Count + 1u);
    }
    chain.Submit();
    for (uint32_t side = 0u; side < count; ++side) job.Segments[side] = scratch.Get({counts[side] + faces[side].Count, 1u})[0];
    const uint32_t total = job.Segments[0] + job.Segments[1];
    std::vector<std::vector<vec2>> loops;
    if (!total) return loops;
    {
        const auto bounds = scratch.Get({job.Bounds, 8u});
        float largest = 0.f;
        for (uint32_t side = 0u; side < count; ++side) {
            if (!job.Segments[side]) continue;
            for (uint32_t k = 0u; k < 4u; ++k) largest = std::max(largest, std::abs(OrderedFloat(bounds[4u * side + k])));
        }
        if (OrderedFloat(scratch.Get({deviation, 1u})[0]) > RelativePrecision * largest) return std::nullopt;
    }
    job.SegmentRecords = TakeScratch(chain, 3ull * total).Offset;
    job.IncidentFirst = TakeScratch(chain, uint64_t(vertices) + 1u).Offset;
    job.IncidentCursor = TakeScratch(chain, vertices).Offset;
    job.Incident = TakeScratch(chain, 2ull * total).Offset;
    job.Collider = AllocateCollider([&](uint64_t words) { return TakeScratch(chain, words).Offset; }, total);
    job.CrossingFirst = TakeScratch(chain, uint64_t(total) + 1u).Offset;
    std::ranges::fill(scratch.GetMutable({job.IncidentFirst, vertices + 1u}), 0u);
    upload();
    for (uint32_t side = 0u; side < count; ++side) {
        segments[side].Segments = {scratch.Buffer.Slot, job.SegmentRecords};
        segments[side].SegmentBase = side ? job.Segments[0] : 0u;
        if (job.Segments[side]) chain.Threads(pipelines[MeshPass::RegionSegmentEmit], segments[side], faces[side].Count);
    }
    chain.Threads(pipelines[MeshPass::RegionIncidentCount], pc, total);
    ScanScratch(chain, pipelines, job.IncidentFirst, vertices + 1u);
    chain.Threads(pipelines[MeshPass::RegionIncidentCursors], pc, vertices);
    chain.Threads(pipelines[MeshPass::RegionIncidentFill], pc, total);
    chain.Threads(pipelines[MeshPass::RegionMoves], pc, vertices);
    chain.Threads(pipelines[MeshPass::RegionMorton], pc, total);
    RecordColliderBuild(chain, pipelines, {scratch.Buffer.Slot, record + uint32_t(offsetof(RegionJob, Collider) / sizeof(uint32_t))}, TileCount(total, TileElements), 1u);
    chain.Threads(pipelines[MeshPass::RegionRefit], pc, total);
    chain.Threads(pipelines[MeshPass::RegionCrossingCount], pc, total + 1u);
    ScanScratch(chain, pipelines, job.CrossingFirst, total + 1u);
    chain.Submit();
    const uint32_t crossings = scratch.Get({job.CrossingFirst + total, 1u})[0];
    profile::RecordCounter("RegionCrossings", crossings);
    job.Crossings = TakeScratch(chain, uint64_t(crossings) * (sizeof(RegionCrossing) / sizeof(uint32_t))).Offset;
    job.Windings = TakeScratch(chain, 2ull * vertices).Offset;
    job.Pieces = crossings + total;
    job.PieceKept = TakeScratch(chain, uint64_t(job.Pieces) + 1u).Offset;
    job.PieceReversed = TakeScratch(chain, job.Pieces).Offset;
    upload();
    chain.Threads(pipelines[MeshPass::RegionCrossingEmit], pc, total);
    chain.Threads(pipelines[MeshPass::RegionWindings], pc, vertices);
    chain.Threads(pipelines[MeshPass::RegionPieces], pc, total + 1u);
    ScanScratch(chain, pipelines, job.PieceKept, job.Pieces + 1u);
    chain.Submit();
    job.KeptPieces = scratch.Get({job.PieceKept + job.Pieces, 1u})[0];
    if (!job.KeptPieces) return loops;
    const uint32_t kept = job.KeptPieces;
    job.Kept = TakeScratch(chain, kept).Offset;
    job.Next = TakeScratch(chain, kept).Offset;
    job.Labels = TakeScratch(chain, 2ull * kept).Offset;
    job.Jumps = TakeScratch(chain, 2ull * kept).Offset;
    job.Distances = TakeScratch(chain, 2ull * kept).Offset;
    job.Successors = TakeScratch(chain, 2ull * kept).Offset;
    job.LoopLengths = TakeScratch(chain, uint64_t(kept) + 1u).Offset;
    job.LoopFlags = TakeScratch(chain, uint64_t(kept) + 1u).Offset;
    job.Output = TakeScratch(chain, 2ull * kept).Offset;
    upload();
    chain.Threads(pipelines[MeshPass::RegionKeptFill], pc, job.Pieces);
    chain.Threads(pipelines[MeshPass::RegionNext], pc, kept);
    const auto round_pc = RecordLoopRanks(chain, pipelines, pc, {MeshPass::RegionLabel, MeshPass::RegionRankInit, MeshPass::RegionRank, MeshPass::RegionLoopCount}, kept, kept, job.LoopLengths, job.LoopFlags);
    chain.Submit();
    if (scratch.Get({job.Fault, 1u})[0]) return std::nullopt;
    const uint32_t loop_count = scratch.Get({job.LoopFlags + kept, 1u})[0];
    job.LoopRecords = TakeScratch(chain, 2ull * loop_count).Offset;
    upload();
    chain.Threads(pipelines[MeshPass::RegionOutput], round_pc, kept);
    chain.Submit();
    const auto records = scratch.Get({job.LoopRecords, 2u * loop_count});
    loops.resize(loop_count);
    for (uint32_t l = 0u; l < loop_count; ++l) loops[l] = ReadScratch<vec2>(chain, job.Output + 2u * records[2u * l], records[2u * l + 1u]);
    return loops;
}

MeshData RegionData(const RegionPlane &plane, std::span<const std::vector<vec2>> loops) {
    float largest = 0.f;
    for (const auto &loop : loops) {
        for (const auto p : loop) largest = std::max({largest, std::abs(p.x), std::abs(p.y)});
    }
    const float tolerance = RelativePrecision * largest;
    MeshData data;
    std::vector<vec2> points;
    std::vector<uint32_t> face;
    for (const auto &loop : loops) {
        points.clear();
        for (const auto p : loop) {
            if (points.empty() || Length(p - points.back()) > tolerance) points.push_back(p);
        }
        while (points.size() > 1u && Length(points.front() - points.back()) <= tolerance) points.pop_back();
        if (points.size() < 3u) continue;
        face.clear();
        for (const auto p : points) {
            face.push_back(uint32_t(data.Positions.size()));
            data.Positions.push_back(plane.Lift(p));
        }
        data.AddFace(face);
    }
    return data;
}
