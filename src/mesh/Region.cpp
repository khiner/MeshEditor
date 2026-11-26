#include "mesh/Region.h"

#include "Profile.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshCreate.h"
#include "mesh/Neighborhood.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

#include <cstring>
#include <numeric>
#include <ranges>

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
    return PlaneAt(VertexPositions(meshes, mesh.GetStoreId(), vertex).front(), FaceNormals(meshes, mesh.GetStoreId(), face).front());
}

namespace {
// Float bits whose unsigned order is the floats' order, matching the kernels.
float OrderedFloat(uint32_t bits) { return std::bit_cast<float>((bits & 0x80000000u) ? bits & 0x7fffffffu : ~bits); }
// A task's fill between its submits: its job record in the chain's scratch and what its passes read back.
struct RegionWork {
    RegionJob Job{};
    uint32_t Record{}, Deviation{}, Aliases{}, Operands{};
    std::array<RegionSegmentsJob, 2> Segments{};
    std::array<ClosureSeed, 2> Faces{};
    std::array<uint32_t, 2> Counts{};
    std::optional<std::vector<std::vector<vec2>>> Loops{std::vector<std::vector<vec2>>{}};
    bool Done{};
};
// A Project node's projection of its record onto its plane.
struct ProjectionRequest {
    uint32_t Record;
    RegionPlane Plane;
    std::function<void(std::optional<uint32_t>)> Then;
};
} // namespace

std::vector<std::optional<std::vector<std::vector<vec2>>>> RegionLoops(state::Registry &r, std::span<const RegionTask> tasks) {
    const profile::CpuScope scope{"RegionLoops"};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto &scratch = chain.Scratch;
    std::vector<RegionWork> works(tasks.size());
    const auto upload = [&](const RegionWork &work) { scratch.Buffer.Update(std::as_bytes(std::span{&work.Job, 1u}), uint64_t(work.Record) * sizeof(uint32_t)); };
    // Each task's passes record in a lane of its own, so a pass of every task records once.
    const auto record = [&](const RegionWork &work, MeshPass pass, uint32_t threads) { mtl::GatherPass(chain, pipelines[pass], SlotOffset{scratch.Buffer.Slot, work.Record}, threads); };
    const auto active = [&] { return std::views::iota(0u, uint32_t(works.size())) | std::views::filter([&](uint32_t t) { return !works[t].Done; }); };
    for (uint32_t t = 0u; t < tasks.size(); ++t) {
        const auto &[plane, operands, fill] = tasks[t];
        auto &work = works[t];
        auto &job = work.Job;
        work.Operands = uint32_t(operands.size());
        work.Record = TakeScratch(chain, sizeof(RegionJob) / sizeof(uint32_t)).Offset;
        job = {.Fill = fill, .Bounds = TakeScratch(chain, 8u).Offset, .Parity = TakeScratch(chain, 1u).Offset, .Fault = TakeScratch(chain, 1u).Offset};
        work.Deviation = TakeScratch(chain, 1u).Offset;
        const auto bounds = scratch.GetMutable({job.Bounds, 8u});
        for (uint32_t side = 0u; side < 2u; ++side) {
            std::ranges::fill(bounds.subspan(4u * side, 2u), ~0u);
            std::ranges::fill(bounds.subspan(4u * side + 2u, 2u), 0u);
        }
        scratch.GetMutable({work.Deviation, 1u})[0] = 0u;
        scratch.GetMutable({job.Fault, 1u})[0] = 0u;
        chain.Lane(t, [&] {
            for (uint32_t side = 0u; side < work.Operands; ++side) {
                job.Vertices[side] = a.Vertices.HandleBound(meshes.Get(operands[side].Record).Vertices);
                work.Faces[side] = EncodeAllSeed(r, chain, operands[side].Record, Element::Face);
            }
        });
        const uint32_t vertices = job.Vertices[0] + job.Vertices[1];
        job.Positions = TakeScratch(chain, 2ull * vertices).Offset;
        job.Moves = TakeScratch(chain, 2ull * vertices).Offset;
        work.Aliases = TakeScratch(chain, vertices).Offset;
        std::ranges::iota(scratch.GetMutable({work.Aliases, vertices}), 0u);
        for (uint32_t side = 0u; side < work.Operands; ++side) {
            const auto &faces = work.Faces[side];
            work.Counts[side] = TakeScratch(chain, uint64_t(faces.Count) + 1u).Offset;
            work.Segments[side] = {
                .Connectivity = meshes.GetConnectivityRef(operands[side].Record),
                .VertexSlot = a.Vertices.Buffer.Slot,
                .FaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
                .Faces = faces.Work,
                .FaceCount = faces.Count,
                .Operand = side,
                .VertexBase = side ? job.Vertices[0] : 0u,
                .Cusps = operands[side].Silhouette ? 1u : 0u,
                .Rotation = operands[side].Rotation,
                .Translation = operands[side].Translation,
                .Point = plane.Point,
                .Normal = plane.Normal,
                .U = plane.U,
                .V = plane.V,
                .Aliases = {scratch.Buffer.Slot, work.Aliases},
                .Counts = {scratch.Buffer.Slot, work.Counts[side]},
                .Positions = {scratch.Buffer.Slot, job.Positions},
                .Bounds = {scratch.Buffer.Slot, job.Bounds},
                .Deviation = {scratch.Buffer.Slot, work.Deviation},
            };
            chain.Lane(t, [&] {
                mtl::GatherPass(chain, pipelines[MeshPass::RegionSegmentAlias], work.Segments[side], faces.Count);
                mtl::GatherPass(chain, pipelines[MeshPass::RegionSegmentCount], work.Segments[side], faces.Count + 1u);
                ScanScratch(chain, pipelines, std::array{Range{work.Counts[side], faces.Count + 1u}});
            });
        }
    }
    chain.Submit();
    for (const auto t : active()) {
        auto &work = works[t];
        auto &job = work.Job;
        for (uint32_t side = 0u; side < work.Operands; ++side) job.Segments[side] = scratch.Get({work.Counts[side] + work.Faces[side].Count, 1u})[0];
        const uint32_t total = job.Segments[0] + job.Segments[1], vertices = job.Vertices[0] + job.Vertices[1];
        if (!total) {
            work.Done = true;
            continue;
        }
        const auto bounds = scratch.Get({job.Bounds, 8u});
        float largest = 0.f;
        for (uint32_t side = 0u; side < work.Operands; ++side) {
            if (!job.Segments[side]) continue;
            for (uint32_t k = 0u; k < 4u; ++k) largest = std::max(largest, std::abs(OrderedFloat(bounds[4u * side + k])));
        }
        if (OrderedFloat(scratch.Get({work.Deviation, 1u})[0]) > RelativePrecision * largest) {
            work.Loops.reset();
            work.Done = true;
            continue;
        }
        job.SegmentRecords = TakeScratch(chain, 3ull * total).Offset;
        job.IncidentFirst = TakeScratch(chain, uint64_t(vertices) + 1u).Offset;
        job.IncidentCursor = TakeScratch(chain, vertices).Offset;
        job.Incident = TakeScratch(chain, 2ull * total).Offset;
        job.Collider = AllocateCollider([&](uint64_t words) { return TakeScratch(chain, words).Offset; }, total);
        job.CrossingFirst = TakeScratch(chain, uint64_t(total) + 1u).Offset;
        std::ranges::fill(scratch.GetMutable({job.IncidentFirst, vertices + 1u}), 0u);
        upload(work);
        chain.Lane(t, [&] {
            for (uint32_t side = 0u; side < work.Operands; ++side) {
                auto &emit = work.Segments[side];
                emit.Segments = {scratch.Buffer.Slot, job.SegmentRecords};
                emit.SegmentBase = side ? job.Segments[0] : 0u;
                if (job.Segments[side]) mtl::GatherPass(chain, pipelines[MeshPass::RegionSegmentEmit], emit, work.Faces[side].Count);
            }
            record(work, MeshPass::RegionIncidentCount, total);
            ScanScratch(chain, pipelines, std::array{Range{job.IncidentFirst, vertices + 1u}});
            record(work, MeshPass::RegionIncidentCursors, vertices);
            record(work, MeshPass::RegionIncidentFill, total);
            record(work, MeshPass::RegionMoves, vertices);
            record(work, MeshPass::RegionMorton, total);
            RecordColliderBuild(chain, pipelines, std::array{ColliderBuild{work.Record + uint32_t(offsetof(RegionJob, Collider) / sizeof(uint32_t)), total}});
            record(work, MeshPass::RegionRefit, total);
            record(work, MeshPass::RegionCrossingCount, total + 1u);
            ScanScratch(chain, pipelines, std::array{Range{job.CrossingFirst, total + 1u}});
        });
    }
    chain.Submit();
    for (const auto t : active()) {
        auto &work = works[t];
        auto &job = work.Job;
        const uint32_t total = job.Segments[0] + job.Segments[1], vertices = job.Vertices[0] + job.Vertices[1];
        const uint32_t crossings = scratch.Get({job.CrossingFirst + total, 1u})[0];
        profile::RecordCounter("RegionCrossings", crossings);
        job.Crossings = TakeScratch(chain, uint64_t(crossings) * (sizeof(RegionCrossing) / sizeof(uint32_t))).Offset;
        job.Windings = TakeScratch(chain, 2ull * vertices).Offset;
        job.Pieces = crossings + total;
        job.PieceKept = TakeScratch(chain, uint64_t(job.Pieces) + 1u).Offset;
        job.PieceReversed = TakeScratch(chain, job.Pieces).Offset;
        upload(work);
        chain.Lane(t, [&] {
            record(work, MeshPass::RegionCrossingEmit, total);
            record(work, MeshPass::RegionWindings, vertices);
            record(work, MeshPass::RegionPieces, total + 1u);
            ScanScratch(chain, pipelines, std::array{Range{job.PieceKept, job.Pieces + 1u}});
        });
    }
    chain.Submit();
    for (const auto t : active()) {
        auto &work = works[t];
        auto &job = work.Job;
        job.KeptPieces = scratch.Get({job.PieceKept + job.Pieces, 1u})[0];
        if (!job.KeptPieces) {
            work.Done = true;
            continue;
        }
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
        upload(work);
        const SlotOffset ranked{scratch.Buffer.Slot, work.Record};
        chain.Lane(t, [&] {
            record(work, MeshPass::RegionKeptFill, job.Pieces);
            record(work, MeshPass::RegionNext, kept);
            GatherLoopRanks(chain, pipelines, {MeshPass::RegionLabel, MeshPass::RegionRankInit, MeshPass::RegionRank, MeshPass::RegionLoopCount}, ranked, TileCount(kept, JobTileThreads), [&chain](const SlotOffset &gathered) {
                RegionJob held;
                std::memcpy(&held, chain.Scratch.Get({gathered.Offset, uint32_t(sizeof(RegionJob) / sizeof(uint32_t))}).data(), sizeof(RegionJob));
                return LoopList{held.KeptPieces, held.LoopLengths, held.LoopFlags};
            });
        });
    }
    chain.Submit();
    for (const auto t : active()) {
        auto &work = works[t];
        auto &job = work.Job;
        if (scratch.Get({job.Fault, 1u})[0]) {
            work.Loops.reset();
            work.Done = true;
            continue;
        }
        const uint32_t loop_count = scratch.Get({job.LoopFlags + job.KeptPieces, 1u})[0];
        job.LoopRecords = TakeScratch(chain, 2ull * loop_count).Offset;
        upload(work);
        chain.Lane(t, [&] { record(work, MeshPass::RegionOutput, job.KeptPieces); });
    }
    chain.Submit();
    for (const auto t : active()) {
        auto &work = works[t];
        const auto &job = work.Job;
        const uint32_t loop_count = scratch.Get({job.LoopFlags + job.KeptPieces, 1u})[0];
        const auto records = scratch.Get({job.LoopRecords, 2u * loop_count});
        auto &loops = *work.Loops;
        loops.resize(loop_count);
        for (uint32_t l = 0u; l < loop_count; ++l) loops[l] = ReadScratch<vec2>(chain, job.Output + 2u * records[2u * l], records[2u * l + 1u]);
    }
    return works | std::views::transform(&RegionWork::Loops) | std::ranges::to<std::vector>();
}

void RecordProjection(state::Registry &r, mtl::ComputeChain &chain, uint32_t record, const RegionPlane &plane, std::function<void(std::optional<uint32_t>)> then) {
    static constexpr char Projections{};
    chain.Lane(record, [&] {
        chain.Gather(&Projections, std::vector{ProjectionRequest{record, plane, std::move(then)}}, 0u, [&r, &chain](std::vector<ProjectionRequest> requests) {
            chain.AfterSubmit([&r, requests = std::move(requests)] {
                const auto tasks = requests | std::views::transform([](const auto &request) { return RegionTask{request.Plane, {RegionOperand{.Record = request.Record, .Silhouette = true}}, RegionFill::Positive}; }) | std::ranges::to<std::vector>();
                const auto loops = RegionLoops(r, tasks);
                std::vector<MeshSource> sources;
                std::vector<uint32_t> projected;
                for (uint32_t i = 0u; i < loops.size(); ++i) {
                    if (!loops[i] || loops[i]->empty()) continue;
                    sources.push_back({.Data = RegionData(tasks[i].Plane, *loops[i])});
                    projected.push_back(i);
                }
                std::vector<std::optional<uint32_t>> records(requests.size());
                if (!sources.empty()) {
                    for (uint32_t k = 0u; const auto &created : CreateMeshes(r, sources)) records[projected[k++]] = created.StoreId;
                    ShadeNewMeshes(r);
                }
                for (uint32_t i = 0u; i < requests.size(); ++i) requests[i].Then(records[i]);
            });
        });
    });
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
