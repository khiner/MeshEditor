#include "mesh/PositionOperations.h"
#include "Profile.h"
#include "SortUnique.h"
#include "gpu/FaceAttributeEditPushConstants.h"
#include "gpu/Vertex.h"
#include "mesh/EdgeChains.h"
#include "mesh/EdgeSlide.h"
#include "mesh/Flatten.h"
#include "mesh/Mesh.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/SnapSymmetry.h"
#include "metal/Dispatch.h"
#include "numeric/QuaternionMath.h"
#include "render/ElementWorkOps.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <unordered_set>

namespace {
bool Finite(vec3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
bool ValidPlane(quat orientation, vec3 center, float roll) {
    const float norm = orientation.x * orientation.x + orientation.y * orientation.y + orientation.z * orientation.z + orientation.w * orientation.w;
    return norm > 0.f && std::isfinite(norm) && Finite(center) && std::isfinite(roll);
}
void ValidateTargets(const MeshStore &meshes, std::span<const PositionOperationTarget> targets) {
    std::unordered_set<uint32_t> ids;
    for (const auto &target : targets) {
        if (!ids.insert(target.StoreId).second) throw std::invalid_argument("Position operations require distinct meshes.");
        ValidateGeometrySelection(meshes, target.StoreId, target.Selection);
        ValidateGeometrySelection(meshes, target.StoreId, target.Excluded);
    }
}
} // namespace

// Only selected face ranges and attribute pages participate; vertex fans are irrelevant.
std::vector<FaceAttributeOperationChange> EncodeFaceAttributeOperation(MeshStore &meshes, MeshPipelines &pipelines, mtl::ComputeChain &chain, std::span<const PositionOperationTarget> targets, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets || operation > 2u) return {};
    ValidateTargets(meshes, targets);
    auto &a = meshes.Arenas();
    std::vector<FaceAttributeEditPushConstants> jobs;
    std::vector<FaceAttributeOperationChange> changes;
    uint64_t corners = 0u, face_count = 0u;
    for (uint32_t target_index = 0u; target_index < targets.size(); ++target_index) {
        const auto &target = targets[target_index];
        const auto id = target.StoreId;
        const Mesh mesh{meshes, id};
        const auto &record = meshes.Get(id);
        if (!(record.CornerAttributes & (colors ? MeshAttributeBit_Color0 : MeshAttributeBit_TexCoord0 << uv_set))) continue;
        const bool clear_tangents = !colors && (record.CornerAttributes & MeshAttributeBit_Tangent);
        const auto &faces = target.Selection.Faces;
        uint32_t incidence = 0u;
        for (const auto f : faces) {
            const auto range = a.FaceRanges.Get({f, 1u})[0];
            const Range handles{range.x, range.y - range.x};
            if (colors) a.CornerColors.CaptureHandles(handles);
            else a.CornerUvs[uv_set].CaptureHandles(handles);
            if (clear_tangents) a.CornerTangents.CaptureHandles(handles);
            incidence += handles.Count;
        }
        if (faces.empty()) continue;
        const auto &seed = changes.emplace_back(FaceAttributeOperationChange{target_index, {.Work = SeedElementWorkHandles(chain.Scratch, a.FaceTriangles.Capacity(), faces), .Count = uint32_t(faces.size()), .Incidence = incidence}}).Faces;
        jobs.push_back({
            .Connectivity = meshes.GetConnectivityRef(id),
            .Faces = seed.Work,
            .Attribute = colors ? a.CornerColors.Ref() : a.CornerUvs[uv_set].Ref(),
            .Tangents = a.CornerTangents.Ref(clear_tangents),
            .FaceCount = mesh.FaceCount(),
            .Count = seed.Count,
            .Operation = operation,
        });
        corners += incidence;
        face_count += seed.Count;
    }
    if (jobs.empty()) return {};
    const auto &pipeline = pipelines[colors ? MeshPass::EditFaceColors : MeshPass::EditFaceUvs];
    chain.Concurrent([&] { for (const auto &pc : jobs) chain.Threads(pipeline, pc, pc.Count); });
    profile::RecordCounter("AttributeEditFaces", face_count);
    profile::RecordCounter("AttributeEditCorners", corners);
    return changes;
}

// Position operators share sparse capture and iterative canonical GPU writes.
std::vector<PositionOperationChange> EncodePositionOperations(MeshStore &meshes, MeshPipelines &pipelines, mtl::ComputeChain &chain, std::span<const PositionOperationTarget> targets, PositionEditOp operation, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    const auto &[axes, requested_flags, direction, gradient, center, warp, bend, randomize, slide, edge_slide, symmetry, curve, circle, transform] = options;
    uint32_t flags = requested_flags;
    if (!repeat || axes > 7u || uint32_t(operation) > uint32_t(PositionEditOp::Transform)) return {};
    if (bool(warp) != (operation == PositionEditOp::Warp) || bool(bend) != (operation == PositionEditOp::Bend) ||
        bool(randomize) != (operation == PositionEditOp::Randomize) || bool(slide) != (operation == PositionEditOp::VertexSlide) ||
        bool(edge_slide) != (operation == PositionEditOp::EdgeSlide) || bool(symmetry) != (operation == PositionEditOp::SnapSymmetry) ||
        bool(transform) != (operation == PositionEditOp::Transform) || ((curve || circle) && operation != PositionEditOp::Copy) || (curve && circle)) return {};
    if (!Finite(center) || !Finite(direction) || !Finite(gradient)) return {};
    if (warp && (!ValidPlane(warp->Orientation, warp->Center, warp->OffsetAngle) || (!warp->AutoRange && (!std::isfinite(warp->Min) || !std::isfinite(warp->Max) || warp->Min == warp->Max)))) return {};
    if (bend && (!ValidPlane(bend->Orientation, bend->Center, bend->OffsetAngle) || !std::isfinite(bend->Radius) || bend->Radius == 0.f)) return {};
    if (randomize && (!std::isfinite(randomize->Uniform) || !std::isfinite(randomize->Normal))) return {};
    for (const auto &parameters : {slide, edge_slide})
        if (parameters && (!(Length(parameters->Direction) > 0.f) || !std::isfinite(Length(parameters->Direction)))) return {};
    if (circle && (!std::isfinite(circle->Radius) || !std::isfinite(circle->Angle))) return {};
    if (symmetry && (!std::has_single_bit(axes) || !std::isfinite(symmetry->Threshold) || symmetry->Threshold <= 0.f)) return {};
    if (transform && (!Finite(transform->Delta.P) || !Finite(transform->Delta.S) || !Finite(transform->Pivot) || !ValidPlane(transform->Delta.R, {}, 0.f))) return {};
    ValidateTargets(meshes, targets);
    if (warp) flags = (flags & ~PositionEditWarpAutoRange) | (warp->AutoRange ? PositionEditWarpAutoRange : 0u);
    for (const auto &parameters : {slide, edge_slide})
        if (parameters) {
            flags = (flags & ~(PositionEditSlideEven | PositionEditSlideFlipped)) |
                (parameters->Even ? PositionEditSlideEven : 0u) | (parameters->Flipped ? PositionEditSlideFlipped : 0u);
        }
    const bool zero_moves = symmetry || warp || (slide && slide->Even && slide->Flipped) || (edge_slide && edge_slide->Even);
    if (targets.empty() || !axes || (!zero_moves && factor == 0.f) || !std::isfinite(factor)) return {};
    const bool planar = operation == PositionEditOp::Planar;
    const bool ordered_chains = curve || circle;
    const bool flatten = operation == PositionEditOp::Copy && !ordered_chains;
    const bool flatten_view = flatten && (flags & PositionEditFlattenView);
    const bool relax = operation == PositionEditOp::RelaxEdgeLoops;
    const bool edge_chains = relax || operation == PositionEditOp::SpaceEvenly;
    const bool sphere = operation == PositionEditOp::ToSphere;
    const auto gather = edge_chains ? (relax ? MeshPass::RelaxEdgeLoopsGather : MeshPass::SpaceEvenlyGather) : MeshPass::PositionVerticesGather;
    const bool centered = sphere || operation == PositionEditOp::PushPull || operation == PositionEditOp::Shear;
    const bool transformed = centered || warp || bend || slide || edge_slide || flatten_view || transform;
    if (slide && !(flags & PositionEditSlideUnclamped)) factor = std::clamp(factor, 0.f, 1.f);
    else if (edge_slide && !(flags & PositionEditSlideUnclamped)) factor = std::clamp(factor, -1.f, 1.f);
    else if (transform || sphere || symmetry || flatten || ordered_chains || operation == PositionEditOp::SpaceEvenly) factor = std::clamp(factor, 0.f, 1.f);
    else if (planar || operation == PositionEditOp::Smooth) factor = std::clamp(factor, -10.f, 10.f);
    if (!zero_moves && factor == 0.f) return {};
    vec3 plane_x{}, plane_y{}, plane_center{};
    float bend_pivot = 0.f;
    if (transformed)
        for (const auto &target : targets)
            if (!Finite(target.World.P) || !Finite(target.World.S) || !ValidPlane(target.World.R, {}, 0.f)) return {};
    if (transform)
        for (const auto &target : targets)
            if (target.World.S.x == 0.f || target.World.S.y == 0.f || target.World.S.z == 0.f) return {};
    if (warp || bend) {
        const auto rotation = Normalize(warp ? warp->Orientation : bend->Orientation);
        const auto right = rotation * vec3{1, 0, 0}, up = rotation * vec3{0, 1, 0};
        const float roll = warp ? warp->OffsetAngle : bend->OffsetAngle;
        const float c = std::cos(roll), s = std::sin(roll);
        plane_x = c * right - s * up;
        plane_y = s * right + c * up;
        plane_center = warp ? warp->Center : bend->Center;
        if (bend) {
            // Equivalent to Blender's shell_angle_to_dist, with a stable
            // small-angle denominator and saturation past a quarter turn.
            const float angle = std::abs(factor);
            const float shell = angle >= std::numbers::pi_v<float> * .5f ? 1.f : 1.f / std::sin(angle);
            bend_pivot = -std::copysign(1.f, factor) * bend->Radius * shell;
            if (!std::isfinite(bend_pivot)) return {};
        }
    }
    uint64_t selected_count = 0u;
    uint32_t reduction_blocks = 0u;
    if (!planar) {
        for (const auto &target : targets) {
            const auto count = target.Selection.Vertices.size();
            selected_count += count;
            if (sphere || (warp && warp->AutoRange)) reduction_blocks += (count + 255u) / 256u;
        }
        // Symmetry can also move an unselected partner for each selected vertex.
        const uint32_t vertex_words = edge_slide || symmetry ? 10u : 4u;
        const uint32_t parameter_words = transform ? sizeof(TransformPositionParameters) / sizeof(uint32_t) : warp || bend ? 18u :
            randomize                                                                                                      ? 3u :
            slide                                                                                                          ? 8u :
            edge_slide                                                                                                     ? 1u :
                                                                                                                             0u;
        chain.Scratch.ReserveAdditional(vertex_words * selected_count + 4ull * reduction_blocks + 1u + uint64_t(parameter_words) * targets.size());
    }
    const bool statistics = sphere || (warp && warp->AutoRange);
    const auto partials = chain.Scratch.Allocate(2u * reduction_blocks);
    uint32_t partial_offset = partials.Offset;
    std::vector<VertexPositionEditPushConstants> jobs;
    struct PositionBatch {
        uint32_t Index;
        std::vector<Range> Batches;
    };
    std::vector<PositionBatch> batches;
    std::vector<PositionOperationChange> changes;
    for (uint32_t target_index = 0u; target_index < targets.size(); ++target_index) {
        const auto &target = targets[target_index];
        const auto id = target.StoreId;
        const Mesh mesh{meshes, id};
        VertexPositionEditPushConstants pc{
            .Connectivity = meshes.GetConnectivityRef(id),
            .VertexSlot = meshes.Slots().Vertices,
            .CornerSlot = meshes.Arenas().FaceCorners.Buffer.Slot,
            .FaceCount = mesh.FaceCount(),
            .Axes = axes,
            .Factor = factor,
            .FaceNormalSlot = meshes.Arenas().BaseFaceNormals.Buffer.Slot,
            .VertexNormalSlot = meshes.Arenas().BaseVertexNormals.Buffer.Slot,
            .Flags = flags,
            .Operation = operation,
        };
        if (!planar && operation == PositionEditOp::ShrinkFatten)
            pc.Faces = SeedElementWorkHandles(chain.Scratch, meshes.Arenas().FaceTriangles.Capacity(), target.Selection.Faces);
        PositionBatch batch{.Index = uint32_t(jobs.size())};
        const auto world = transformed ? target.World : Transform{};
        const auto inverse = Conjugate(world.R);
        const auto local_direction = [&](vec3 direction) {
            auto local = inverse * direction;
            for (uint32_t axis = 0u; axis < 3u; ++axis) local[axis] = world.S[axis] != 0.f ? local[axis] / world.S[axis] : 0.f;
            return local;
        };
        if (flatten_view) {
            pc.Direction = local_direction(direction);
            const float length = Length(pc.Direction);
            if (!(length > 0.f) || !std::isfinite(length)) continue;
            pc.Direction /= length;
        }
        if (centered) {
            pc.Center = local_direction(center - world.P);
            pc.Direction = local_direction(direction);
            pc.Gradient = (inverse * gradient) * world.S;
        }
        if (warp || bend) {
            const PositionPlane plane{(inverse * plane_x) * world.S, (inverse * plane_y) * world.S, local_direction(plane_x), local_direction(plane_y), {Dot(world.P - plane_center, plane_x), Dot(world.P - plane_center, plane_y)}};
            if (warp) pc.Parameters = chain.Upload(as_bytes(WarpParameters{plane, std::min(warp->Min, warp->Max), std::max(warp->Min, warp->Max)}));
            else pc.Parameters = chain.Upload(as_bytes(BendParameters{plane, bend->Radius, bend_pivot}));
        }
        if (transform) {
            auto frame = target.World;
            auto delta = transform->Delta;
            frame.R = Normalize(frame.R);
            delta.R = Normalize(delta.R);
            pc.Parameters = chain.Upload(as_bytes(TransformPositionParameters{frame, delta, transform->Pivot}));
        }
        if (randomize) pc.Parameters = chain.Upload(as_bytes(RandomizeParameters{std::clamp(randomize->Uniform, 0.f, 1.f), std::clamp(randomize->Normal, 0.f, 1.f), randomize->Seed}));
        if (slide || edge_slide) {
            const auto &selected = target.Selection.Vertices;
            uint32_t reference = selected.empty() ? InvalidOffset : selected.front();
            if (std::ranges::binary_search(selected, target.Reference)) reference = target.Reference;
            if (edge_slide) {
                const auto directions = PlanEdgeSlide(mesh, target.Selection, inverse * Normalize(edge_slide->Direction), world.S, reference);
                if (directions.empty()) continue;
                pc.Parameters = chain.Upload(as_bytes(directions));
                if (edge_slide->Even) {
                    uint32_t rank = 0u, reference_rank = 0u;
                    for (const auto v : selected) {
                        if (v == reference) reference_rank = rank;
                        ++rank;
                    }
                    const auto &ref = directions[reference_rank];
                    pc.ReductionResult = chain.Upload(as_bytes(Length(ref.Positive - ref.Negative)));
                }
            } else {
                const VertexSlideParameters parameters{inverse * Normalize(slide->Direction), world.S, reference};
                pc.Parameters = chain.Upload(as_bytes(parameters));
                if (slide->Even) pc.ReductionResult = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(1u).Offset};
            }
        }
        std::vector<uint32_t> vertices;
        if (flatten) {
            auto plan = PlanFlatten(mesh, target.Selection);
            if (plan.Vertices.empty()) continue;
            chain.Scratch.ReserveAdditional(plan.Words.size() + plan.Groups.size() + 4ull * plan.Vertices.size());
            // Flatten's parameters hold packed groups; Planes indexes their offsets.
            pc.Parameters = chain.Upload(as_bytes(plan.Words));
            pc.Planes = chain.Upload(as_bytes(plan.Groups));
            pc.PlaneCount = uint32_t(plan.Groups.size());
            batch.Batches = std::move(plan.Batches);
            vertices = std::move(plan.Vertices);
        } else if (planar) {
            std::vector<uint32_t> faces;
            for (const auto f : target.Selection.Faces)
                if (mesh.GetValence(he::FH{f}) > 3u) faces.push_back(f);
            if (faces.empty()) continue;
            // The existing face seed gathers only connectivity handles on the host.
            // Its vertex list also gives history the exact pages to capture.
            ClosureSeed seed{.Work = SeedElementWorkHandles(chain.Scratch, meshes.Arenas().FaceTriangles.Capacity(), faces), .Count = uint32_t(faces.size())};
            for (const auto f : faces)
                for (const auto v : mesh.fv_range(he::FH{f})) seed.Vertices.push_back(*v);
            SortUnique(seed.Vertices);
            pc.Faces = seed.Work;
            pc.PlaneCount = seed.Count;
            pc.Planes = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(4u * seed.Count).Offset};
            vertices = std::move(seed.Vertices);
        } else if (edge_chains || ordered_chains) {
            auto plan = circle ? PlanCircularize(mesh, target.Selection, target.Excluded) : curve ? PlanCurveBetweenSelected(mesh, target.Selection, target.Excluded, curve->Extend) :
                                                                                                    PlanSelectedEdgeChains(mesh, target.Selection, relax);
            if (plan.Outputs.empty()) continue;
            const uint32_t stride = circle ? 0u : curve ? 7u :
                relax                                   ? 10u :
                                                          6u;
            const uint32_t extra = circle ? 0u : curve ? 2u :
                                                         1u;
            chain.Scratch.ReserveAdditional(uint64_t(stride + 1u) * plan.Inputs.size() + (sizeof(EdgeChain) / sizeof(uint32_t) + extra) * plan.Chains.size() + 4ull * plan.Outputs.size() + plan.Phases.size());
            const auto inputs = chain.Scratch.Allocate(std::span<const uint32_t>{plan.Inputs});
            const auto phases = chain.Scratch.Allocate(std::span<const uint32_t>{plan.Phases});
            const auto work = chain.Scratch.Allocate(stride * uint32_t(plan.Inputs.size()) + extra * uint32_t(plan.Chains.size()));
            uint32_t work_offset = work.Offset;
            for (uint32_t i = 0u; i < plan.Chains.size(); ++i) {
                auto &descriptor = plan.Chains[i];
                descriptor.WorkOffset = work_offset;
                work_offset += stride * descriptor.Count + extra;
                descriptor.InputOffset += inputs.Offset;
                if (relax || curve) descriptor.PhaseOffset += phases.Offset;
            }
            pc.Parameters = chain.Upload(as_bytes(plan.Chains));
            pc.ChainCount = uint32_t(plan.Chains.size());
            if (ordered_chains) batch.Batches = std::move(plan.Batches);
            if (circle) pc.Direction = {circle->Radius, circle->Angle, 0.f};
            vertices = std::move(plan.Outputs);
        } else if (symmetry) {
            const auto plan = PlanSymmetrySnap(meshes, mesh, target.Selection, std::countr_zero(axes), symmetry->Threshold, symmetry->Center);
            if (plan.empty()) continue;
            std::vector<uint32_t> partners;
            for (const auto [v, partner] : plan) {
                vertices.push_back(v);
                partners.push_back(partner);
            }
            pc.Parameters = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(std::span<const uint32_t>{partners}).Offset};
        } else {
            vertices = target.Selection.Vertices;
        }
        if (vertices.empty()) continue;
        pc.Handles = chain.Upload(as_bytes(vertices));
        pc.Count = uint32_t(vertices.size());
        // Fitting jobs keep their own output order; only history capture needs sorted handles.
        if (edge_chains || ordered_chains) std::ranges::sort(vertices);
        std::vector<Range> runs;
        ForEachIndexRun(vertices, [&](size_t first, size_t count) { runs.push_back({vertices[first], uint32_t(count)}); });
        if (statistics) {
            pc.ReductionBlocks = {chain.Scratch.Buffer.Slot, partial_offset};
            partial_offset += 2u * ((pc.Count + 255u) / 256u);
        }
        pc.Positions = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(3u * pc.Count).Offset};
        meshes.Arenas().Vertices.Buffer.CaptureWriteRanges(runs, sizeof(Vertex));
        changes.push_back({target_index, std::move(vertices), std::move(runs)});
        jobs.push_back(pc);
        if (!batch.Batches.empty()) batches.push_back(std::move(batch));
    }
    if (jobs.empty()) return {};
    uint64_t vertex_count = 0u, plane_count = 0u;
    for (const auto &pc : jobs) {
        vertex_count += pc.Count;
        plane_count += pc.PlaneCount;
    }
    profile::RecordCounter("PositionEditVertices", vertex_count);
    profile::RecordCounter("PositionEditPlanes", plane_count);
    uint64_t curve_chain_count = 0u, curve_batch_count = 0u;
    if (ordered_chains)
        for (const auto &job : batches) {
            curve_chain_count += jobs[job.Index].ChainCount;
            curve_batch_count += job.Batches.size();
        }
    profile::RecordCounter("CurveEditChains", curve_chain_count);
    profile::RecordCounter("CurveEditBatches", curve_batch_count);
    if (statistics) {
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Groups(pipelines[MeshPass::PositionStatisticsGather], pc, (pc.Count + 255u) / 256u);
        });
        const auto reduce = [&](SlotOffset input, uint32_t count) {
            // Sphere needs a final normalization even for one partial; bounds
            // already contain their result when a mesh fits in one group.
            if (sphere || count > 1u) {
                do {
                    const uint32_t next = (count + 255u) / 256u;
                    const SlotOffset output{chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * next).Offset};
                    const PositionReducePushConstants pc{.Input = input, .Output = output, .Count = count, .Scale = sphere && next == 1u ? 1.f / float(selected_count) : 1.f, .Bounds = !sphere};
                    chain.Groups(pipelines[MeshPass::PositionStatisticsReduce], pc, next);
                    input = output;
                    count = next;
                } while (count > 1u);
            }
            return input;
        };
        if (sphere) {
            const auto result = reduce({chain.Scratch.Buffer.Slot, partials.Offset}, reduction_blocks);
            for (auto &pc : jobs) pc.ReductionResult = result;
        } else
            for (auto &pc : jobs) pc.ReductionResult = reduce(pc.ReductionBlocks, (pc.Count + 255u) / 256u);
        profile::RecordCounter(sphere ? "PositionRadiusBlocks" : "PositionWarpBoundsBlocks", reduction_blocks);
    }
    if (planar) chain.Concurrent([&] {
        for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::PlanarFacePlanes], pc, pc.PlaneCount);
    });
    if (slide && slide->Even) chain.Concurrent([&] {
        for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::VertexSlideReference], pc, 1u);
    });
    for (uint32_t step = 0u; step < repeat; ++step) {
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Threads(pipelines[gather], pc, edge_chains ? pc.ChainCount : pc.Count);
        });
        uint32_t depths = 0u;
        for (const auto &job : batches) depths = std::max(depths, uint32_t(job.Batches.size()));
        for (uint32_t depth = 0u; depth < depths; ++depth) chain.Concurrent([&] {
            for (const auto &job : batches)
                if (depth < job.Batches.size()) {
                    const auto batch = job.Batches[depth];
                    auto pc = jobs[job.Index];
                    if (flatten) {
                        pc.Planes.Offset += batch.Offset;
                        chain.Groups(pipelines[MeshPass::FlattenGroups], pc, batch.Count);
                    } else {
                        pc.Parameters.Offset += batch.Offset * uint32_t(sizeof(EdgeChain) / sizeof(uint32_t));
                        pc.ChainCount = batch.Count;
                        chain.Threads(pipelines[circle ? MeshPass::Circularize : MeshPass::CurveBetweenSelected], pc, pc.ChainCount);
                    }
                }
        });
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::WriteEditedPositions], pc, pc.Count);
        });
    }
    return changes;
}
