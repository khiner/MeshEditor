#include "mesh/PositionOperations.h"
#include "Profile.h"
#include "SortUnique.h"
#include "gpu/FaceAttributeEditPushConstants.h"
#include "mesh/EdgeChains.h"
#include "mesh/EdgeSlide.h"
#include "mesh/Flatten.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/SnapSymmetry.h"
#include "metal/Dispatch.h"
#include "numeric/QuaternionMath.h"
#include "render/ElementWorkOps.h"

namespace {
bool ValidPlane(quat orientation, vec3 center, float roll) {
    const float norm = orientation.x * orientation.x + orientation.y * orientation.y + orientation.z * orientation.z + orientation.w * orientation.w;
    return norm > 0.f && std::isfinite(norm) && Finite(center) && std::isfinite(roll);
}
} // namespace

std::optional<FaceAttributeOperationChange> EncodeFaceAttributeOperation(MeshStore &meshes, MeshPipelines &pipelines, mtl::ComputeChain &chain, const PositionOperationTarget &target, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets || operation > 2u) return {};
    const auto id = target.StoreId;
    const SelectedHandles selection{meshes, id, target.Selection};
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    if (!(record.CornerAttributes & (colors ? MeshAttributeBit_Color0 : MeshAttributeBit_TexCoord0 << uv_set))) return {};
    const auto tangents = colors ? Range{} : record.CornerTangents;
    const auto &faces = selection->Faces;
    uint32_t incidence = 0u;
    for (const auto f : faces) {
        const auto range = a.FaceRanges.Get({a.FaceTriangles.Address(record.FaceData, ElementHandle{f}).Value(), 1u})[0];
        const Range corners{range.x, range.y - range.x};
        if (colors) a.CornerColors.Capture(record.CornerColors, corners);
        else a.CornerUvs[uv_set].Capture(record.CornerUvs[uv_set], corners);
        if (tangents.Count) a.CornerTangents.Capture(tangents, corners);
        incidence += corners.Count;
    }
    if (faces.empty()) return {};
    const FaceAttributeOperationChange change{{.Work = SeedElementWorkHandles(chain.Scratch, a.FaceTriangles.HandleBound(record.FaceData), faces), .Count = uint32_t(faces.size()), .Incidence = incidence}};
    const FaceAttributeEditPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Faces = change.Faces.Work,
        .Attribute = colors ? a.CornerColors.Ref(record.CornerColors) : a.CornerUvs[uv_set].Ref(record.CornerUvs[uv_set]),
        .Tangents = a.CornerTangents.Ref(tangents),
        .Count = change.Faces.Count,
        .Operation = operation,
    };
    chain.Threads(pipelines[colors ? MeshPass::EditFaceColors : MeshPass::EditFaceUvs], pc, pc.Count);
    profile::RecordCounter("AttributeEditFaces", pc.Count);
    profile::RecordCounter("AttributeEditCorners", incidence);
    return change;
}

std::optional<PositionOperationChange> EncodePositionOperations(MeshStore &meshes, MeshPipelines &pipelines, mtl::ComputeChain &chain, const PositionOperationTarget &target, PositionEditOp operation, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    const auto &[axes, requested_flags, direction, gradient, center, warp, bend, randomize, slide, edge_slide, symmetry, curve, circle, transform, attribute, factor_attribute, sphere_radius] = options;
    uint32_t flags = requested_flags;
    if (!repeat || axes > 7u || uint32_t(operation) > uint32_t(PositionEditOp::Attribute)) return {};
    if (bool(warp) != (operation == PositionEditOp::Warp) || bool(bend) != (operation == PositionEditOp::Bend) ||
        bool(randomize) != (operation == PositionEditOp::Randomize) || bool(slide) != (operation == PositionEditOp::VertexSlide) ||
        bool(edge_slide) != (operation == PositionEditOp::EdgeSlide) || bool(symmetry) != (operation == PositionEditOp::SnapSymmetry) ||
        bool(transform) != (operation == PositionEditOp::Transform) || bool(attribute) != (operation == PositionEditOp::Attribute) || ((curve || circle) && operation != PositionEditOp::Copy) || (curve && circle)) return {};
    if (!Finite(center) || !Finite(direction) || !Finite(gradient)) return {};
    if (warp && (!ValidPlane(warp->Orientation, warp->Center, warp->OffsetAngle) || (!warp->AutoRange && (!std::isfinite(warp->Min) || !std::isfinite(warp->Max) || warp->Min == warp->Max)))) return {};
    if (bend && (!ValidPlane(bend->Orientation, bend->Center, bend->OffsetAngle) || !std::isfinite(bend->Radius) || bend->Radius == 0.f)) return {};
    if (randomize && (!std::isfinite(randomize->Uniform) || !std::isfinite(randomize->Normal))) return {};
    for (const auto &parameters : {slide, edge_slide}) {
        if (!parameters) continue;
        if (!(Length(parameters->Direction) > 0.f) || !std::isfinite(Length(parameters->Direction))) return {};
        flags = (flags & ~(PositionEditSlideEven | PositionEditSlideFlipped)) |
            (parameters->Even ? PositionEditSlideEven : 0u) | (parameters->Flipped ? PositionEditSlideFlipped : 0u);
    }
    if (circle && (!std::isfinite(circle->Radius) || !std::isfinite(circle->Angle))) return {};
    if (symmetry && (!std::has_single_bit(axes) || !std::isfinite(symmetry->Threshold) || symmetry->Threshold <= 0.f)) return {};
    if (transform && (!Finite(transform->Delta.P) || !Finite(transform->Delta.S) || !Finite(transform->Pivot) || !ValidPlane(transform->Delta.R, {}, 0.f))) return {};
    const auto id = target.StoreId;
    const SelectedHandles selection{meshes, id, target.Selection}, excluded{meshes, id, target.Excluded};
    if (warp) flags = (flags & ~PositionEditWarpAutoRange) | (warp->AutoRange ? PositionEditWarpAutoRange : 0u);
    const bool zero_moves = symmetry || warp || (slide && slide->Even && slide->Flipped) || (edge_slide && edge_slide->Even);
    if (!axes || (!zero_moves && factor == 0.f) || !std::isfinite(factor)) return {};
    const bool planar = operation == PositionEditOp::Planar;
    const bool ordered_chains = curve || circle;
    const bool flatten = operation == PositionEditOp::Copy && !ordered_chains;
    const bool flatten_direction = flatten && (flags & PositionEditFlattenDirection);
    const bool relax = operation == PositionEditOp::RelaxEdgeLoops;
    const bool edge_chains = relax || operation == PositionEditOp::SpaceEvenly;
    const bool sphere = operation == PositionEditOp::ToSphere;
    const auto gather = edge_chains ? (relax ? MeshPass::RelaxEdgeLoopsGather : MeshPass::SpaceEvenlyGather) : MeshPass::PositionVerticesGather;
    const bool centered = sphere || operation == PositionEditOp::PushPull || operation == PositionEditOp::Shear;
    const bool transformed = centered || warp || bend || slide || edge_slide || flatten_direction || transform;
    if (slide && !(flags & PositionEditSlideUnclamped)) factor = std::clamp(factor, 0.f, 1.f);
    else if (edge_slide && !(flags & PositionEditSlideUnclamped)) factor = std::clamp(factor, -1.f, 1.f);
    else if (transform || attribute || sphere || symmetry || flatten || ordered_chains || operation == PositionEditOp::SpaceEvenly) factor = std::clamp(factor, 0.f, 1.f);
    else if (planar || operation == PositionEditOp::Smooth) factor = std::clamp(factor, -10.f, 10.f);
    if (!zero_moves && factor == 0.f) return {};
    vec3 plane_x{}, plane_y{}, plane_center{};
    float bend_pivot = 0.f;
    if (transformed && (!Finite(target.World.P) || !Finite(target.World.S) || !ValidPlane(target.World.R, {}, 0.f))) return {};
    if (transform && (target.World.S.x == 0.f || target.World.S.y == 0.f || target.World.S.z == 0.f)) return {};
    if (warp || bend) {
        const auto rotation = Normalize(warp ? warp->Orientation : bend->Orientation);
        const auto right = rotation * vec3{1, 0, 0}, up = rotation * vec3{0, 1, 0};
        const float roll = warp ? warp->OffsetAngle : bend->OffsetAngle;
        const float c = std::cos(roll), s = std::sin(roll);
        plane_x = c * right - s * up;
        plane_y = s * right + c * up;
        plane_center = warp ? warp->Center : bend->Center;
        if (bend) {
            // Equivalent to Blender's shell_angle_to_dist, with a stable small-angle denominator and saturation past a quarter turn.
            const float angle = std::abs(factor);
            const float shell = angle >= std::numbers::pi_v<float> * .5f ? 1.f : 1.f / std::sin(angle);
            bend_pivot = -std::copysign(1.f, factor) * bend->Radius * shell;
            if (!std::isfinite(bend_pivot)) return {};
        }
    }
    const bool statistics = (sphere && !sphere_radius) || (warp && warp->AutoRange);
    const uint64_t selected_count = selection->Vertices.size();
    const uint32_t reduction_blocks = statistics ? TileCount(selected_count, TileElements) : 0u;
    if (!planar) {
        // Symmetry can also move an unselected partner for each selected vertex.
        const uint32_t vertex_words = edge_slide || symmetry ? 10u : 4u;
        const uint32_t parameter_words = transform ? sizeof(TransformPositionParameters) / sizeof(uint32_t) : attribute ? sizeof(NamedAttributeRef) / sizeof(uint32_t) :
            warp || bend                                                                                                ? 18u :
            randomize                                                                                                   ? 3u :
            slide                                                                                                       ? 8u :
            edge_slide                                                                                                  ? 1u :
                                                                                                                          0u;
        chain.Scratch.ReserveAdditional(vertex_words * selected_count + 4ull * reduction_blocks + 1u + parameter_words);
    }
    const auto partials = chain.Scratch.Allocate(2u * reduction_blocks);
    const Mesh mesh{meshes, id};
    const auto face_bound = meshes.Arenas().FaceTriangles.HandleBound(meshes.Get(id).FaceData);
    VertexPositionEditPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .VertexSlot = meshes.Slots().Vertices,
        .Axes = axes,
        .Factor = factor,
        .FaceNormalSlot = meshes.Arenas().BaseFaceNormals.Buffer.Slot,
        .VertexNormalSlot = meshes.Arenas().BaseVertexNormals.Buffer.Slot,
        .Flags = flags,
        .Operation = operation,
        .FactorAttribute = factor_attribute,
    };
    if (!planar && operation == PositionEditOp::ShrinkFatten) pc.Faces = SeedElementWorkHandles(chain.Scratch, face_bound, selection->Faces);
    std::vector<Range> batches;
    const auto world = transformed ? target.World : Transform{};
    const auto inverse = Conjugate(world.R);
    const auto local_direction = [&](vec3 direction) {
        auto local = inverse * direction;
        for (uint32_t axis = 0u; axis < 3u; ++axis) local[axis] = world.S[axis] != 0.f ? local[axis] / world.S[axis] : 0.f;
        return local;
    };
    if (flatten_direction) {
        pc.Direction = local_direction(direction);
        const float length = Length(pc.Direction);
        if (!(length > 0.f) || !std::isfinite(length)) return {};
        pc.Direction /= length;
    }
    if (centered) {
        pc.Center = local_direction(center - world.P);
        pc.Direction = local_direction(direction);
        pc.Gradient = (inverse * gradient) * world.S;
    }
    if (warp || bend) {
        const PositionPlane plane{(inverse * plane_x) * world.S, (inverse * plane_y) * world.S, local_direction(plane_x), local_direction(plane_y), {Dot(world.P - plane_center, plane_x), Dot(world.P - plane_center, plane_y)}};
        if (warp) pc.Parameters = chain.Upload(BytesOf(WarpParameters{plane, std::min(warp->Min, warp->Max), std::max(warp->Min, warp->Max)}));
        else pc.Parameters = chain.Upload(BytesOf(BendParameters{plane, bend->Radius, bend_pivot}));
    }
    if (transform) {
        auto frame = target.World;
        auto delta = transform->Delta;
        frame.R = Normalize(frame.R);
        delta.R = Normalize(delta.R);
        pc.Parameters = chain.Upload(BytesOf(TransformPositionParameters{frame, delta, transform->Pivot, transform->Mirror}));
    }
    if (attribute) pc.Parameters = chain.Upload(BytesOf(*attribute));
    if (randomize) pc.Parameters = chain.Upload(BytesOf(RandomizeParameters{std::clamp(randomize->Uniform, 0.f, 1.f), std::clamp(randomize->Normal, 0.f, 1.f), randomize->Seed}));
    if (slide || edge_slide) {
        const auto &selected = selection->Vertices;
        uint32_t reference = selected.empty() ? InvalidOffset : selected.front();
        if (std::ranges::binary_search(selected, target.Reference)) reference = target.Reference;
        if (edge_slide) {
            const auto directions = PlanEdgeSlide(meshes, pipelines, mesh, *selection, inverse * Normalize(edge_slide->Direction), world.S, reference);
            if (directions.empty()) return {};
            pc.Parameters = chain.Upload(BytesOf(directions));
            if (edge_slide->Even) {
                const auto &ref = directions[std::ranges::lower_bound(selected, reference) - selected.begin()];
                pc.ReductionResult = chain.Upload(BytesOf(Length(ref.Positive - ref.Negative)));
            }
        } else {
            const VertexSlideParameters parameters{inverse * Normalize(slide->Direction), world.S, reference};
            pc.Parameters = chain.Upload(BytesOf(parameters));
            if (slide->Even) pc.ReductionResult = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(1u).Offset};
        }
    }
    std::vector<uint32_t> vertices;
    if (flatten) {
        auto plan = PlanFlatten(meshes, pipelines, mesh, *selection);
        if (plan.Vertices.empty()) return {};
        chain.Scratch.ReserveAdditional(plan.Words.size() + plan.Groups.size() + 4ull * plan.Vertices.size());
        // Parameters hold Flatten's packed groups, and Planes holds their offsets.
        pc.Parameters = chain.Upload(BytesOf(plan.Words));
        pc.Planes = chain.Upload(BytesOf(plan.Groups));
        pc.PlaneCount = uint32_t(plan.Groups.size());
        batches = std::move(plan.Batches);
        vertices = std::move(plan.Vertices);
    } else if (planar) {
        const auto corners = FaceCorners(meshes, pipelines, id, selection->Faces);
        std::vector<uint32_t> faces, face_vertices;
        for (uint32_t i = 0u; i < selection->Faces.size(); ++i) {
            if (corners.Of(i).size() <= 3u) continue;
            faces.push_back(selection->Faces[i]);
            for (const auto &corner : corners.Of(i)) face_vertices.push_back(corner.Vertex);
        }
        if (faces.empty()) return {};
        SortUnique(face_vertices);
        pc.Faces = SeedElementWorkHandles(chain.Scratch, face_bound, faces);
        pc.PlaneCount = uint32_t(faces.size());
        pc.Planes = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(4u * pc.PlaneCount).Offset};
        vertices = std::move(face_vertices);
    } else if (edge_chains || ordered_chains) {
        auto plan = circle ? PlanCircularize(meshes, pipelines, mesh, *selection, *excluded) : curve ? PlanCurveBetweenSelected(meshes, pipelines, mesh, *selection, *excluded, curve->Extend) :
                                                                                                       PlanSelectedEdgeChains(meshes, pipelines, mesh, *selection, relax);
        if (plan.Outputs.empty()) return {};
        const uint32_t stride = circle ? 0u : curve ? 7u :
            relax                                   ? 10u :
                                                      6u;
        const uint32_t extra = circle ? 0u : curve ? 2u :
                                                     1u;
        chain.Scratch.ReserveAdditional(uint64_t(stride + 1u) * plan.Inputs.size() + (sizeof(EdgeChain) / sizeof(uint32_t) + extra) * plan.Chains.size() + 4ull * plan.Outputs.size() + plan.Phases.size());
        const auto inputs = chain.Scratch.Allocate(plan.Inputs);
        const auto phases = chain.Scratch.Allocate(plan.Phases);
        const auto work = chain.Scratch.Allocate(stride * uint32_t(plan.Inputs.size()) + extra * uint32_t(plan.Chains.size()));
        uint32_t work_offset = work.Offset;
        for (auto &descriptor : plan.Chains) {
            descriptor.WorkOffset = work_offset;
            work_offset += stride * descriptor.Count + extra;
            descriptor.InputOffset += inputs.Offset;
            if (relax || curve) descriptor.PhaseOffset += phases.Offset;
        }
        pc.Parameters = chain.Upload(BytesOf(plan.Chains));
        pc.ChainCount = uint32_t(plan.Chains.size());
        if (ordered_chains) batches = std::move(plan.Batches);
        if (circle) pc.Direction = {circle->Radius, circle->Angle, 0.f};
        vertices = std::move(plan.Outputs);
    } else if (symmetry) {
        const auto plan = PlanSymmetrySnap(meshes, pipelines, mesh, *selection, std::countr_zero(axes), symmetry->Threshold, symmetry->Center);
        if (plan.empty()) return {};
        std::vector<uint32_t> partners;
        for (const auto [v, partner] : plan) {
            vertices.push_back(v);
            partners.push_back(partner);
        }
        pc.Parameters = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(partners).Offset};
    } else {
        vertices = selection->Vertices;
    }
    if (vertices.empty()) return {};
    pc.Handles = chain.Upload(BytesOf(vertices));
    pc.Count = uint32_t(vertices.size());
    // Only history capture needs sorted handles, and fitting jobs keep their own output order.
    if (edge_chains || ordered_chains) std::ranges::sort(vertices);
    std::vector<Range> runs;
    ForEachIndexRun(vertices, [&](size_t first, size_t count) { runs.push_back({vertices[first], uint32_t(count)}); });
    if (statistics) pc.ReductionBlocks = {chain.Scratch.Buffer.Slot, partials.Offset};
    pc.Positions = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(3u * pc.Count).Offset};
    const auto &arena = meshes.Arenas().Vertices;
    const auto set = meshes.Get(id).Vertices;
    for (const auto run : runs) arena.ForEachAddressRun(set, ElementHandle{run.Offset}, run.Count, [&](Range addresses) { arena.Buffer.CaptureWrite(uint64_t(addresses.Offset) * sizeof(Vertex), uint64_t(addresses.Count) * sizeof(Vertex)); });
    PositionOperationChange change{std::move(vertices), std::move(runs)};
    profile::RecordCounter("PositionEditVertices", pc.Count);
    profile::RecordCounter("PositionEditPlanes", pc.PlaneCount);
    profile::RecordCounter("CurveEditChains", ordered_chains && !batches.empty() ? pc.ChainCount : 0u);
    profile::RecordCounter("CurveEditBatches", ordered_chains ? batches.size() : 0u);
    if (statistics) {
        chain.Groups(pipelines[MeshPass::PositionStatisticsGather], pc, TileCount(pc.Count, TileElements));
        const auto reduce = [&](SlotOffset input, uint32_t count) {
            // Sphere normalizes even one partial, and other bounds already hold their result when a mesh fits in one group.
            if (sphere || count > 1u) {
                do {
                    const uint32_t next = TileCount(count, TileElements);
                    const SlotOffset output{chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * next).Offset};
                    const PositionReducePushConstants reduction{.Input = input, .Output = output, .Count = count, .Scale = sphere && next == 1u ? 1.f / float(selected_count) : 1.f, .Bounds = !sphere};
                    chain.Groups(pipelines[MeshPass::PositionStatisticsReduce], reduction, next);
                    input = output;
                    count = next;
                } while (count > 1u);
            }
            return input;
        };
        pc.ReductionResult = reduce(pc.ReductionBlocks, reduction_blocks);
        profile::RecordCounter(sphere ? "PositionRadiusBlocks" : "PositionWarpBoundsBlocks", reduction_blocks);
    }
    if (sphere && sphere_radius) pc.ReductionResult = chain.Upload(BytesOf(*sphere_radius));
    if (planar) chain.Threads(pipelines[MeshPass::PlanarFacePlanes], pc, pc.PlaneCount);
    if (slide && slide->Even) chain.Threads(pipelines[MeshPass::VertexSlideReference], pc, 1u);
    for (uint32_t step = 0u; step < repeat; ++step) {
        chain.Threads(pipelines[gather], pc, edge_chains ? pc.ChainCount : pc.Count);
        for (const auto batch : batches) {
            auto batch_pc = pc;
            if (flatten) {
                batch_pc.Planes.Offset += batch.Offset;
                chain.Groups(pipelines[MeshPass::FlattenGroups], batch_pc, batch.Count);
            } else {
                batch_pc.Parameters.Offset += batch.Offset * uint32_t(sizeof(EdgeChain) / sizeof(uint32_t));
                batch_pc.ChainCount = batch.Count;
                chain.Threads(pipelines[circle ? MeshPass::Circularize : MeshPass::CurveBetweenSelected], batch_pc, batch_pc.ChainCount);
            }
        }
        chain.Threads(pipelines[MeshPass::WriteEditedPositions], pc, pc.Count);
    }
    return change;
}
