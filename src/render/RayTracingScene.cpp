#include "render/RayTracingScene.h"
#include "Profile.h"
#include "armature/ArmatureComponents.h"
#include "gpu/RayEnvironment.h"
#include "gpu/RayTracingRecord.h"
#include "mesh/MeshStore.h"
#include "metal/MetalCpp.h"
#include "numeric/MatrixMath.h"
#include "render/GpuBuffers.h"
#include "render/ObjectGeometry.h"
#include "render/Pipelines.h"
#include "state/Registry.h"
#include <Metal/MTL4AccelerationStructure.hpp>
#include <map>
#include <set>

namespace {
struct Structure {
    mtl::Resident<MTL::AccelerationStructure> Handle;
    mtl::ResidentBuffer Scratch;
    Structure(const mtl::Context &ctx, MTL4::AccelerationStructureDescriptor *desc) {
        const auto sizes = ctx.Device->accelerationStructureSizes(desc);
        Handle = {ctx, NS::TransferPtr(ctx.Device->newAccelerationStructure(sizes.accelerationStructureSize))};
        if (!Handle) throw std::runtime_error("Could not allocate ray tracing acceleration structure");
        Scratch = {ctx, std::max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize)};
    }
    void Build(mtl::ComputeEncoder encoder, MTL4::AccelerationStructureDescriptor *desc, bool refit = false) {
        const MTL4::BufferRange scratch{Scratch.Address(), Scratch.Length()};
        if (refit) encoder->refitAccelerationStructure(Handle.get(), desc, Handle.get(), scratch);
        else encoder->buildAccelerationStructure(Handle.get(), desc, scratch);
    }
};
} // namespace
struct RayTracingScene::Storage {
    explicit Storage(const mtl::Context &ctx) : Context(ctx), Environment(ctx, RayEnvironmentCount * sizeof(float) + sizeof(RayMediumState)) { CameraMedia() = {}; }
    RayMediumState &CameraMedia() { return *reinterpret_cast<RayMediumState *>(Environment.Contents().data() + RayEnvironmentCount * sizeof(float)); }
    const mtl::Context &Context;
    struct Geometry {
        std::unique_ptr<Structure> Acceleration;
        mtl::ResidentBuffer Indices;
        std::pair<uint64_t, uint64_t> VertexBuffer{}; // Address and byte length used by the acceleration structure.
        uint32_t VertexBase{}, VertexCount{};
        MTL::IndexType IndexType{MTL::IndexTypeUInt32};
        std::tuple<uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t> Revision{};
        uint32_t TriangleOffset{};
    };
    std::map<std::pair<uint32_t, uint32_t>, Geometry> Bottom;
    uint64_t Epoch{};
    uint64_t Revision{}, GeometryRevision{};
    bool PositionsDirty{}, CanonicalGeometryDirty{}, PoseSnapshotValid{}, CapturePose{}, DeformationHistory{};
    mtl::ResidentBuffer PreviousPositions;
    std::vector<uint32_t> PositionBlocks;
    bool PositionLayoutDirty{};
    std::unique_ptr<Structure> Top;
    mtl::ResidentBuffer Instances, HitInstances, Triangles, Environment, Motion;
};
RayTracingScene::RayTracingScene(const mtl::Context &ctx) : Data(std::make_unique<Storage>(ctx)) {}
RayTracingScene::~RayTracingScene() = default;
bool RayTracingScene::HasDeformationHistory() const { return Data->DeformationHistory; }
void RayTracingScene::BindPreviousPositions(mtl::Command &cmd) const { cmd.SetBuffer(*Data->PreviousPositions, 0, 12); }
void RayTracingScene::CapturePositions(mtl::PassChain &chain, const GpuBuffers &buffers) {
    auto &out = *Data;
    if (!out.CapturePose) return;
    const auto &source = buffers.PosedPositions.Values.Buffer;
    // One address per source block precedes tightly packed 256-position blocks.
    const size_t map_bytes = (size_t(out.PositionBlocks.back()) + 1u) * sizeof(uint32_t);
    const size_t payload = (map_bytes + sizeof(vec3) - 1u) / sizeof(vec3) * sizeof(vec3);
    const size_t block_bytes = 256u * sizeof(vec3);
    const size_t bytes = payload + out.PositionBlocks.size() * block_bytes;
    if (bytes / sizeof(vec3) > InvalidOffset) throw std::length_error("Ray position history exceeds its address space.");
    if (!out.PreviousPositions || out.PreviousPositions.Length() != bytes) {
        out.PreviousPositions = {out.Context, bytes};
        out.PositionLayoutDirty = true;
    }
    if (out.PositionLayoutDirty) {
        auto *map = reinterpret_cast<uint32_t *>(out.PreviousPositions.Contents().data());
        std::fill_n(map, map_bytes / sizeof(uint32_t), InvalidOffset);
        for (size_t i = 0; i < out.PositionBlocks.size(); ++i)
            map[out.PositionBlocks[i]] = uint32_t(payload / sizeof(vec3) + i * 256u);
        out.PositionLayoutDirty = false;
    }
    const auto copy = chain.BeginBlit(MTL::StageDispatch | MTL::StageAccelerationStructure);
    for (size_t first = 0; first < out.PositionBlocks.size();) {
        size_t end = first + 1;
        while (end < out.PositionBlocks.size() && out.PositionBlocks[end] == out.PositionBlocks[end - 1] + 1u) ++end;
        copy->copyFromBuffer(*source, size_t(out.PositionBlocks[first]) * block_bytes, *out.PreviousPositions, payload + first * block_bytes, (end - first) * block_bytes);
        first = end;
    }
    out.PoseSnapshotValid = true;
    out.CapturePose = false;
    profile::RecordCounter("RayPreviousPositionBytes", bytes);
    profile::RecordCounter("RayPreviousPositionCopiedBytes", out.PositionBlocks.size() * block_bytes);
}
void RayTracingScene::InvalidatePositions(bool canonical_geometry) {
    Data->PositionsDirty = true;
    Data->CanonicalGeometryDirty |= canonical_geometry;
}
bool RayTracingScene::PreviousCameraInsideMedium() const { return Data->CameraMedia().Count != 0u; }
uint32_t RayTracingScene::InstanceCount() const { return Data->Top ? uint32_t(Data->Instances.Length() / sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor)) : 0u; }
uint64_t RayTracingScene::Revision() const { return Data->Revision; }
uint64_t RayTracingScene::GeometryRevision() const { return Data->GeometryRevision; }
RayTracingMaterials RayTracingMaterialUsage(const MeshStore &meshes, const GpuBuffers &buffers, const SceneViewUBO &view, bool classify_casters) {
    RayTracingMaterials usage;
    const auto states = buffers.InstanceStatesFor(view);
    if (states.empty()) return {};
    const auto instances = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>();
    const auto materials = buffers.MaterialsFor(view);
    enum class Casters { None,
                         SingleSided,
                         DoubleSided,
                         Complex };
    std::map<uint32_t, Casters> checked;
    bool single_sided = false, double_sided = false;
    for (const auto slot : buffers.GpuInstanceSlots.GetSpan<uint32_t>()) {
        const auto mesh = instances[slot].Mesh;
        if (mesh == InvalidOffset || (states[slot] & InstanceStateHidden)) continue;
        auto [entry, first] = checked.try_emplace(mesh, Casters::None);
        if (!first) {
            if (entry->second == Casters::Complex) usage.ComplexCasterSlots.push_back(slot);
            continue;
        }
        const auto *record = meshes.TryGet(mesh);
        if (!record || !record->TriangleCount) continue;
        const auto assignments = meshes.Arenas().PrimitiveMaterials.Get(record->PrimitiveMaterials);
        meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
            if (primitive.Topology != 0u || !primitive.TriangleCount) return;
            const uint32_t index = primitive.PrimitiveIndex < assignments.size() ? assignments[primitive.PrimitiveIndex] : 0u;
            const auto &material = materials[index];
            if (material.AlphaMode == MaterialAlphaMode::Blend) {
                if (!material.Unlit && material.Transmission.Factor > 0.f) usage.BlendedTransmission = true;
                else usage.Coverage = true;
            }
            usage.OpaqueSingleSided &= material.AlphaMode == MaterialAlphaMode::Opaque && !material.DoubleSided;
            usage.OpaqueDoubleSided &= material.AlphaMode == MaterialAlphaMode::Opaque && material.DoubleSided;
            const bool opaque = material.AlphaMode == MaterialAlphaMode::Opaque && (material.Unlit || material.Transmission.Factor == 0.f);
            usage.Opaque |= opaque;
            if (classify_casters) {
                const auto kind = !opaque ? Casters::Complex : material.DoubleSided ? Casters::DoubleSided :
                                                                                      Casters::SingleSided;
                entry->second = entry->second == Casters::None || entry->second == kind ? kind : Casters::Complex;
            }
        });
        if (entry->second == Casters::Complex) usage.ComplexCasterSlots.push_back(slot);
        single_sided |= entry->second == Casters::SingleSided;
        double_sided |= entry->second == Casters::DoubleSided;
    }
    if (single_sided == double_sided) usage.ComplexCasterSlots.clear();
    usage.CullOpaqueBackfaces = single_sided;
    return usage;
}
void RayTracingScene::Bind(mtl::Command &cmd, uint64_t hit_instances) const {
    cmd.Arguments()->setResource(Data->Top->Handle->gpuResourceID(), 5);
    cmd.FragmentArguments()->setResource(Data->Top->Handle->gpuResourceID(), 5);
    cmd.SetAddress(hit_instances, 6);
    cmd.SetBuffer(*Data->Triangles, 0, 7);
    cmd.SetBuffer(*Data->Environment, 0, 8);
    cmd.SetBuffer(*Data->Motion, 0, 10);
}
uint64_t RayTracingScene::Build(mtl::PassChain &chain, const MeshStore &meshes, const GpuBuffers &buffers, const SceneViewUBO &view, const state::Registry &registry, const MainPipeline &pipelines) {
    const profile::CpuScope scope{"BuildRayTracingScene"};
    // Frame submissions complete before the next frame mutates the shared scene arenas.
    auto &out = *Data;
    const bool positions_changed = std::exchange(out.PositionsDirty, false);
    const auto &ctx = out.Context;
    if (out.Epoch != registry.Epoch) {
        out.Bottom.clear();
        out.Top.reset();
        out.Epoch = registry.Epoch;
        out.CameraMedia() = {};
        out.PoseSnapshotValid = false;
    }
    uint32_t builds{}, refits{};
    const bool canonical_geometry_changed = std::exchange(out.CanonicalGeometryDirty, false);
    bool geometry_updated{}, history_invalid = canonical_geometry_changed, has_pose{};
    out.DeformationHistory = out.CapturePose = false;
    const auto environment = chain.BeginCompute("RayEnvironment", MTL::StageFragment | MTL::StageDispatch);
    pipelines.RayEnvironmentRows.Bind(environment);
    chain.Cmd.SetBuffer(buffers.Ctx.Slots.Table(), 0, 0);
    chain.Cmd.SetBytes(view.Ibl.SpecularEnvSamplerSlot, 1);
    chain.Cmd.SetBuffer(*out.Environment, 0, 8);
    environment->dispatchThreads({RayEnvironmentRows, 1, 1}, {64, 1, 1});
    mtl::ComputeBarrier(environment);
    pipelines.RayEnvironmentMarginal.Bind(environment);
    environment->dispatchThreads({1, 1, 1}, {1, 1, 1});
    const auto instances = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>();
    const auto *pose = buffers.RenderPoseFor(view);
    // Equal deformation inputs produce equal mesh-local positions. Instances retain their
    // own transforms, hit identities and position history even when their BLAS is shared.
    std::map<uint32_t, uint32_t> armature_counts;
    for (const auto [_, armature] : registry.view<const ArmaturePoseState>().each())
        for (const auto range : armature.GpuDeformRanges)
            if (range.Count) armature_counts.emplace(range.Offset, range.Count);
    const auto armature_values = (pose ? pose->ArmatureDeform : buffers.ArmatureDeformBuffer.Buffer).GetSpan<mat4>();
    const auto morph_values = (pose ? pose->MorphWeights : buffers.MorphWeightBuffer.Buffer).GetSpan<float>();
    std::map<std::pair<uint32_t, std::vector<std::byte>>, uint32_t> pose_equivalents;
    std::set<std::pair<uint32_t, uint32_t>> position_roots;
    std::vector<uint32_t> position_blocks;

    if (pose) out.PoseSnapshotValid = false;
    const auto transforms = (pose ? pose->Transforms : buffers.Instances.TransformBuffer).GetSpan<Transform>();
    const auto materials = buffers.MaterialsFor(view);
    const auto meshRecords = meshes.Arenas().Render.MeshRecords.Buffer.GetSpan<MeshRecord>();
    const auto states = buffers.InstanceStatesFor(view);
    const auto slots = states.empty() ? std::span<const uint32_t>{} : buffers.GpuInstanceSlots.GetSpan<uint32_t>();
    std::vector<MTL::IndirectAccelerationStructureInstanceDescriptor> native;
    std::vector<RayTracingInstance> mappings;
    std::vector<uint32_t> triangleHandles;
    std::set<std::pair<uint32_t, uint32_t>> used_geometry;
    std::map<uint32_t, uint32_t> triangle_offsets;
    uint64_t ray_triangles{};
    mtl::ComputeEncoder encoder{};
    const auto acceleration = [&] {
        if (!encoder.Pass) encoder = chain.BeginAccelerationStructure(MTL::StageDispatch | MTL::StageFragment);
        return encoder;
    };
    for (const auto slot : slots) {
        const auto &instance = instances[slot];
        if (instance.Mesh == InvalidOffset || (states[slot] & InstanceStateHidden)) continue;
        const auto *record = meshes.TryGet(instance.Mesh);
        if (!record || !record->TriangleCount) continue;
        const auto root = instance.PositionNamespace != InvalidOffset ? instance.PositionNamespace : meshRecords[instance.Mesh].Display.PositionNamespace;
        uint32_t geometry_root = root;
        if (root != InvalidOffset) {
            const auto &display = meshRecords[instance.Mesh].Display;
            const auto matrix_count = armature_counts.find(instance.ArmatureDeformOffset);
            const bool skinned = display.BoneDeformOffset != InvalidOffset;
            const bool morphed = display.MorphDeformOffset != InvalidOffset;
            if ((!skinned || matrix_count != armature_counts.end()) && (!morphed || instance.MorphWeightsOffset != InvalidOffset)) {
                std::vector<std::byte> inputs;
                const std::array counts{skinned ? matrix_count->second : 0u, morphed ? display.MorphTargetCount : 0u};
                inputs.append_range(BytesOf(counts));
                if (skinned) inputs.append_range(std::as_bytes(armature_values.subspan(instance.ArmatureDeformOffset, counts[0])));
                if (morphed) inputs.append_range(std::as_bytes(morph_values.subspan(instance.MorphWeightsOffset, counts[1])));
                if (skinned || morphed) geometry_root = pose_equivalents.try_emplace(std::pair{instance.Mesh, std::move(inputs)}, root).first->second;
            }
            if (position_roots.emplace(instance.Mesh, root).second) {
                const auto positions = buffers.PosedPositions.View(root);
                for (const auto block : meshes.GetBlockList(instance.Mesh, MeshStore::ElementDomain::Vertex).Blocks)
                    position_blocks.push_back(positions.Index(block * 256u) / 256u);
            }
        }
        const auto key = std::pair{instance.Mesh, geometry_root};
        auto &cached = out.Bottom[key];
        if (used_geometry.insert(key).second) {
            const auto pose = buffers.PosedPositions.View(geometry_root);
            const auto [triangle_entry, first_mesh] = triangle_offsets.try_emplace(instance.Mesh, uint32_t(triangleHandles.size()));
            const auto offset = triangle_entry->second;
            const auto &vertices = pose.empty() ? meshes.Arenas().Vertices.Buffer : buffers.PosedPositions.Values.Buffer;
            const std::pair<uint64_t, uint64_t> vertex_buffer{(*vertices)->gpuAddress(), vertices.UsedSize};
            const auto revision = std::tuple{record->MeshletRevision, meshes.GetDerived(instance.Mesh).NormalRevision, record->Vertices.Index, record->TriangleData.Index, record->TriangleCount, meshes.Arenas().Vertices.Set(record->Vertices).Revision, pose.empty() ? 0u : buffers.PosedPositions.MappingRevision()};
            const bool same_geometry = cached.Acceleration && cached.VertexBuffer == vertex_buffer && cached.Revision == revision;
            has_pose |= !pose.empty();
            history_invalid |= !same_geometry || (!pose.empty() && !out.PoseSnapshotValid);
            bool same_indices = true;
            if (!same_geometry) {
                const TriangleVertexView triangles{meshes.GetTriangleCorners(instance.Mesh), meshes.Arenas().FaceCorners.Buffer.GetSpan<uint32_t>()};
                std::vector<uint32_t> indices;
                meshes.Arenas().Triangles.ForEach(record->TriangleData, [&](ElementHandle h, ElementAddress) {
                    if (first_mesh) triangleHandles.push_back(h.Value);
                    for (const auto vertex : triangles.TriangleAtHandle(h.Value)) {
                        const auto address = meshes.Arenas().Vertices.Address(record->Vertices, ElementHandle{vertex}).Value();
                        indices.push_back(pose.empty() ? address : pose.Index(address));
                    }
                });
                const auto [first, last] = std::ranges::minmax_element(indices);
                const uint32_t base = *first, count = *last - base + 1u;
                const bool compact = count <= uint32_t(UINT16_MAX) + 1u;
                std::vector<uint16_t> compact_indices;
                if (compact) {
                    compact_indices.reserve(indices.size());
                    for (const auto index : indices) compact_indices.push_back(uint16_t(index - base));
                }
                const auto index_type = compact ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
                const auto index_bytes = compact ? std::as_bytes(std::span{compact_indices}) : std::as_bytes(std::span{indices});
                same_indices = cached.Indices && cached.IndexType == index_type && cached.VertexBase == (compact ? base : 0u) && std::ranges::equal(cached.Indices.Contents(), index_bytes);
                cached.IndexType = index_type;
                cached.VertexBase = compact ? base : 0u;
                cached.VertexCount = compact ? count : uint32_t(vertices.UsedSize / sizeof(vec3));
                if (!same_indices) {
                    cached.Indices = {ctx, index_bytes.size()};
                    std::memcpy(cached.Indices.Contents().data(), index_bytes.data(), index_bytes.size());
                }
                cached.Revision = revision;
            } else if (first_mesh) {
                const auto *previous = reinterpret_cast<const uint32_t *>(out.Triangles.Contents().data()) + cached.TriangleOffset;
                triangleHandles.insert(triangleHandles.end(), previous, previous + record->TriangleCount);
            }
            cached.TriangleOffset = offset;
            ray_triangles += record->TriangleCount;
            if (!same_geometry || canonical_geometry_changed || (!pose.empty() && positions_changed)) {
                const auto geometry = NS::TransferPtr(MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init());
                geometry->setVertexBuffer({(*vertices)->gpuAddress() + size_t(cached.VertexBase) * sizeof(vec3), size_t(cached.VertexCount) * sizeof(vec3)});
                geometry->setVertexStride(sizeof(vec3));
                geometry->setVertexFormat(MTL::AttributeFormatFloat3);
                geometry->setIndexBuffer({cached.Indices.Address(), cached.Indices.Length()});
                geometry->setIndexType(cached.IndexType);
                geometry->setTriangleCount(record->TriangleCount);
                geometry->setOpaque(true);
                const auto desc = NS::TransferPtr(MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
                const NS::Object *objects[]{geometry.get()};
                const auto array = NS::TransferPtr(NS::Array::alloc()->init(objects, 1));
                desc->setGeometryDescriptors(array.get());
                desc->setUsage(MTL::AccelerationStructureUsageRefit);
                const bool refit = same_indices && cached.Acceleration && cached.VertexBuffer == vertex_buffer;
                if (!refit) cached.Acceleration = std::make_unique<Structure>(ctx, desc.get());
                cached.Acceleration->Build(acceleration(), desc.get(), refit);
                geometry_updated = true;
                if (refit) ++refits;
                else ++builds;
                cached.VertexBuffer = vertex_buffer;
            }
        }
        const auto &transform = transforms[slot];
        const auto rotation = numeric::ToMat3(transform.R);
        const bool pending = view.IsTransforming && view.InteractionMode != InteractionMode::Edit && (states[slot] & 1u);
        const auto delta_rotation = numeric::ToMat3(view.PendingRotation);
        const auto delta = [&](vec3 p) { return delta_rotation * (view.PendingScale * p); };
        MTL::IndirectAccelerationStructureInstanceDescriptor descriptor{};
        for (uint32_t c = 0; c < 3; ++c) {
            vec3 axis = rotation[c] * transform.S[c];
            if (pending) axis = delta(axis);
            descriptor.transformationMatrix.columns[c] = {axis.x, axis.y, axis.z};
        }
        const vec3 translation = pending ? view.PendingPivot + delta(transform.P - view.PendingPivot) + view.PendingTranslation : transform.P;
        descriptor.transformationMatrix.columns[3] = {translation.x, translation.y, translation.z};
        uint32_t volume_material = InvalidOffset;
        for (const auto index : meshes.Arenas().PrimitiveMaterials.Get(record->PrimitiveMaterials)) {
            const auto &material = materials[index];
            if (!material.Unlit && material.Transmission.Factor > 0.f && material.Volume.ThicknessFactor > 0.f) {
                volume_material = index;
                break;
            }
        }
        descriptor.mask = volume_material != InvalidOffset ? 3u : 1u;
        descriptor.accelerationStructureID = cached.Acceleration->Handle->gpuResourceID();
        native.push_back(descriptor);
        mappings.push_back({slot, cached.TriangleOffset, volume_material});
    }
    const auto removed = std::erase_if(out.Bottom, [&](const auto &entry) { return !used_geometry.contains(entry.first); });
    history_invalid |= removed != 0u;
    std::ranges::sort(position_blocks);
    position_blocks.erase(std::unique(position_blocks.begin(), position_blocks.end()), position_blocks.end());
    if (out.PositionBlocks != position_blocks) {
        out.PositionBlocks = std::move(position_blocks);
        out.PositionLayoutDirty = true;
        history_invalid = true;
    }
    profile::RecordCounter("RayPositionNamespaces", position_roots.size());
    profile::RecordCounter("RayDistinctPoseInputs", pose_equivalents.size());
    if (native.empty()) {
        out.Top.reset();
        out.PreviousPositions = {};
        out.PoseSnapshotValid = false;
        return {};
    }
    const auto upload = [&](mtl::ResidentBuffer &buffer, std::span<const std::byte> bytes) {
        if (!buffer || buffer.Length() != bytes.size()) buffer = {ctx, bytes.size()};
        else if (std::ranges::equal(buffer.Contents(), bytes)) return false;
        std::memcpy(buffer.Contents().data(), bytes.data(), bytes.size());
        return true;
    };
    const auto native_bytes = BytesOf(native);
    const auto hit_bytes = BytesOf(mappings);
    const bool mappings_changed = upload(out.HitInstances, hit_bytes);
    const bool resize_top = InstanceCount() != native.size();
    std::vector<RayMotionTransform> motion(instances.size());
    bool identity_changed = false;
    const auto old_motion_count = out.Motion ? out.Motion.Length() / sizeof(RayMotionTransform) : 0u;
    const auto *old_motion = out.Motion ? reinterpret_cast<const RayMotionTransform *>(out.Motion.Contents().data()) : nullptr;
    for (const auto &mapping : mappings) {
        const auto slot = mapping.InstanceSlot;
        motion[slot].Identity = state::Integral(registry.EntityAt(instances[slot].ObjectId - 1u));
        identity_changed |= slot >= old_motion_count || old_motion[slot].Identity != motion[slot].Identity;
    }
    const auto matrix = [](const auto &descriptor) {
        mat4 result{1};
        for (uint32_t c = 0; c < 4; ++c) {
            const auto p = descriptor.transformationMatrix.columns[c];
            result[c] = vec4{p.x, p.y, p.z, c == 3 ? 1.f : 0.f};
        }
        return result;
    };
    // Shutter samples do not reproject between times; their motion records contain identities only.
    if (!pose && !history_invalid && !mappings_changed && !identity_changed && out.Instances) {
        const auto *previous = reinterpret_cast<const MTL::IndirectAccelerationStructureInstanceDescriptor *>(out.Instances.Contents().data());
        for (size_t i = 0; i < native.size(); ++i) {
            const auto old_matrix = matrix(previous[i]), current_matrix = matrix(native[i]);
            const auto delta = old_matrix == current_matrix ? mat4{1} : old_matrix * numeric::Inverse(current_matrix);
            const mat3 linear{vec3{delta[0]}, vec3{delta[1]}, vec3{delta[2]}};
            const auto slot = mappings[i].InstanceSlot;
            const auto &instance = instances[slot];
            const bool deformed = instance.PositionNamespace != InvalidOffset || meshRecords[instance.Mesh].Display.PositionNamespace != InvalidOffset;
            motion[slot] = {delta, numeric::Transpose(numeric::Inverse(linear)), deformed ? RayMotionDeformed : RayMotionRigid, motion[slot].Identity};
            out.DeformationHistory |= deformed;
        }
    }
    out.CapturePose = has_pose && !pose && (!out.PoseSnapshotValid || positions_changed || history_invalid);
    if (!has_pose) {
        out.PreviousPositions = {};
        out.PositionBlocks.clear();
        out.PoseSnapshotValid = false;
    }
    upload(out.Motion, BytesOf(motion));
    const bool instances_changed = upload(out.Instances, native_bytes);
    upload(out.Triangles, BytesOf(triangleHandles));
    const uint64_t hit_instances = pose ? chain.Cmd.Upload(hit_bytes) : out.HitInstances.Address();
    profile::RecordCounter("RayBlasBuilds", builds);
    profile::RecordCounter("RayBlasRefits", refits);
    profile::RecordCounter("RayTriangles", ray_triangles);
    profile::RecordCounter("RayTriangleMapBytes", triangleHandles.size() * sizeof(uint32_t));
    uint64_t index_bytes{}, compact_geometries{};
    for (const auto &[key, geometry] : out.Bottom) {
        index_bytes += geometry.Indices.Length();
        compact_geometries += geometry.IndexType == MTL::IndexTypeUInt16;
    }
    profile::RecordCounter("RayIndexBytes", index_bytes);
    profile::RecordCounter("RayCompactGeometries", compact_geometries);
    const bool update_top = !out.Top || instances_changed || geometry_updated;
    if (update_top || mappings_changed) ++out.Revision;
    if (history_invalid || mappings_changed || identity_changed) ++out.GeometryRevision;
    profile::RecordCounter("RayTlasBuilds", update_top);
    if (!update_top) return hit_instances;
    acceleration();
    const auto desc = NS::TransferPtr(MTL4::InstanceAccelerationStructureDescriptor::alloc()->init());
    desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    desc->setInstanceDescriptorStride(sizeof(native[0]));
    // CPU encoding continues through every shutter sample before this command executes.
    const uint64_t instance_address = pose ? chain.Cmd.Upload(native_bytes) : out.Instances.Address();
    desc->setInstanceDescriptorBuffer({instance_address, native_bytes.size()});
    desc->setInstanceCount(native.size());
    if (resize_top) out.Top = std::make_unique<Structure>(ctx, desc.get());
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure, MTL4::VisibilityOptionDevice);
    out.Top->Build(encoder, desc.get());
    return hit_instances;
}
