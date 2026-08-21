#pragma once

#include "gpu/CullEntry.h"
#include "gpu/CullFlag.h"
#include "gpu/DrawData.h"
#include "metal/MetalCpp.h"
#include "render/ShaderPipelineType.h"

#include <entt/entity/entity.hpp>

#include <optional>
#include <unordered_map>

// Metal draws one indirect command per call, so a batch walks its commands at this stride.
inline constexpr uint64_t IndirectCommandStride{sizeof(MTL::DrawIndexedPrimitivesIndirectArguments)};

struct DrawBatchInfo {
    uint32_t DrawDataSlotOffset{0}, DrawCount{0};
    uint64_t IndirectOffset{0};
    bool Cull{false};
};

struct DrawListBuilder {
    std::vector<DrawData> Draws;
    std::vector<CullEntry> CullEntries; // Parallel to Draws, for the cull pass.
    std::vector<MTL::DrawIndexedPrimitivesIndirectArguments> IndirectCommands;
    uint32_t MaxIndexCount{0};

    // `cull` marks the batch frustum-cullable. Only order-independent (opaque) batches qualify:
    // culling repacks a command's instances in arrival order.
    DrawBatchInfo BeginBatch(bool cull = false) const { return {uint32_t(Draws.size()), 0, IndirectCommands.size() * IndirectCommandStride, cull}; }

    void Append(DrawBatchInfo &batch, const DrawData &draw, uint32_t index_count, uint32_t instance_count) {
        if (index_count == 0 || instance_count == 0) return;
        const uint32_t draw_data_start = Draws.size();
        const uint32_t cmd_index = uint32_t(IndirectCommands.size()) | (batch.Cull ? 0u : uint32_t(CullFlag::KeepOrder));
        for (uint32_t i = 0; i < instance_count; ++i) {
            DrawData per_instance = draw;
            per_instance.FirstInstance = draw.FirstInstance + i;
            Draws.emplace_back(per_instance);
            CullEntries.emplace_back(cmd_index, draw_data_start);
        }
        const uint32_t first_instance = draw_data_start - batch.DrawDataSlotOffset;
        IndirectCommands.push_back({index_count, instance_count, 0, 0, first_instance});
        MaxIndexCount = std::max(MaxIndexCount, index_count);
        ++batch.DrawCount;
    }
};

struct SelectionDrawInfo {
    ShaderPipelineType Pipeline;
    MTL::PrimitiveType Primitive;
    DrawBatchInfo Batch;
};

// Posed-buffer layout for one mesh entity's instance run: base offsets plus per-instance strides.
struct PosedRanges {
    // One instance's derived normals, absent for a mesh with no faces to derive them from.
    struct NormalRanges {
        uint32_t VertexOffset{InvalidOffset}, SeamOffset{InvalidOffset}, FaceOffset{InvalidOffset};
        uint32_t SeamCount{0}, FaceCount{0};
    };

    uint32_t FirstInstance{0};
    bool PerInstance{false};
    uint32_t PositionBase{InvalidOffset};
    uint32_t VertexCount{0};
    uint32_t MeshletBoundsBase{InvalidOffset};
    uint32_t MeshletCount{0};
    std::optional<NormalRanges> Normals{}; // Instance 0's offsets.

    uint32_t PositionOffset(uint32_t i) const { return PositionBase + i * VertexCount; }
    uint32_t MeshletBoundsOffset(uint32_t i) const { return MeshletBoundsBase + i * MeshletCount; }
    std::optional<NormalRanges> NormalsAt(uint32_t i) const {
        if (!Normals) return std::nullopt;
        return NormalRanges{
            Normals->VertexOffset + i * VertexCount,
            Normals->SeamOffset + i * Normals->SeamCount,
            Normals->FaceOffset + i * Normals->FaceCount,
            Normals->SeamCount,
            Normals->FaceCount,
        };
    }
};

// Persistent draw data, rebuilt when scene structure or batch routing changes.
struct DrawState {
    DrawListBuilder List;
    std::unordered_map<entt::entity, PosedRanges> PosedByEntity;
    DrawBatchInfo FillLine, FillPoint; // Point and line meshes, shaded in the scene pass.
    DrawBatchInfo EdgeQuad, WireLine, Point, ExtrasLine;
    DrawBatchInfo BoneFill, BoneWire, BoneSphereFill, BoneSphereWire;
    DrawBatchInfo OverlayFaceNormals, OverlayVertexNormals;

    // Cached selection pass draw list — reused when only the camera changed.
    DrawListBuilder SelectionList;
    std::vector<SelectionDrawInfo> SelectionDraws;
    bool SelectionStale{true}; // Selection fragment data no longer matches the scene. Cleared after RenderSelectionPass.
};
