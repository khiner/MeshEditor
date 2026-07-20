#pragma once

#include "gpu/CullEntry.h"
#include "gpu/CullFlag.h"
#include "gpu/DrawData.h"
#include "render/ShaderPipelineType.h"

#include <entt/entity/entity.hpp>
#include <vulkan/vulkan.hpp>

#include <unordered_map>

struct DrawBatchInfo {
    uint32_t DrawDataSlotOffset{0}, DrawCount{0};
    vk::DeviceSize IndirectOffset{0};
    bool Cull{false};
};

struct DrawListBuilder {
    std::vector<DrawData> Draws;
    std::vector<CullEntry> CullEntries; // Parallel to Draws, for the cull pass.
    std::vector<vk::DrawIndexedIndirectCommand> IndirectCommands;
    uint32_t MaxIndexCount{0};

    // `cull` marks the batch frustum-cullable. Only order-independent (opaque) batches qualify:
    // culling repacks a command's instances in arrival order.
    DrawBatchInfo BeginBatch(bool cull = false) const { return {uint32_t(Draws.size()), 0, IndirectCommands.size() * sizeof(vk::DrawIndexedIndirectCommand), cull}; }

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
        IndirectCommands.emplace_back(index_count, instance_count, 0, 0, first_instance);
        MaxIndexCount = std::max(MaxIndexCount, index_count);
        ++batch.DrawCount;
    }
};

struct SelectionDrawInfo {
    ShaderPipelineType Pipeline;
    DrawBatchInfo Batch;
};

// Posed-buffer layout for one mesh entity's instance run: base offsets plus per-instance strides.
struct PosedRanges {
    uint32_t FirstInstance{0};
    bool PerInstance{false};
    uint32_t PositionBase{InvalidOffset};
    uint32_t VertexNormalBase{InvalidOffset}, SeamNormalBase{InvalidOffset}, FaceNormalBase{InvalidOffset};
    uint32_t VertexCount{0}, SeamCount{0}, FaceCount{0};

    uint32_t PositionOffset(uint32_t i) const { return PositionBase + i * VertexCount; }
    uint32_t VertexNormalOffset(uint32_t i) const { return VertexNormalBase + i * VertexCount; }
    uint32_t SeamNormalOffset(uint32_t i) const { return SeamNormalBase + i * SeamCount; }
    uint32_t FaceNormalOffset(uint32_t i) const { return FaceNormalBase + i * FaceCount; }
};

// Per-frame draw-list scratch storage
struct DrawState {
    DrawListBuilder List;
    uint32_t MainDrawCount{0}; // Draws.size() after main batches, before silhouette
    uint32_t MainIndirectCount{0}; // IndirectCommands.size() after main batches
    std::unordered_map<entt::entity, PosedRanges> PosedByEntity;
    DrawBatchInfo Silhouette;
    DrawBatchInfo FillOpaque, FillBlend;
    // Opaque split by material transmission, for real-transmission frames: the prepass draws every
    // material with non-transmissive texels, and the scene pass composites the result and re-draws
    // only the transmissive batch. Textured-transmission materials appear in both.
    DrawBatchInfo FillOpaquePrepass, FillOpaqueTransmissive;
    DrawBatchInfo EdgeQuad, WireLine, Point, ExtrasLine;
    DrawBatchInfo BoneFill, BoneWire, BoneSphereFill, BoneSphereWire;
    DrawBatchInfo OverlayFaceNormals, OverlayVertexNormals;

    // Cached selection pass draw list — reused when only the camera changed.
    DrawListBuilder SelectionList;
    std::vector<SelectionDrawInfo> SelectionDraws;
    bool SelectionStale{true}; // Selection fragment data no longer matches the scene. Cleared after RenderSelectionPass.
};
