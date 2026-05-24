#pragma once

#include "ShaderPipelineType.h"
#include "gpu/DrawData.h"

#include <vulkan/vulkan.hpp>

struct DrawBatchInfo {
    uint32_t DrawDataSlotOffset{0}, DrawCount{0};
    vk::DeviceSize IndirectOffset{0};
};

struct DrawListBuilder {
    std::vector<DrawData> Draws;
    std::vector<vk::DrawIndexedIndirectCommand> IndirectCommands;
    uint32_t MaxIndexCount{0};

    DrawBatchInfo BeginBatch() { return {uint32_t(Draws.size()), 0, IndirectCommands.size() * sizeof(vk::DrawIndexedIndirectCommand)}; }

    void Append(DrawBatchInfo &batch, const DrawData &draw, uint32_t index_count, uint32_t instance_count) {
        if (index_count == 0 || instance_count == 0) return;
        const uint32_t draw_data_start = Draws.size();
        for (uint32_t i = 0; i < instance_count; ++i) {
            DrawData per_instance = draw;
            per_instance.FirstInstance = draw.FirstInstance + i;
            Draws.emplace_back(per_instance);
        }
        const uint32_t first_instance = draw_data_start - batch.DrawDataSlotOffset;
        IndirectCommands.emplace_back(vk::DrawIndexedIndirectCommand{index_count, instance_count, 0, 0, first_instance});
        MaxIndexCount = std::max(MaxIndexCount, index_count);
        ++batch.DrawCount;
    }
};

struct SelectionDrawInfo {
    ShaderPipelineType Pipeline;
    DrawBatchInfo Batch;
};

// Per-scene draw-list scratch storage. Re-populated each frame by RecordRenderCommandBuffer;
// vector capacity is amortized across frames. Component on the viewport singleton entity.
struct DrawState {
    DrawListBuilder List;
    uint32_t MainDrawCount{0}; // Draws.size() after main batches, before silhouette
    uint32_t MainIndirectCount{0}; // IndirectCommands.size() after main batches
    DrawBatchInfo Silhouette;
    DrawBatchInfo FillOpaque, FillBlend;
    DrawBatchInfo EdgeQuad, WireLine, Point;
    DrawBatchInfo ExtrasLine;
    DrawBatchInfo BoneFill, BoneWire, BoneSphereFill, BoneSphereWire;
    DrawBatchInfo OverlayFaceNormals, OverlayVertexNormals;

    // Cached selection pass draw list — reused when only the camera changed.
    DrawListBuilder SelectionList;
    std::vector<SelectionDrawInfo> SelectionDraws;
};
