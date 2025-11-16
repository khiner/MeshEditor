// GPU Box Selection Implementation
#include "Scene.h"
#include "ComputePipeline.h"
#include "Entity.h"
#include "mesh/MeshRender.h"
#include "Widgets.h"
#include "Vulkan/UniqueBuffers.h"

#include <entt/entity/registry.hpp>
#include <print>

using namespace he;

constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }

void Scene::BoxSelectGPU(const glm::vec2 &box_min, const glm::vec2 &box_max) {
    Element element = EditingHandle.Element;
    std::println("BoxSelectGPU: element type = {}", static_cast<int>(element));

    if (InteractionMode == ::InteractionMode::Edit) {
        // Edit mode: Only process active entity
        auto active = GetActiveMeshEntity();
        if (active == entt::null) return;

        const auto mesh_entity = R.get<MeshInstance>(active).MeshEntity;
        DispatchBoxSelect(mesh_entity, element, box_min, box_max);

    } else if (InteractionMode == ::InteractionMode::Object) {
        // Object mode: Process all visible meshes
        auto view = R.view<MeshInstance, Visible>();
        std::unordered_set<entt::entity> unique_meshes;

        for (auto entity : view) {
            unique_meshes.insert(R.get<MeshInstance>(entity).MeshEntity);
        }

        for (auto mesh_entity : unique_meshes) {
            DispatchBoxSelect(mesh_entity, element, box_min, box_max);
        }
    }
}

void Scene::DispatchBoxSelect(
    entt::entity mesh_entity,
    Element element,
    const glm::vec2 &box_min,
    const glm::vec2 &box_max
) {
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    const auto &models_buffer = R.get<ModelsBuffer>(mesh_entity);

    // Get element count
    uint32_t element_count = 0;
    switch (element) {
        case Element::Vertex: element_count = mesh.VertexCount(); break;
        case Element::Edge:   element_count = mesh.EdgeCount(); break;
        case Element::Face:   element_count = mesh.FaceCount(); break;
        case Element::None:   return; // No selection for None type
    }
    if (element_count == 0) return;

    // Get instance count - count how many entities reference this mesh
    uint32_t instance_count = 0;
    std::vector<uint32_t> instance_to_entity;
    for (auto [entity, instance] : R.view<MeshInstance>().each()) {
        if (instance.MeshEntity == mesh_entity) {
            instance_to_entity.push_back(static_cast<uint32_t>(entity));
            instance_count++;
        }
    }
    if (instance_count == 0) return;

    SelectionBuffers.InstanceToEntity->Update(as_bytes(instance_to_entity));

    // Update parameters
    const auto size = ToGlm(ImGui::GetContentRegionAvail());
    auto proj = Camera.Projection(size.x / size.y);
    proj[1][1] *= -1;  // Flip Y for Vulkan (GLM's perspective is OpenGL-style)
    BoxSelectionParams params{
        .ViewProj = proj * Camera.View(),
        .BoxMin = box_min,
        .BoxMax = box_max,
        .ElementType = static_cast<uint32_t>(element),
        .ElementCount = element_count,
        .InstanceCount = instance_count,
        .DepthTolerance = 0.0001f,
        .EnableDepthTest = 0  // Disabled for now, will enable with depth buffer later
    };
    std::println("Box select: min=({}, {}), max=({}, {}), elements={}, instances={}",
                 box_min.x, box_min.y, box_max.x, box_max.y, element_count, instance_count);
    SelectionBuffers.Params->Update(as_bytes(params));

    // Clear results
    uint32_t zero = 0;
    SelectionBuffers.Results->Update(as_bytes(zero), 0);

    // Check if element type exists in mesh buffers
    if (!mesh_buffers.Mesh.contains(element)) return;

    // Update descriptor sets
    const auto &elem_buffers = mesh_buffers.Mesh.at(element);
    BoxSelectPipeline->UpdateDescriptorSet(0, *SelectionBuffers.Params->DeviceBuffer, vk::DescriptorType::eUniformBuffer);
    BoxSelectPipeline->UpdateDescriptorSet(1, *elem_buffers.Vertices.DeviceBuffer, vk::DescriptorType::eStorageBuffer);
    BoxSelectPipeline->UpdateDescriptorSet(2, *elem_buffers.Indices.DeviceBuffer, vk::DescriptorType::eStorageBuffer);
    BoxSelectPipeline->UpdateDescriptorSet(3, *models_buffer.Buffer.DeviceBuffer, vk::DescriptorType::eStorageBuffer);
    BoxSelectPipeline->UpdateDescriptorSet(4, *SelectionBuffers.Results->DeviceBuffer, vk::DescriptorType::eStorageBuffer);

    // Create command buffer for compute
    auto cmd_buffers = Vk.Device.allocateCommandBuffersUnique({
        *CommandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    });
    auto &cmd = cmd_buffers[0];

    cmd->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    BoxSelectPipeline->Bind(*cmd);

    // Dispatch
    uint32_t workgroups = (element_count + 255) / 256;
    BoxSelectPipeline->Dispatch(*cmd, workgroups);

    // Memory barrier
    vk::MemoryBarrier barrier{
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eTransferRead
    };
    cmd->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        barrier,
        {},
        {}
    );

    // Copy results from GPU to CPU-readable staging buffer
    vk::BufferCopy copy_region{0, 0, SelectionBuffers.Results->UsedSize};
    cmd->copyBuffer(
        *SelectionBuffers.Results->DeviceBuffer,
        SelectionBuffers.ResultsReadback->Get(),
        copy_region
    );

    cmd->end();

    // Submit and wait
    vk::SubmitInfo submit_info{
        {},
        {},
        *cmd
    };
    Vk.Queue.submit(submit_info, *TransferFence);
    WaitFor(*TransferFence);
    Vk.Device.resetFences(*TransferFence);

    // Read and apply results
    auto results = ReadSelectionResults();
    std::println("Box selection found {} results", results.size());
    for (const auto &[entity, element_idx] : results) {
        std::println("  Entity: {}, Element: {}", static_cast<uint32_t>(entity), element_idx);
        if (InteractionMode == ::InteractionMode::Object) {
            R.emplace_or_replace<Selected>(entity);
        } else {
            // Edit mode: Add element handle to selection
            // TODO: Implement element handle selection
        }
    }

    InvalidateCommandBuffer();
}

std::vector<std::pair<entt::entity, uint32_t>> Scene::ReadSelectionResults() {
    std::vector<std::pair<entt::entity, uint32_t>> results;

    // Read results from CPU-readable staging buffer
    auto data_bytes = SelectionBuffers.ResultsReadback->GetData();
    auto *data = reinterpret_cast<const uint32_t*>(data_bytes.data());
    uint32_t count = data[0];

    // Read instance-to-entity mapping
    auto instance_bytes = SelectionBuffers.InstanceToEntity->HostBuffer.GetData();
    auto *instance_map = reinterpret_cast<const uint32_t*>(instance_bytes.data());

    for (uint32_t i = 0; i < count; i++) {
        uint32_t packed = data[1 + i];
        uint32_t instance_idx = packed >> 24;
        uint32_t element_idx = packed & 0x00FFFFFF;

        // Look up entity from instance index
        entt::entity entity = static_cast<entt::entity>(instance_map[instance_idx]);

        results.emplace_back(entity, element_idx);
    }

    return results;
}
