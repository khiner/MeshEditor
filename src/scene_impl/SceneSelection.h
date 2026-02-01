#pragma once

#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ClickSelectElementPushConstants.h"
#include "gpu/ClickSelectPushConstants.h"

namespace {

std::unordered_set<uint32_t> ConvertSelectionElement(const MeshSelection &selection, const Mesh &mesh, Element from_element, Element new_element) {
    if (from_element == Element::None || selection.Handles.empty()) return {};
    if (from_element == new_element) return selection.Handles;

    const auto &handles = selection.Handles;
    std::unordered_set<uint32_t> new_handles;
    if (from_element == Element::Face) {
        if (new_element == Element::Edge) {
            for (auto f : handles) {
                for (const auto heh : mesh.fh_range(FH{f})) new_handles.emplace(*mesh.GetEdge(heh));
            }
        } else if (new_element == Element::Vertex) {
            for (auto f : handles) {
                for (const auto vh : mesh.fv_range(FH{f})) new_handles.emplace(*vh);
            }
        }
    } else if (from_element == Element::Edge) {
        if (new_element == Element::Vertex) {
            for (auto eh_raw : handles) {
                const auto heh = mesh.GetHalfedge(EH{eh_raw}, 0);
                new_handles.emplace(*mesh.GetFromVertex(heh));
                new_handles.emplace(*mesh.GetToVertex(heh));
            }
        } else if (new_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                const bool all_selected = all_of(mesh.fh_range(fh), [&](auto heh) {
                    return handles.contains(*mesh.GetEdge(heh));
                });
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    } else if (from_element == Element::Vertex) {
        if (new_element == Element::Edge) {
            for (const auto eh : mesh.edges()) {
                const auto heh = mesh.GetHalfedge(eh, 0);
                if (handles.contains(*mesh.GetFromVertex(heh)) && handles.contains(*mesh.GetToVertex(heh))) {
                    new_handles.emplace(*eh);
                }
            }
        } else if (new_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                const bool all_selected = all_of(mesh.fv_range(fh), [&](auto vh) { return handles.contains(*vh); });
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    }
    return new_handles;
}

// Returns primary edit instance per selected mesh: active instance if selected, else first selected instance.
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &r) {
    std::unordered_map<entt::entity, entt::entity> primaries;
    const auto active = FindActiveEntity(r);
    for (const auto [e, mi] : r.view<const MeshInstance, const Selected>().each()) {
        auto &primary = primaries[mi.MeshEntity];
        if (primary == entt::entity{} || e == active) primary = e;
    }
    return primaries;
}
bool HasSelectedInstance(const entt::registry &r, entt::entity mesh_entity) {
    return any_of(r.view<const MeshInstance, const Selected>().each(), [mesh_entity](const auto &t) { return std::get<1>(t).MeshEntity == mesh_entity; });
}
auto GetSelectedMeshEntities(const entt::registry &r) {
    return r.view<const MeshInstance, const Selected>().each() | transform([](const auto &t) { return std::get<1>(t).MeshEntity; }) | to<std::unordered_set>();
}

void RunSelectionCompute(
    vk::CommandBuffer cb, vk::Queue queue, vk::Fence fence, vk::Device device,
    const auto &compute, const auto &pc, auto &&dispatch, vk::Semaphore wait_semaphore = {}
) {
    const Timer timer{"RunSelectionCompute"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    dispatch(cb);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    static constexpr vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eComputeShader};
    if (wait_semaphore) {
        submit.setWaitSemaphores(wait_semaphore);
        submit.setWaitDstStageMask(wait_stage);
    }
    queue.submit(submit, fence);
    WaitFor(fence, device);
}

uint32_t GetElementCount(const Mesh &mesh, Element element) {
    if (element == Element::Vertex) return mesh.VertexCount();
    if (element == Element::Edge) return mesh.EdgeCount();
    if (element == Element::Face) return mesh.FaceCount();
    return 0;
}

// After rendering elements to selection buffer, dispatch compute shader to find the nearest element to mouse_px.
// Returns 0-based element index, or nullopt if no element found.
std::optional<uint32_t> FindNearestSelectionElement(
    const SceneBuffers &buffers, const ComputePipeline &compute, vk::CommandBuffer cb,
    vk::Queue queue, vk::Fence fence, vk::Device device,
    uint32_t head_image_index, uint32_t selection_nodes_slot, uint32_t click_result_slot,
    glm::uvec2 mouse_px, uint32_t max_element_id, Element element,
    vk::Semaphore wait_semaphore
) {
    const uint32_t radius = element == Element::Face ? 0u : ClickSelectRadiusPx;
    const uint32_t group_count = element == Element::Face ? 1u : SceneBuffers::ClickSelectElementGroupCount;
    RunSelectionCompute(
        cb, queue, fence, device, compute,
        ClickSelectElementPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius,
            .HeadImageIndex = head_image_index,
            .SelectionNodesIndex = selection_nodes_slot,
            .ClickResultIndex = click_result_slot,
        },
        [group_count](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_count, 1, 1); },
        wait_semaphore
    );

    const auto *candidates = reinterpret_cast<const ClickElementCandidate *>(buffers.ClickElementResultBuffer.GetData().data());
    ClickElementCandidate best{.ObjectId = 0, .Depth = 1.0f, .DistanceSq = std::numeric_limits<uint32_t>::max()};
    for (uint32_t i = 0; i < group_count; ++i) {
        const auto &candidate = candidates[i];
        if (candidate.ObjectId == 0) continue;
        if (candidate.DistanceSq < best.DistanceSq || (candidate.DistanceSq == best.DistanceSq && candidate.Depth < best.Depth)) {
            best = candidate;
        }
    }

    if (best.ObjectId == 0 || best.ObjectId > max_element_id) return {};
    return best.ObjectId - 1;
}

} // namespace
