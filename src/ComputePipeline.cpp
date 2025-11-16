#include "ComputePipeline.h"

ComputePipeline::ComputePipeline(vk::Device device, vk::DescriptorPool descriptor_pool, ::Shaders &&shaders)
    : Device(device), Shaders(std::move(shaders)) {

    // Compile shaders
    auto shader_stages = Shaders.CompileAll(device);

    // Create descriptor set layout
    auto layout_bindings = Shaders.GetLayoutBindings();
    vk::DescriptorSetLayoutCreateInfo layout_info{
        {},
        static_cast<uint32_t>(layout_bindings.size()),
        layout_bindings.data()
    };
    DescriptorSetLayout = device.createDescriptorSetLayoutUnique(layout_info);

    // Create pipeline layout
    vk::PipelineLayoutCreateInfo pipeline_layout_info{
        {},
        1,
        &DescriptorSetLayout.get()
    };
    PipelineLayout = device.createPipelineLayoutUnique(pipeline_layout_info);

    // Create compute pipeline
    vk::ComputePipelineCreateInfo pipeline_info{
        {},
        shader_stages[0],  // Should only have one compute shader stage
        PipelineLayout.get()
    };

    auto result = device.createComputePipelineUnique({}, pipeline_info);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    Pipeline = std::move(result.value);

    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo alloc_info{
        descriptor_pool,
        1,
        &DescriptorSetLayout.get()
    };
    auto descriptor_sets = device.allocateDescriptorSetsUnique(alloc_info);
    DescriptorSet = std::move(descriptor_sets[0]);
}

void ComputePipeline::Bind(vk::CommandBuffer cmd) const {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, Pipeline.get());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, PipelineLayout.get(), 0, DescriptorSet.get(), {});
}

void ComputePipeline::Dispatch(vk::CommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) const {
    cmd.dispatch(x, y, z);
}

void ComputePipeline::UpdateDescriptorSet(uint32_t binding, vk::Buffer buffer, vk::DescriptorType type) const {
    vk::DescriptorBufferInfo buffer_info{
        buffer,
        0,
        VK_WHOLE_SIZE
    };

    vk::WriteDescriptorSet write{
        DescriptorSet.get(),
        binding,
        0,
        1,
        type,
        nullptr,
        &buffer_info,
        nullptr
    };

    Device.updateDescriptorSets(write, {});
}

void ComputePipeline::UpdateDescriptorSet(uint32_t binding, vk::ImageView image_view, vk::Sampler sampler) const {
    vk::DescriptorImageInfo image_info{
        sampler,
        image_view,
        vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::WriteDescriptorSet write{
        DescriptorSet.get(),
        binding,
        0,
        1,
        vk::DescriptorType::eCombinedImageSampler,
        &image_info,
        nullptr,
        nullptr
    };

    Device.updateDescriptorSets(write, {});
}
