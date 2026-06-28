#pragma once

#include "vulkan/Buffer.h"
#include "vulkan/VulkanResources.h"

#include <filesystem>
#include <optional>

// Captures a sub-rect of the viewport's final color image each frame and pipes raw BGRA bytes
// to an `ffmpeg` subprocess for on-the-fly H.264 encoding. `ffmpeg` must be on PATH; if it isn't,
// IsActive() is false and CaptureFrame is a no-op. The capture region is locked at construction.
struct VideoRecorder {
    VideoRecorder(
        const VulkanResources &, mvk::BufferContext &,
        const std::filesystem::path &output_path, vk::Offset3D offset, vk::Extent2D extent, int fps
    );
    ~VideoRecorder();

    VideoRecorder(const VideoRecorder &) = delete;
    VideoRecorder &operator=(const VideoRecorder &) = delete;

    // Image must be in eShaderReadOnlyOptimal; left in the same layout on return.
    // Copies the sub-rect locked at construction.
    void CaptureFrame(vk::Image);

    bool IsActive() const { return bool(Pipe); }
    uint64_t CapturedFrameCount() const { return FrameCount; }

private:
    struct PipeCloser {
        void operator()(std::FILE *) const noexcept;
    };

    void Stop();

    vk::Device Device;
    vk::Queue Queue;
    vk::Offset3D Offset;
    vk::Extent2D Ex;
    vk::DeviceSize FrameBytes;

    vk::UniqueCommandPool CommandPool;
    vk::UniqueCommandBuffer CommandBuffer;
    vk::UniqueFence Fence;

    std::optional<mvk::Buffer> Staging;

    std::unique_ptr<std::FILE, PipeCloser> Pipe;
    uint64_t FrameCount{0};
};
