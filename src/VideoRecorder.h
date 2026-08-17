#pragma once

#include "vulkan/Buffer.h"
#include "vulkan/VulkanResources.h"

#include <filesystem>
#include <optional>
#include <span>

struct WavWriter;

// Captures a sub-rect of the viewport's final color image each frame and pipes raw BGRA bytes
// to an `ffmpeg` subprocess for on-the-fly H.264 encoding. `ffmpeg` must be on PATH; if it isn't,
// IsActive() is false and CaptureFrame is a no-op. The capture region is locked at construction.
// A nonzero `audio_sample_rate` also writes the frames handed to CaptureAudio and muxes them in when the recording stops, which costs a second pass over the video.
// At zero the video path is untouched.
// A `.wav` output path records audio only: no GPU capture, no ffmpeg, the samples stream straight into a 32-bit float wav, and CaptureFrame only counts the frame for the recording's pacing.
struct VideoRecorder {
    VideoRecorder(
        const VulkanResources &, mvk::BufferContext &,
        const std::filesystem::path &output_path, vk::Offset3D offset, vk::Extent2D extent, int fps,
        uint32_t audio_sample_rate = 0
    );
    ~VideoRecorder();

    VideoRecorder(const VideoRecorder &) = delete;
    VideoRecorder &operator=(const VideoRecorder &) = delete;

    // Image must be in eShaderReadOnlyOptimal; left in the same layout on return.
    // Copies the sub-rect locked at construction.
    void CaptureFrame(vk::Image);

    // Mono frames at the rate given at construction. A no-op when recording without audio.
    void CaptureAudio(std::span<const float>);

    // Whichever sink the output path selected: the wav for audio only, the ffmpeg pipe otherwise.
    bool IsActive() const { return Wav || Pipe; }
    uint64_t CapturedFrameCount() const { return FrameCount; }

    // Whether any recording this process made failed, which the process reports as its exit status.
    static bool AnyFailed();

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

    // The file this recording writes.
    // VideoPath and AudioPath name the video-only file muxed into it and the wav muxed with it, and are set only when a video recording also carries audio.
    std::filesystem::path FinalPath, VideoPath, AudioPath;
    // The audio sink: the recording's own file where it records audio only, and the sidecar the mux reads where it carries audio alongside video.
    std::unique_ptr<WavWriter> Wav;
};
