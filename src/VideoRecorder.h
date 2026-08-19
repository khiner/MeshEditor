#pragma once

#include "metal/Image.h"

#include <filesystem>
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
        const mtl::Context &,
        const std::filesystem::path &output_path, uint32_t x, uint32_t y, mtl::Extent2D extent, int fps,
        uint32_t audio_sample_rate = 0
    );
    ~VideoRecorder();

    VideoRecorder(const VideoRecorder &) = delete;
    VideoRecorder &operator=(const VideoRecorder &) = delete;

    // Copy the sub-rect fixed at construction from the render target.
    void CaptureFrame(const mtl::Texture &);

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

    const mtl::Context *Ctx{nullptr};
    uint32_t OffsetX{0}, OffsetY{0};
    mtl::Extent2D Ex{};
    size_t FrameBytes{0};

    // Shared readback for the private render target.
    mtl::Owned<MTL::Buffer> Staging;

    std::unique_ptr<std::FILE, PipeCloser> Pipe;
    uint64_t FrameCount{0};

    // The file this recording writes.
    // VideoPath and AudioPath name the video-only file muxed into it and the wav muxed with it, and are set only when a video recording also carries audio.
    std::filesystem::path FinalPath, VideoPath, AudioPath;
    // The audio sink: the recording's own file where it records audio only, and the sidecar the mux reads where it carries audio alongside video.
    std::unique_ptr<WavWriter> Wav;
};
