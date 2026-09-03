#pragma once

#include "metal/Image.h"

#include <filesystem>
#include <span>

struct WavWriter;

// Record a fixed viewport region to H.264 through ffmpeg, or record mono float audio to a .wav path.
// Require ffmpeg on PATH for video; IsActive returns false when startup fails.
// A nonzero audio sample rate muxes captured audio into the video during Stop.
struct VideoRecorder {
    VideoRecorder(
        const mtl::Context &,
        const std::filesystem::path &output_path, uint32_t x, uint32_t y, mtl::Extent2D extent, int fps,
        uint32_t audio_sample_rate = 0
    );
    ~VideoRecorder();

    VideoRecorder(const VideoRecorder &) = delete;
    VideoRecorder &operator=(const VideoRecorder &) = delete;

    // Capture the configured region from the render target.
    void CaptureFrame(const mtl::Texture &);

    // Capture mono frames at the configured sample rate.
    void CaptureAudio(std::span<const float>);

    bool IsActive() const { return Wav || Pipe; }
    uint64_t CapturedFrameCount() const { return FrameCount; }

    // Return whether any recording failed during this process.
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

    NS::SharedPtr<MTL::Buffer> Staging;

    std::unique_ptr<std::FILE, PipeCloser> Pipe;
    uint64_t FrameCount{0};

    std::filesystem::path FinalPath, VideoPath, AudioPath;
    std::unique_ptr<WavWriter> Wav;
};
