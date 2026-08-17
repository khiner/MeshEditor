#include "VideoRecorder.h"
#include "audio/WavWriter.h"
#include "vulkan/Image.h"

#include <sys/wait.h>

#include <print>

void VideoRecorder::PipeCloser::operator()(std::FILE *p) const noexcept {
    if (p) ::pclose(p);
}

namespace {
// Set by any recording that failed, so a run that produced a truncated or missing capture leaves the process with a nonzero status.
bool RecordingFailed{false};

// The subprocess's exit code, decoded from the wait status popen and system return.
// A subprocess killed by a signal never exited, and reports as -1.
int ExitCode(int wait_status) { return WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1; }

std::string BuildFfmpegCommand(const std::filesystem::path &out, vk::Extent2D extent, int fps) {
    // `-y` overwrite, `-loglevel warning` mutes per-frame progress.
    // Input: raw BGRA frames on stdin with declared size/framerate.
    // `-vf vflip`: the viewport uses a negative-height Vulkan viewport so row 0 in image
    // memory is the bottom of the screen; flip rows to get upright video.
    // Output: H.264 in yuv444p (full chroma) at crf 10 to preserve edge/aliasing detail.
    // An unchanged render produces a byte-identical file.
    return std::format(
        "ffmpeg -y -loglevel warning -f rawvideo -pix_fmt bgra -s {}x{} -r {} -i - "
        "-vf vflip -c:v libx264 -pix_fmt yuv444p -preset veryfast -crf 10 "
        "-x264-params keyint=infinite:threads=8 -bitexact \"{}\"",
        extent.width, extent.height, fps, out.string()
    );
}

} // namespace

VideoRecorder::VideoRecorder(
    const VulkanResources &vk_res, mvk::BufferContext &buf_ctx,
    const std::filesystem::path &output_path, vk::Offset3D offset, vk::Extent2D extent, int fps,
    uint32_t audio_sample_rate
) : Device{vk_res.Device}, Queue{vk_res.Queue}, Offset{offset}, Ex{extent},
    FrameBytes{extent.width * extent.height * 4}, FinalPath{output_path} {
    // Audio only: the samples stream straight into a float wav, and no GPU or ffmpeg resource is touched.
    if (output_path.extension() == ".wav") {
        if (audio_sample_rate == 0) {
            std::println(stderr, "VideoRecorder: {} needs audio; not recording.", output_path.string());
            return;
        }
        Wav = std::make_unique<WavWriter>(output_path, audio_sample_rate);
        if (!Wav->IsOpen()) {
            Wav.reset();
            std::println(stderr, "VideoRecorder: could not open {}; not recording.", output_path.string());
            return;
        }
        std::println("VideoRecorder: audio only @ {} Hz -> {}", audio_sample_rate, output_path.string());
        return;
    }
    if (extent.width == 0 || extent.height == 0) {
        std::println(stderr, "VideoRecorder: viewport extent is zero; not recording.");
        return;
    }
    if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
        std::println(stderr, "VideoRecorder: 'ffmpeg' not found on PATH; not recording.");
        return;
    }

    CommandPool = Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, vk_res.QueueFamily});
    CommandBuffer = std::move(Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    Fence = Device.createFenceUnique({});
    Staging.emplace(buf_ctx, FrameBytes, mvk::MemoryUsage::CpuOnly, vk::BufferUsageFlagBits::eTransferDst);

    // Muxing needs the finished video, so with audio the encode goes to a neighbouring file and the final path is written once at Stop.
    // Without audio the video path is written directly.
    auto video_path = output_path;
    if (audio_sample_rate > 0) {
        VideoPath = std::filesystem::path{output_path}.replace_extension(".video.mp4");
        AudioPath = std::filesystem::path{output_path}.replace_extension(".audio.wav");
        video_path = VideoPath;
        Wav = std::make_unique<WavWriter>(AudioPath, audio_sample_rate);
        if (!Wav->IsOpen()) {
            Wav.reset();
            std::println(stderr, "VideoRecorder: could not open {}; recording without audio", AudioPath.string());
        }
    }

    const auto cmd = BuildFfmpegCommand(video_path, extent, fps);
    std::println("VideoRecorder: {}x{} @ {}fps{} -> {}", extent.width, extent.height, fps, Wav ? " with audio" : "", output_path.string());
    Pipe.reset(::popen(cmd.c_str(), "w"));
    if (!Pipe) std::println(stderr, "VideoRecorder: popen failed");
}

VideoRecorder::~VideoRecorder() { Stop(); }

bool VideoRecorder::AnyFailed() { return RecordingFailed; }

void VideoRecorder::Stop() {
    if (!Pipe) {
        if (Wav) std::println("VideoRecorder: wrote {} audio frames to {}", Wav->FramesWritten(), FinalPath.string());
        Wav.reset();
        return;
    }

    std::fflush(Pipe.get());
    const int encode = ExitCode(::pclose(Pipe.release()));
    if (encode != 0) {
        RecordingFailed = true;
        std::println(stderr, "VideoRecorder: ffmpeg exited {} encoding {}", encode, VideoPath.empty() ? FinalPath.string() : VideoPath.string());
    }
    std::println("VideoRecorder: wrote {} frames", FrameCount);
    if (!Wav) return;

    // The sidecar carries its own header, so the mux reads its format rather than being told it.
    Wav.reset();
    const auto mux = std::format(
        "ffmpeg -y -loglevel warning -i \"{}\" -i \"{}\" "
        "-c:v copy -c:a aac -b:a 192k -shortest \"{}\"",
        VideoPath.string(), AudioPath.string(), FinalPath.string()
    );
    // A failed mux keeps both inputs, so the capture is still there to recover or to diagnose from.
    if (const int mux_status = ExitCode(std::system(mux.c_str())); mux_status != 0) {
        RecordingFailed = true;
        std::println(stderr, "VideoRecorder: ffmpeg exited {} muxing audio into {}; keeping {} and {}", mux_status, FinalPath.string(), VideoPath.string(), AudioPath.string());
        return;
    }
    std::println("VideoRecorder: muxed audio into {}", FinalPath.string());
    std::error_code ec;
    std::filesystem::remove(VideoPath, ec);
    std::filesystem::remove(AudioPath, ec);
}

void VideoRecorder::CaptureAudio(std::span<const float> frames) {
    if (!Wav || Wav->Write(frames)) return;
    RecordingFailed = true;
    std::println(stderr, "VideoRecorder: wav write failed for {}; stopping.", (Pipe ? AudioPath : FinalPath).string());
    Wav.reset();
}

void VideoRecorder::CaptureFrame(vk::Image image) {
    // Audio only: no pixels leave the GPU, and the count keeps the recording's pacing and duration accounting.
    if (!Pipe) {
        if (Wav) ++FrameCount;
        return;
    }

    auto cb = *CommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    mvk::RecordImageToBufferCopy(cb, image, **Staging, Offset, Ex);
    cb.end();

    Queue.submit(vk::SubmitInfo{{}, {}, cb, {}}, *Fence);
    if (Device.waitForFences(*Fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
        std::println(stderr, "VideoRecorder: fence wait failed; stopping.");
        Stop();
        return;
    }
    Device.resetFences(*Fence);

    if (const auto written = std::fwrite(Staging->GetMappedData().data(), 1, FrameBytes, Pipe.get()); written != FrameBytes) {
        std::println(stderr, "VideoRecorder: pipe write short ({}/{}); stopping.", written, FrameBytes);
        Stop();
        return;
    }
    ++FrameCount;
}
