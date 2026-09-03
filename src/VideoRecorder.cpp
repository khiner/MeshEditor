#include "VideoRecorder.h"
#include "audio/WavWriter.h"
#include "metal/Buffer.h"

#include <sys/wait.h>

#include <print>

void VideoRecorder::PipeCloser::operator()(std::FILE *p) const noexcept {
    if (p) ::pclose(p);
}

namespace {
// Preserve recording failures until the process reports its exit status.
bool RecordingFailed{false};

// Decode the subprocess exit status and return -1 for termination by a signal.
int ExitCode(int wait_status) { return WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1; }

std::string BuildFfmpegCommand(const std::filesystem::path &out, mtl::Extent2D extent, int fps) {
    // Use full-chroma H.264 at CRF 10 to preserve edge and aliasing detail in deterministic captures.
    return std::format(
        "ffmpeg -y -loglevel warning -f rawvideo -pix_fmt bgra -s {}x{} -r {} -i - "
        "-c:v libx264 -pix_fmt yuv444p -preset veryfast -crf 10 "
        "-x264-params keyint=infinite:threads=8 -bitexact \"{}\"",
        extent.Width, extent.Height, fps, out.string()
    );
}

} // namespace

VideoRecorder::VideoRecorder(
    const mtl::Context &ctx,
    const std::filesystem::path &output_path, uint32_t x, uint32_t y, mtl::Extent2D extent, int fps,
    uint32_t audio_sample_rate
) : Ctx{&ctx}, OffsetX{x}, OffsetY{y}, Ex{extent},
    FrameBytes{size_t(extent.Width) * extent.Height * 4}, FinalPath{output_path} {
    // Audio-only recording writes directly to a float WAV file.
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
    if (extent.Width == 0 || extent.Height == 0) {
        std::println(stderr, "VideoRecorder: viewport extent is zero; not recording.");
        return;
    }
    if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
        std::println(stderr, "VideoRecorder: 'ffmpeg' not found on PATH; not recording.");
        return;
    }

    Staging = mtl::NewBuffer(ctx, FrameBytes);

    // Encode video with audio to a sidecar because muxing requires a completed video stream.
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
    std::println("VideoRecorder: {}x{} @ {}fps{} -> {}", extent.Width, extent.Height, fps, Wav ? " with audio" : "", output_path.string());
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

    // Specify mono layout because ffmpeg logs a heap address when inferring an unstated mono layout.
    Wav.reset();
    const auto mux = std::format(
        "ffmpeg -y -loglevel warning -i \"{}\" -ch_layout mono -i \"{}\" "
        "-c:v copy -c:a aac -b:a 192k -shortest \"{}\"",
        VideoPath.string(), AudioPath.string(), FinalPath.string()
    );
    // Preserve both sidecars after a failed mux for recovery and diagnosis.
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

void VideoRecorder::CaptureFrame(const mtl::Texture &texture) {
    // Count audio-only frames for pacing and duration accounting.
    if (!Pipe) {
        if (Wav) ++FrameCount;
        return;
    }

    if (!mtl::CopyTextureRegion(*Ctx, texture, OffsetX, OffsetY, Ex, Staging.get(), Ex.Width * 4)) {
        std::println(stderr, "VideoRecorder: frame copy failed; stopping.");
        Stop();
        return;
    }

    if (const auto written = std::fwrite(Staging->contents(), 1, FrameBytes, Pipe.get()); written != FrameBytes) {
        std::println(stderr, "VideoRecorder: pipe write short ({}/{}); stopping.", written, FrameBytes);
        Stop();
        return;
    }
    ++FrameCount;
}
