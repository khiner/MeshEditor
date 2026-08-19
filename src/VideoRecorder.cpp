#include "VideoRecorder.h"
#include "audio/WavWriter.h"
#include "metal/Buffer.h"

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

std::string BuildFfmpegCommand(const std::filesystem::path &out, mtl::Extent2D extent, int fps) {
    // `-y` overwrite, `-loglevel warning` mutes per-frame progress.
    // Input: raw BGRA frames on stdin with declared size/framerate.
    // Output: H.264 in yuv444p (full chroma) at crf 10 to preserve edge/aliasing detail.
    // An unchanged render produces a byte-identical file.
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
    if (extent.Width == 0 || extent.Height == 0) {
        std::println(stderr, "VideoRecorder: viewport extent is zero; not recording.");
        return;
    }
    if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
        std::println(stderr, "VideoRecorder: 'ffmpeg' not found on PATH; not recording.");
        return;
    }

    Staging = mtl::NewBuffer(ctx, FrameBytes);

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

    // The sidecar holds its own header, so the mux reads its format rather than being told it.
    // `-ch_layout mono` is the exception: ffmpeg's guess for an unstated mono layout prints a heap address.
    Wav.reset();
    const auto mux = std::format(
        "ffmpeg -y -loglevel warning -i \"{}\" -ch_layout mono -i \"{}\" "
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

void VideoRecorder::CaptureFrame(const mtl::Texture &texture) {
    // Audio only: no pixels leave the GPU, and the count keeps the recording's pacing and duration accounting.
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
