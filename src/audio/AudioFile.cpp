#include "AudioSystem.h"
#include "CoreAudioTypes.h"
#include "WavWriter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>

namespace {
CFURLRef FileUrl(const std::filesystem::path &path) {
    const auto utf8 = path.string();
    return CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(utf8.data()), utf8.size(), false
    );
}

} // namespace

struct WavWriter::Impl {
    ExtAudioFileRef File{nullptr};
    uint64_t FramesWritten{0};
};

WavWriter::WavWriter(const std::filesystem::path &path, uint32_t sample_rate) : State{std::make_unique<Impl>()} {
    if (!sample_rate) return;
    CFURLRef url = FileUrl(path);
    if (!url) return;
    const auto format = MonoFloatAudioFormat(sample_rate);
    const OSStatus status = ExtAudioFileCreateWithURL(url, kAudioFileWAVEType, &format, nullptr, kAudioFileFlags_EraseFile, &State->File);
    CFRelease(url);
    if (status != noErr) State->File = nullptr;
}

WavWriter::~WavWriter() {
    if (State->File) ExtAudioFileDispose(State->File);
}

bool WavWriter::IsOpen() const { return State->File != nullptr; }

bool WavWriter::Write(std::span<const float> frames) {
    if (!State->File) return false;
    while (!frames.empty()) {
        constexpr size_t MaxFramesPerWrite = std::numeric_limits<UInt32>::max() / sizeof(float);
        const auto count = UInt32(std::min(frames.size(), MaxFramesPerWrite));
        const AudioBufferList buffers{
            .mNumberBuffers = 1,
            .mBuffers = {{.mNumberChannels = 1, .mDataByteSize = count * UInt32(sizeof(float)), .mData = const_cast<float *>(frames.data())}},
        };
        if (ExtAudioFileWrite(State->File, count, &buffers) != noErr) return false;
        State->FramesWritten += count;
        frames = frames.subspan(count);
    }
    return true;
}

uint64_t WavWriter::FramesWritten() const { return State->FramesWritten; }

std::vector<float> LoadAudioFrames(const std::string &file_path, uint32_t sample_rate) {
    CFURLRef url = FileUrl(file_path);
    ExtAudioFileRef file{nullptr};
    if (!url || ExtAudioFileOpenURL(url, &file) != noErr) {
        if (url) CFRelease(url);
        std::cerr << std::format("Failed to open audio file: {}\n", file_path);
        return {};
    }
    CFRelease(url);
    const auto fail = [&](std::string_view operation) {
        ExtAudioFileDispose(file);
        std::cerr << std::format("Failed to {} audio file: {}\n", operation, file_path);
        return std::vector<float>{};
    };

    AudioStreamBasicDescription source_format{};
    UInt32 property_size = sizeof(source_format);
    if (ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileDataFormat, &property_size, &source_format) != noErr) return fail("read");
    if (!sample_rate) sample_rate = uint32_t(std::lround(source_format.mSampleRate));
    const auto client_format = MonoFloatAudioFormat(sample_rate);
    if (ExtAudioFileSetProperty(file, kExtAudioFileProperty_ClientDataFormat, sizeof(client_format), &client_format) != noErr) return fail("configure");

    SInt64 source_frames = 0;
    property_size = sizeof(source_frames);
    std::vector<float> frames;
    if (ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileLengthFrames, &property_size, &source_frames) == noErr && source_frames > 0 && source_format.mSampleRate > 0) {
        const double converted = std::ceil(double(source_frames) * double(sample_rate) / source_format.mSampleRate);
        if (converted < double(frames.max_size())) frames.reserve(size_t(converted));
    }

    std::array<float, 4096> chunk{};
    for (;;) {
        UInt32 count = chunk.size();
        AudioBufferList buffers{
            .mNumberBuffers = 1,
            .mBuffers = {{.mNumberChannels = 1, .mDataByteSize = count * UInt32(sizeof(float)), .mData = chunk.data()}},
        };
        if (ExtAudioFileRead(file, &count, &buffers) != noErr) return fail("decode");
        frames.insert(frames.end(), chunk.begin(), chunk.begin() + count);
        if (count == 0) break;
    }
    ExtAudioFileDispose(file);
    return frames;
}
