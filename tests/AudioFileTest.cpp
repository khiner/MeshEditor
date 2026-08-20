#include "RunSuites.h"
#include "audio/AudioSystem.h"
#include "audio/WavWriter.h"

#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

using namespace boost::ut;
namespace fs = std::filesystem;

int main() {
    "CoreAudio float wav round trip"_test = [] {
        constexpr uint32_t SampleRate = 48'000;
        const auto path = fs::temp_directory_path() / "MeshEditor-audio-file-test.wav";
        fs::remove(path);

        std::vector<float> source(8'192);
        for (size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 257) - 128) / 128.f;
        {
            WavWriter writer{path, SampleRate};
            expect(writer.IsOpen());
            expect(writer.Write(std::span{source}.first(3'000)));
            expect(writer.Write(std::span{source}.subspan(3'000)));
            expect(writer.FramesWritten() == source.size());
        }

        const auto decoded = LoadAudioFrames(path.string(), SampleRate);
        expect(decoded.size() == source.size());
        float max_error = 0.f;
        for (size_t i = 0; i < std::min(decoded.size(), source.size()); ++i) {
            max_error = std::max(max_error, std::abs(decoded[i] - source[i]));
        }
        expect(max_error <= 0.0000001_f) << max_error;

        const auto half_rate = LoadAudioFrames(path.string(), SampleRate / 2);
        const auto expected_half_frames = source.size() / 2;
        expect(half_rate.size() >= expected_half_frames - 2);
        expect(half_rate.size() <= expected_half_frames + 2);

        fs::remove(path);
    };

    return RunSuites();
}
