#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

// Streams mono float frames into a 32-bit float wav file.
struct WavWriter {
    // Opens the file for writing. A writer that failed to open reports !IsOpen() and writes nothing.
    WavWriter(const std::filesystem::path &, uint32_t sample_rate);
    ~WavWriter();

    WavWriter(const WavWriter &) = delete;
    WavWriter &operator=(const WavWriter &) = delete;

    bool IsOpen() const;
    // Appends frames. Returns false when a write fails.
    bool Write(std::span<const float>);
    uint64_t FramesWritten() const;

private:
    struct Impl;
    std::unique_ptr<Impl> State;
};
