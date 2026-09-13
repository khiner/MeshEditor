#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Append queued data in order on one writer thread.
namespace store {
struct WriteBehind {
    ~WriteBehind() { Stop(); }

    std::string Start(std::vector<std::filesystem::path> paths, bool truncate) {
        const auto mode = std::ios::binary | (truncate ? std::ios::trunc : std::ios::app);
        std::vector<std::ofstream> streams(paths.size());
        for (size_t i = 0; i < paths.size(); ++i) {
            streams[i].open(paths[i], mode);
            if (!streams[i]) return "cannot open " + paths[i].string();
        }
        if (auto error = Stop(); !error.empty()) return error;
        Streams = std::move(streams);
        Paths = std::move(paths);
        PeakQueuedBytes = 0;
        Running = true;
        Writer = std::thread([this] { Run(); });
        return {};
    }

    // Complete queued writes, flush streams, and join the writer thread.
    std::string Stop() {
        if (!Writer.joinable()) return Failure;
        {
            std::lock_guard lk{M};
            Running = false;
        }
        CV.notify_one();
        Writer.join();
        for (size_t i = 0; i < Streams.size(); ++i) {
            Streams[i].close();
            if (!Streams[i] && Failure.empty()) Failure = "cannot close " + Paths[i].string();
        }
        Streams.clear();
        return std::exchange(Failure, {});
    }

    bool Started() const { return Writer.joinable(); }

    void Append(size_t file, std::vector<std::byte> bytes) {
        {
            std::unique_lock lk{M};
            // Limit queued and active writes to 16 MiB, allowing a larger append when no writes are pending.
            Done.wait(lk, [&] { return QueuedBytes == 0 || QueuedBytes + bytes.capacity() <= (16u << 20); });
            QueuedBytes += bytes.capacity();
            PeakQueuedBytes = std::max(PeakQueuedBytes, QueuedBytes);
            Queue.push_back({file, std::move(bytes)});
        }
        CV.notify_one();
    }

    // Wait for queued writes and stream flushes, returning the first error.
    std::string FlushAndWait() {
        if (!Writer.joinable()) return Failure;
        std::unique_lock lk{M};
        FlushRequested = true;
        CV.notify_one();
        Done.wait(lk, [&] { return !FlushRequested; });
        return Failure;
    }

    std::string Error() {
        std::lock_guard lk{M};
        return Failure;
    }

    // Vector capacities queued or being written, excluding stream and allocator overhead.
    std::pair<uint64_t, uint64_t> PendingMemory() const {
        std::lock_guard lk{M};
        return {QueuedBytes, PeakQueuedBytes};
    }

    struct Item {
        size_t File;
        std::vector<std::byte> Bytes;
    };

    void Run() {
        std::unique_lock lk{M};
        for (;;) {
            CV.wait(lk, [&] { return !Queue.empty() || FlushRequested || !Running; });
            while (!Queue.empty()) {
                const auto item = std::move(Queue.front());
                Queue.pop_front();
                if (Failure.empty()) {
                    lk.unlock();
                    Streams[item.File].write(reinterpret_cast<const char *>(item.Bytes.data()), std::streamsize(item.Bytes.size()));
                    lk.lock();
                    if (!Streams[item.File]) Failure = "cannot write " + Paths[item.File].string();
                }
                QueuedBytes -= item.Bytes.capacity();
                Done.notify_one();
            }
            for (size_t i = 0; i < Streams.size(); ++i) {
                Streams[i].flush();
                if (!Streams[i] && Failure.empty()) Failure = "cannot flush " + Paths[i].string();
            }
            FlushRequested = false;
            Done.notify_one();
            if (!Running) return;
        }
    }

    std::vector<std::ofstream> Streams;
    std::vector<std::filesystem::path> Paths;
    std::string Failure;
    std::thread Writer;
    mutable std::mutex M;
    std::condition_variable CV, Done;
    std::deque<Item> Queue;
    uint64_t QueuedBytes{}, PeakQueuedBytes{};
    bool Running{}, FlushRequested{};
};
} // namespace store
