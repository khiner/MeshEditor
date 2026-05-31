#pragma once

#include <readerwriterqueue.h>

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <thread>
#include <variant>
#include <vector>

namespace action {
// Poison pill enqueued last at shutdown: wakes the writer and tells it to exit after it has drained
// every real record ahead of it in the FIFO.
struct Stop {};

// Single-producer/single-consumer write-behind log. The producer thread calls Enqueue with zero IO
// or serialization; a dedicated writer thread serializes each record and appends it to `out`. Generic
// over the record type `T` and a `Serialize(const T&, std::ostream&)` callback, so it stays unaware of
// what it is logging. `out` must outlive the log.
template<typename T>
class WriteBehindLog {
public:
    using Serializer = void (*)(const T &, std::ostream &);

    WriteBehindLog(std::ostream &out, Serializer serialize, std::size_t initial_capacity = 1024)
        : Queue(initial_capacity), Out(out), Serialize(serialize), Writer([this] { Run(); }) {}
    ~WriteBehindLog() { Stop(); }

    WriteBehindLog(const WriteBehindLog &) = delete;
    WriteBehindLog &operator=(const WriteBehindLog &) = delete;

    // Push a record onto the queue and return immediately: no IO, no serialization, no blocking.
    void Enqueue(T &&record) { Queue.enqueue(Record{std::move(record)}); }

    // Enqueue the poison pill and join the writer, which drains all prior records first. Idempotent.
    void Stop() {
        if (!Writer.joinable()) return;
        Queue.enqueue(Record{action::Stop{}});
        Writer.join();
    }

private:
    using Record = std::variant<T, action::Stop>;

    // Block for a record, then drain the rest of the burst before a single flush.
    void Run() {
        Record item;
        for (bool stopping = false; !stopping;) {
            Queue.wait_dequeue(item);
            do {
                if (std::holds_alternative<action::Stop>(item)) {
                    stopping = true;
                    break;
                }
                Serialize(std::get<T>(item), Out);
            } while (Queue.try_dequeue(item));
            Out.flush();
        }
    }

    moodycamel::BlockingReaderWriterQueue<Record> Queue;
    std::ostream &Out;
    Serializer Serialize;
    std::thread Writer;
};

// A `.mea` (Mesh Editor Action) log file in the replay dir, tagged with the unix-seconds timestamp parsed from its name.
struct ReplayLogFile {
    std::filesystem::path Path;
    std::int64_t UnixSeconds;
};

// Replay logs found in the replay dir, sorted most-recent first.
std::vector<ReplayLogFile> ListReplayLogs();

// Create the `replay/` dir, retain only the newest REPLAY_LOG_RETAIN-1 logs, then open a fresh
// `replay/<unix_seconds>.mea` append stream (so at most REPLAY_LOG_RETAIN logs exist after opening).
std::ofstream OpenLogStream();

// Open a fresh `.mea` log and start its writer thread.
void StartLog();
// Flush and join the writer, dropping the just-opened log file if nothing was recorded.
void StopLog();

// Advances the viewport between replayed actions so those resolving against rendered state see it up to date.
using ReplayTick = void (*)(entt::registry &, entt::entity viewport);

// Apply each action in a `.mea` log to the current scene, ticking via `tick` between them. Replayed actions
// are not re-logged. False if the log can't be opened.
bool ReplayLog(entt::registry &, entt::entity viewport, const std::filesystem::path &, ReplayTick tick);
} // namespace action
