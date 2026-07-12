#pragma once

#include <readerwriterqueue.h>

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <ostream>
#include <thread>
#include <variant>
#include <vector>

namespace action {
// Poison pill enqueued last at shutdown:
// Wakes the writer and tells it to exit after it has drained every real record ahead of it in the FIFO.
struct Stop {};

// SPSC write-behind log:
// Producer calls Enqueue with no IO or serialization.
// Writer thread serializes each record and appends it to `out`.
template<typename RecordType>
class WriteBehindLog {
public:
    using Serializer = void (*)(const RecordType &, std::ostream &);

    WriteBehindLog(std::ostream &out, Serializer serialize, size_t initial_capacity = 1024)
        : Queue(initial_capacity), Out(out), Serialize(serialize), Writer([this] { Run(); }) {}
    ~WriteBehindLog() { Stop(); }

    WriteBehindLog(const WriteBehindLog &) = delete;
    WriteBehindLog &operator=(const WriteBehindLog &) = delete;

    // Push a record onto the queue and return immediately: no IO, no serialization, no blocking.
    void Enqueue(RecordType &&record) { Queue.enqueue(Record{std::move(record)}); }

    // Enqueue the poison pill and join the writer, which drains all prior records first. Idempotent.
    void Stop() {
        if (!Writer.joinable()) return;
        Queue.enqueue(Record{action::Stop{}});
        Writer.join();
    }

private:
    using Record = std::variant<RecordType, action::Stop>;

    // Block for a record, drain the rest of the burst, then flush.
    void Run() {
        Record item;
        for (bool stopping = false; !stopping;) {
            Queue.wait_dequeue(item);
            do {
                if (std::holds_alternative<action::Stop>(item)) {
                    stopping = true;
                    break;
                }
                Serialize(std::get<RecordType>(item), Out);
            } while (Queue.try_dequeue(item));
            Out.flush();
        }
    }

    moodycamel::BlockingReaderWriterQueue<Record> Queue;
    std::ostream &Out;
    Serializer Serialize;
    std::thread Writer;
};

struct RestoreSession {
    std::filesystem::path Path;
    uint32_t UnixSeconds;
};
// Working restore directories, sorted most-recent first.
std::vector<RestoreSession> ListRestoreSessions();
// Create and return a new working restore directory, pruning old ones.
std::filesystem::path ReserveRestoreSession();

// Open the action log and start its writer thread. Truncates unless `append`.
void StartLog(std::filesystem::path, bool append = false);
// Flush and join the writer. Returns the log path, or empty if nothing was recorded (an empty new log is dropped).
std::filesystem::path StopLog();

// Advance the viewport between actions so actions read updated derived state.
using ReplayTick = void (*)(entt::registry &, entt::entity viewport);

// Apply each action to the current scene after the first `skip` records, ticking between them.
// False if the log can't be opened.
bool ReplayLog(entt::registry &, entt::entity viewport, const std::filesystem::path &, ReplayTick, uint64_t skip = 0);
} // namespace action
