#pragma once

#include <readerwriterqueue.h>

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <limits>
#include <ostream>
#include <semaphore>
#include <thread>
#include <variant>
#include <vector>

namespace action {
// A final sentinel stops the writer after earlier FIFO records are written.
struct Stop {};
struct Flush {
    std::binary_semaphore *Done;
};

// Single-producer, single-consumer asynchronous log.
template<typename RecordType>
class WriteBehindLog {
public:
    using Serializer = void (*)(const RecordType &, std::ostream &);

    WriteBehindLog(std::ostream &out, Serializer serialize, size_t initial_capacity = 1024)
        : Queue(initial_capacity), Out(out), Serialize(serialize), Writer([this] { Run(); }) {}
    ~WriteBehindLog() { Stop(); }

    WriteBehindLog(const WriteBehindLog &) = delete;
    WriteBehindLog &operator=(const WriteBehindLog &) = delete;

    // Enqueues without I/O, serialization, or blocking.
    void Enqueue(RecordType &&record) { Queue.enqueue(Record{std::move(record)}); }

    // Waits until every earlier record is durable without closing the log.
    void Flush() {
        if (!Writer.joinable()) return;
        std::binary_semaphore done{0};
        Queue.enqueue(Record{action::Flush{&done}});
        done.acquire();
    }

    // Enqueues the stop sentinel and joins after prior records are written.
    void Stop() {
        if (!Writer.joinable()) return;
        Queue.enqueue(Record{action::Stop{}});
        Writer.join();
    }

private:
    using Record = std::variant<RecordType, action::Flush, action::Stop>;

    // Waits for a record, writes the available batch, and flushes.
    void Run() {
        Record item;
        for (bool stopping = false; !stopping;) {
            Queue.wait_dequeue(item);
            do {
                if (std::holds_alternative<action::Stop>(item)) {
                    stopping = true;
                    break;
                }
                if (const auto *flush = std::get_if<action::Flush>(&item)) {
                    Out.flush();
                    flush->Done->release();
                    continue;
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
// Returns working restore directories in descending modification order.
std::vector<RestoreSession> ListRestoreSessions();
// Creates a working restore directory and prunes old directories.
std::filesystem::path ReserveRestoreSession();

// Opens the action log and starts its writer thread.
void StartLog(std::filesystem::path, bool append = false);
void FlushLog();
// Flushes and joins the writer, returning an empty path when no records were written.
std::filesystem::path StopLog();

// Advances derived viewport state between replayed actions.
using ReplayTick = void (*)(entt::registry &, entt::entity viewport);

// Replays records after `skip` and returns false when the log cannot be opened.
bool ReplayLog(
    entt::registry &, entt::entity viewport, const std::filesystem::path &, ReplayTick,
    uint64_t skip = 0, uint64_t count = std::numeric_limits<uint64_t>::max(), bool record = false
);
} // namespace action
