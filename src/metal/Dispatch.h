#pragma once

#include "metal/AutoreleaseScope.h"
#include "metal/BufferArena.h"
#include "metal/Command.h"

#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>

namespace mtl {
struct ComputePipeline;
template<typename P> struct Driver;
using ComputeDriver = Driver<ComputePipeline>;

// A dispatch over the tiles of its jobs: the push constants that locate its job table, and the jobs' tile total.
struct JobTable {
    JobTablePushConstants Constants;
    uint32_t Groups;
};

// Compute passes recorded in order, each after the buffer writes of the passes before it.
// Passes share one command until the submit, which runs after every command committed before it.
// Submit is the only commit and the only wait.
// Scratch word zero is that error word, which a pass sets when its input violates canonical ownership or topology.
struct ComputeChain {
    // The scratch starts with room for `scratch_words`, so a transaction that knows its size does not grow it mid-recording.
    explicit ComputeChain(BufferContext &, uint64_t scratch_words = 1u);
    // Destroying a chain with unsubmitted passes aborts, except during exception unwinding, which drops them.
    ~ComputeChain();
    ComputeChain(const ComputeChain &) = delete;
    ComputeChain &operator=(const ComputeChain &) = delete;

    // Binds the pipeline, the bindless table and the push constants, then dispatches `groups` threadgroups of `width` threads.
    void Groups(const ComputePipeline &pipeline, const auto &pc, uint32_t groups, uint32_t width = 256u, uint32_t depth = 1u) {
        Record({.Pipeline = &pipeline, .Constants = BytesOf(pc), .Count = groups, .Depth = depth, .Width = width});
    }
    void Threads(const ComputePipeline &pipeline, const auto &pc, uint32_t threads, uint32_t width = 256u) {
        Record({.Pipeline = &pipeline, .Constants = BytesOf(pc), .Count = threads, .Width = width, .Threads = true});
    }
    // Binds the driver's pipeline and its program table, then dispatches as Threads does.
    void Threads(const ComputeDriver &driver, const auto &pc, uint32_t threads, uint32_t width = 256u) {
        Record({.Driver = &driver, .Constants = BytesOf(pc), .Count = threads, .Width = width, .Threads = true});
    }
    // Binds the pipeline as Groups does with the table's push constants, and the table's jobs, which the chain's scratch holds, then dispatches its tiles of `width` threads in `depth` layers.
    void Jobs(const ComputePipeline &pipeline, const JobTable &table, uint32_t width, uint32_t depth = 1u) {
        Record({.Pipeline = &pipeline, .Constants = BytesOf(table.Constants), .Jobs = table.Constants.Jobs.Offset, .Count = table.Groups, .Depth = depth, .Width = width});
    }
    // Binds the driver as Threads does and the table as Jobs does, then dispatches its tiles.
    void Jobs(const ComputeDriver &driver, const JobTable &table, uint32_t width) {
        Record({.Driver = &driver, .Constants = BytesOf(table.Constants), .Jobs = table.Constants.Jobs.Offset, .Count = table.Groups, .Width = width});
    }
    // Dispatches the threadgroup count in the three words at `offset` bytes into `arguments`.
    // The arguments bind directly, so their buffer keeps its storage until the chain commits.
    void Indirect(const ComputePipeline &pipeline, const auto &pc, const Buffer &arguments, uint64_t offset, uint32_t width = 256u) {
        Record({.Pipeline = &pipeline, .Constants = BytesOf(pc), .Width = width, .Arguments = &arguments, .ArgumentsOffset = offset});
    }
    // The recording encoder with the bindless table bound, for passes that bind their own arguments.
    // Such a pass ends with its own ComputeBarrier, or inside Concurrent with the group's, and the chain's later passes rebind the table.
    // The encoder is valid only during the callback, which must not end or submit it.
    // Inside a lane, a copy of the callback records with its lane's step, so it holds no reference to the lane's locals.
    template<typename Fn> void Encode(Fn &&fn) {
        if (OpenLane) {
            LaneSteps[*OpenLane].push_back({.Record = [this, fn = std::decay_t<Fn>{fn}] mutable { Encode(fn); }});
            return;
        }
        RecordLanes();
        const AutoreleaseScope pool;
        std::forward<Fn>(fn)(Encoder());
        DeferredBarrier = DeferredBarrier || Deferring;
    }
    // Records the passes of `record()` as one lane, which records with the chain's other pending lanes before any pass recorded outside a lane and before the submit.
    // A pending lane's k-th pass records with the k-th passes of the other pending lanes, after their (k - 1)-th, without barriers between them.
    // A lane's passes read and write nothing another pending lane's passes touch, and its recording neither submits the chain nor reads its results.
    // A lane opened inside another lane's recording returns the recording to that lane once it ends.
    void Lane(auto &&record) {
        const auto outer = std::exchange(OpenLane, uint32_t(LaneSteps.size()));
        LaneSteps.emplace_back();
        record();
        OpenLane = outer;
    }
    // Records the passes of `record()` after those of the pending lane of `key`, or as a new lane of `key` when none is pending.
    void Lane(uint32_t key, auto &&record) {
        const auto [lane, added] = KeyedLanes.try_emplace(key, uint32_t(LaneSteps.size()));
        if (added) LaneSteps.emplace_back();
        const auto outer = std::exchange(OpenLane, lane->second);
        record();
        OpenLane = outer;
    }
    // Records the later passes of `key`, which has no pending lane, after those of the pending lane of `with`.
    void JoinLane(uint32_t key, uint32_t with) {
        const auto lane = KeyedLanes.find(with)->second;
        KeyedLanes.emplace(key, lane);
    }
    // Records `record(values)` once for `values` and the values the other lanes gather under `kind` at the same step, in the order the lanes gathered them, or at once outside a lane.
    // Their share of the recording allocates at most `words` of scratch.
    template<typename T> void Gather(const void *kind, std::vector<T> values, uint64_t words, std::type_identity_t<std::function<void(std::vector<T>)>> record) {
        if (!OpenLane) return record(std::move(values));
        LaneSteps[*OpenLane].push_back({.Kind = kind, .Values = std::make_shared<std::vector<T>>(std::move(values)), .Gathered = [record = std::move(record)](std::span<const std::shared_ptr<void>> lanes) {
            std::vector<T> gathered;
            for (const auto &values : lanes) {
                for (auto &value : *std::static_pointer_cast<std::vector<T>>(values)) gathered.push_back(std::move(value));
            }
            record(std::move(gathered));
        }});
        Gathered += words;
    }
    // The scratch words the pending lanes' gathered values allocate when they record.
    uint64_t GatheredWords() const { return Gathered; }
    // Records the passes of `fn` without barriers between them, followed by one barrier.
    // No pass in the group reads or writes what another pass in it writes.
    template<typename Fn> void Concurrent(Fn &&fn) {
        if (!OpenLane) RecordLanes();
        const bool outer = std::exchange(Deferring, true);
        std::forward<Fn>(fn)();
        Deferring = outer;
        if (!outer) Barrier();
    }
    // Records the pending lanes in lockstep.
    void RecordLanes();
    // Commits the recorded passes, waits for them, reclaims retired buffers, and throws when a pass failed or set the error word.
    // Without recorded passes it only runs the completions.
    void Submit();
    // Releases every scratch allocation of the submitted passes, which no caller reads after the submit.
    void ReleaseScratch();
    // Identifies the submit that completes the passes recorded now, unique across every chain.
    uint64_t Submission() const { return SubmissionSerial; }
    // Keeps a buffer the recorded passes read or write until the next submit has waited for them.
    void Retain(Buffer &&);
    // Runs `fn` on the host once the next submit completes the passes recorded before it without error.
    void AfterSubmit(std::function<void()> fn);
    // Submits until no function waits on a submit and no pass waits to commit, including those the functions add.
    void Drain();

    // Copies the bytes into the scratch on a boundary of `alignment` bytes, a multiple of 4.
    SlotOffset Upload(std::span<const std::byte> bytes, uint32_t alignment = 4u) {
        const auto words = alignment / 4u;
        const auto range = Scratch.Allocate(uint32_t((bytes.size() + 3u) / 4u) + words - 1u);
        const auto offset = (range.Offset + words - 1u) / words * words;
        Scratch.Buffer.Update(bytes, uint64_t(offset) * 4u);
        return {Scratch.Buffer.Slot, offset};
    }

    BufferContext &Buffers;
    BufferArena<uint32_t> Scratch;

private:
    ComputeEncoder Encoder();
    void Barrier();
    void EndEncoding();
    // A dispatch of the pipeline or the driver with its push constants, and the jobs at word Jobs of the scratch bound directly.
    // It covers Count threads or threadgroups of Width threads in Depth layers, or the threadgroup count in the three words at ArgumentsOffset bytes into Arguments.
    struct Pass {
        const ComputePipeline *Pipeline{};
        const ComputeDriver *Driver{};
        std::span<const std::byte> Constants;
        std::optional<uint32_t> Jobs{};
        uint32_t Count{}, Depth{1u}, Width{};
        bool Threads{};
        const Buffer *Arguments{};
        uint64_t ArgumentsOffset{};
    };
    // Queues the pass with a copy of its push constants in the open lane, or records the pending lanes and then the pass.
    void Record(const Pass &);

    std::optional<Command> Recording;
    NS::SharedPtr<MTL4::ComputeCommandEncoder> Encoding;
    uint64_t SubmissionSerial;
    bool Committed{false};
    bool Deferring{false}, DeferredBarrier{false};
    std::vector<Buffer> Retained;
    std::vector<std::function<void()>> Completions;
    // A lane's recorded pass, or values gathered under Kind, which Gathered records with the values of every lane that gathers under Kind at the step.
    struct LaneStep {
        std::function<void()> Record{};
        const void *Kind{};
        std::shared_ptr<void> Values{};
        std::function<void(std::span<const std::shared_ptr<void>>)> Gathered{};
    };
    std::vector<std::vector<LaneStep>> LaneSteps;
    std::unordered_map<uint32_t, uint32_t> KeyedLanes;
    std::optional<uint32_t> OpenLane;
    uint64_t Gathered{};
};

// Records `record(chain, then)` on a chain of its own and returns the value it passes `then` once the chain drained.
template<typename T> T Drained(BufferContext &buffers, auto &&record) {
    ComputeChain chain{buffers};
    T value{};
    record(chain, [&value](T read) { value = std::move(read); });
    chain.Drain();
    return value;
}

// Uploads the jobs to the chain's scratch, with the job count then each job's first tile, where job i takes `tiles[i]` threadgroups.
// A table of no tiles uploads nothing.
template<typename Job> JobTable UploadJobTable(ComputeChain &chain, std::span<const Job> jobs, std::span<const uint32_t> tiles) {
    std::vector<uint32_t> firsts{uint32_t(jobs.size())};
    uint32_t total = 0u;
    for (const auto job_tiles : tiles) firsts.push_back(std::exchange(total, total + job_tiles));
    if (!total) return {{}, 0u};
    return {{.Jobs = chain.Upload(std::as_bytes(jobs), alignof(Job)), .Tiles = chain.Upload(std::as_bytes(std::span{firsts}))}, total};
}

// Records `record(table, gathered)` with every job of the table in order, once for `jobs` and the jobs the other pending lanes gather under `kind` at the same step, or at once outside a lane.
// Job i takes `tiles[i]` threadgroups, and a table of no tiles records nothing.
// Inside a lane, `record` runs when the lanes record, so it holds no reference to the lane's locals.
void GatherJobs(ComputeChain &chain, const void *kind, auto jobs, std::span<const uint32_t> tiles, auto record) {
    using Job = std::ranges::range_value_t<decltype(jobs)>;
    struct Tiled {
        uint32_t Tiles;
        Job Value;
    };
    std::vector<Tiled> entries;
    for (size_t i = 0u; i < jobs.size(); ++i) entries.push_back({tiles[i], jobs[i]});
    // A job takes its record and its first tile, and a table its count and its alignment.
    chain.Gather(kind, std::move(entries), jobs.size() * (sizeof(Job) / sizeof(uint32_t) + 1u) + alignof(Job) / sizeof(uint32_t), [&chain, record = std::move(record)](std::vector<Tiled> gathered) {
        const auto values = gathered | std::views::transform(&Tiled::Value) | std::ranges::to<std::vector>();
        const auto counts = gathered | std::views::transform(&Tiled::Tiles) | std::ranges::to<std::vector>();
        if (const auto table = UploadJobTable(chain, std::span{values}, counts); table.Groups) record(table, std::span{values});
    });
}
// Records the pipeline over the tiles of `jobs` as GatherJobs does, in tiles of `width` threads in `depth` layers.
void GatherPass(ComputeChain &chain, const ComputePipeline &pipeline, auto jobs, std::span<const uint32_t> tiles, uint32_t width, uint32_t depth = 1u) {
    GatherJobs(chain, &pipeline, jobs, tiles, [&chain, &pipeline, width, depth](const JobTable &table, auto) { chain.Jobs(pipeline, table, width, depth); });
}
// Records the pipeline over `threads_of(job)` threads of each job of `jobs` as the form above does, in tiles of JobTileThreads threads.
void GatherPass(ComputeChain &chain, const ComputePipeline &pipeline, auto jobs, auto &&threads_of)
    requires std::invocable<decltype(threads_of), std::ranges::range_reference_t<decltype(jobs)>>
{
    const auto tiles = jobs | std::views::transform([&](const auto &job) { return uint32_t((uint64_t(std::invoke(threads_of, job)) + JobTileThreads - 1u) / JobTileThreads); }) | std::ranges::to<std::vector>();
    GatherPass(chain, pipeline, jobs, tiles, JobTileThreads);
}
// Records the pipeline over `threads` threads of the job as the first form does.
void GatherPass(ComputeChain &chain, const ComputePipeline &pipeline, const auto &job, uint32_t threads, uint32_t width = JobTileThreads, uint32_t depth = 1u) {
    GatherPass(chain, pipeline, std::span{&job, 1u}, std::array{uint32_t((uint64_t(threads) + width - 1u) / width)}, width, depth);
}
// Records the driver over `threads` threads of the job as GatherPass does.
void GatherPass(ComputeChain &chain, const ComputeDriver &driver, const auto &job, uint32_t threads, uint32_t width = JobTileThreads) {
    GatherJobs(chain, &driver, std::span{&job, 1u}, std::array{uint32_t((uint64_t(threads) + width - 1u) / width)}, [&chain, &driver, width](const JobTable &table, auto) { chain.Jobs(driver, table, width); });
}
} // namespace mtl
