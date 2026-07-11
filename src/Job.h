#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>

// Progress and cancellation state shared between a background job and the UI.
// The worker writes Progress and polls CancelRequested at its checkpoints, so a cancel
// takes effect at the next checkpoint.
struct JobMonitor {
    std::atomic<float> Progress{0.f}; // Fraction complete. 0 while indeterminate.
    std::atomic<bool> CancelRequested{false};

    void RequestCancel() { CancelRequested.store(true, std::memory_order_relaxed); }
    bool Cancelled() const { return CancelRequested.load(std::memory_order_relaxed); }
};

// A background task with a monitor for progress display and cooperative cancellation.
// `work` runs on its own thread and receives the monitor.
template<typename Result>
struct Job {
    Job(std::string title, auto &&work)
        : Title(std::move(title)), Monitor(std::make_shared<JobMonitor>()),
          ResultFuture(std::async(std::launch::async, [monitor = Monitor, work = std::forward<decltype(work)>(work)]() mutable { return work(*monitor); })) {}

    // The result once ready, nullopt while still running. Never blocks.
    std::optional<Result> Poll() {
        if (!ResultFuture.valid() || ResultFuture.wait_for(std::chrono::seconds{0}) != std::future_status::ready) return {};
        return ResultFuture.get();
    }

    std::string Title;
    std::shared_ptr<JobMonitor> Monitor;
    std::future<Result> ResultFuture;
};
