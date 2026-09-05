#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>

// A background task with a monitor for progress display and cooperative cancellation.
// `work` runs on its own thread and receives the monitor.
template<typename Result, typename MonitorType>
struct Job {
    Job(std::string title, auto &&work)
        : Title(std::move(title)), Monitor(std::make_shared<MonitorType>()),
          ResultFuture(std::async(std::launch::async, [monitor = Monitor, work = std::forward<decltype(work)>(work)]() mutable { return work(*monitor); })) {}

    // Return the completed result or nullopt without blocking.
    std::optional<Result> Poll() {
        if (!ResultFuture.valid() || ResultFuture.wait_for(std::chrono::seconds{0}) != std::future_status::ready) return {};
        return ResultFuture.get();
    }

    void RequestCancel() { Monitor->CancelRequested.store(true, std::memory_order_relaxed); }
    bool Cancelled() const { return Monitor->CancelRequested.load(std::memory_order_relaxed); }

    std::string Title;
    std::shared_ptr<MonitorType> Monitor;
    std::future<Result> ResultFuture;
};
