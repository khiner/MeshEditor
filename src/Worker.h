#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <thread>

struct Worker {
    Worker(std::string_view launch_label, std::string_view working_message, std::function<void()> work = {})
        : LaunchLabel(launch_label), WorkingMessage(working_message), Work(work) {}

    ~Worker() {
        if (Thread.joinable()) Thread.join();
    }

    bool Render(); // Returns `true` if the work completed.
    void RenderLauncher(const std::function<void()> &work);
    void RenderLauncher(std::string_view launch_label) {
        LaunchLabel = launch_label;
        RenderLauncher(Work);
    }
    void RenderLauncher() { RenderLauncher(Work); }

    void Launch();
    void Launch(const std::function<void()> &work);

    std::thread Thread;
    std::string LaunchLabel, WorkingMessage;
    std::function<void()> Work;
    std::atomic<bool> Working = false;
};
