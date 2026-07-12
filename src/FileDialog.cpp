#include "FileDialog.h"

#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_error.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace FileDialog {
namespace {
std::mutex QueueMutex;
std::vector<std::function<void()>> Queue; // Completed dialogs, drained on the main thread by Pump.

// Owns the callback and the filter strings SDL points into, from the Show* call until the callback fires.
struct Request {
    OnPick Callback;
    std::vector<std::string> Strings; // Backing storage for filter names and patterns.
    std::vector<SDL_DialogFileFilter> Filters;
    std::string DefaultLocation;
};

void SDLCALL DialogCallback(void *userdata, const char *const *filelist, int) {
    std::unique_ptr<Request> req{static_cast<Request *>(userdata)};
    if (!filelist) {
        std::cerr << "File dialog error: " << SDL_GetError() << std::endl;
        return;
    }
    if (!filelist[0]) return; // A null first entry means the user canceled.
    std::lock_guard lock{QueueMutex};
    Queue.emplace_back([cb = std::move(req->Callback), path = std::filesystem::path{filelist[0]}] { cb(path); });
}

Request *MakeRequest(std::span<const Filter> filters, OnPick callback) {
    auto req = std::make_unique<Request>();
    req->Callback = std::move(callback);
    req->Strings.reserve(filters.size() * 2); // Keeps the c_str() pointers below stable.
    req->Filters.reserve(filters.size());
    for (const auto &f : filters) {
        const auto &name = req->Strings.emplace_back(f.Name);
        const auto &pattern = req->Strings.emplace_back(f.Pattern);
        req->Filters.emplace_back(SDL_DialogFileFilter{name.c_str(), pattern.c_str()});
    }
    return req.release(); // SDL holds it as userdata until DialogCallback frees it.
}
} // namespace

void ShowOpen(std::span<const Filter> filters, OnPick callback) {
    auto *req = MakeRequest(filters, std::move(callback));
    SDL_ShowOpenFileDialog(DialogCallback, req, nullptr, req->Filters.data(), int(req->Filters.size()), nullptr, false);
}

void ShowSave(std::span<const Filter> filters, std::string_view default_name, OnPick callback) {
    auto *req = MakeRequest(filters, std::move(callback));
    req->DefaultLocation = default_name;
    SDL_ShowSaveFileDialog(DialogCallback, req, nullptr, req->Filters.data(), int(req->Filters.size()), req->DefaultLocation.c_str());
}

void ShowPickFolder(OnPick callback) {
    auto *req = MakeRequest({}, std::move(callback));
    SDL_ShowOpenFolderDialog(DialogCallback, req, nullptr, nullptr, false);
}

void Pump() {
    std::vector<std::function<void()>> ready;
    {
        std::lock_guard lock{QueueMutex};
        ready.swap(Queue);
    }
    for (auto &deliver : ready) deliver();
}
} // namespace FileDialog
