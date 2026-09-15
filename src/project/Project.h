#pragma once

#include "File.h"
#include "ProcessEvents.h"
#include "action/Action.h"
#include "action/ActionDrain.h"
#include "project/EntityStore.h"
#include "project/store/History.h"
#include "viewport/CameraView.h"

namespace project {
// Construct before engine initialization so all document entities use the versioned allocator.
// Close history before tearing down the scene's stores.
struct Project {
    explicit Project(state::Scene &);
    ~Project();
    void TrackStores(state::Entity viewport);
    bool Begin(const std::filesystem::path &);
    bool New(const std::filesystem::path &, bool empty = true);
    bool Open(const std::filesystem::path &working, const std::filesystem::path &saved = {});
    bool Save();
    bool SaveArchive(const std::filesystem::path &, std::span<const std::byte> workspace = {});
    // Obtain user confirmation before replacing an existing project directory.
    bool SaveAs(const std::filesystem::path &directory, std::span<const std::byte> workspace = {});
    bool RevertSaved();
    bool ClearHistory();
    bool Close();

    int Do(action::Action, std::string label = {});
    void Frame(action::Drained);
    void Enqueue(action::Action a) { Deferred.push_back(std::move(a)); }
    bool HasStaged() const { return GestureBase.has_value(); }
    void CancelGesture();
    void Settle(EventPass = EventPass::Frame);
    void Navigate(int node);
    void Undo();
    void Redo();
    bool Replay();
    void RequestNavigate(int node) { Navigation = node; }
    bool Audit(std::string &why);

    store::History History;
    File::DirectoryLock DirectoryLock;
    std::filesystem::path SavedPath;
    std::vector<std::byte> RestoredWorkspace;
    EntityStore Entities;
    uint64_t MemoryCap{64ull << 20};
    uint64_t Revision{};

    struct ReplayInputs {
        CameraView View{};
        uvec2 ViewportExtent{};
        vec2 DisplayFramebufferScale{1, 1};
        float DeltaTime{}, PlaybackFrame{};
        int CurrentFrame{};
        bool FixedFrameStep{};
        EventPass Pass{EventPass::Frame};
    };
    struct Command {
        ReplayInputs Inputs;
        action::Action Value;
    };
    std::vector<Command> Commands;
    std::vector<action::Action> Deferred;
    std::optional<size_t> StageFirst;
    std::optional<store::Snapshot> GestureBase;
    std::optional<int> Navigation;
    void Tick(const action::Action &, EventPass = EventPass::Frame);
    bool ApplyCommand(action::Action, EventPass, bool staged = false);
    int Commit(std::string label);
    void FinishGesture(EventPass);
    void ReleaseGesture();
    void ClearInteraction();
    void AfterRestore();
    state::Scene &R;
    state::Entity Viewport{state::Null};
};
} // namespace project
