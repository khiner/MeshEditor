#include "action/ActionDrain.h"
#include "action/Errors.h"
#include "state/Registry.h"

using namespace action;

namespace {
// The frame's gesture, which is open while it has no actions.
std::optional<std::pair<std::vector<Action>, Phase>> Emitted;
std::vector<Action> Steps;
bool CommitRequested = false, CancelRequested = false;

void Push(Destination destination, Action a) {
    if (destination == Destination::Step) Steps.push_back(std::move(a));
    else if (Emitted) Emitted->first.push_back(std::move(a));
}
} // namespace

namespace action {
bool Begin(Phase phase) {
    if (Emitted && !Emitted->first.empty()) return false;
    Emitted.emplace(std::vector<Action>{}, phase);
    return true;
}
void Append(Destination d, Core a) { Push(d, std::move(a)); }
void Append(Destination d, selection::Action a) { Push(d, std::move(a)); }
void Append(Destination d, object::Action a) { Push(d, std::move(a)); }
void Append(Destination d, view::Action a) { Push(d, std::move(a)); }
void Append(Destination d, physics::Action a) { Push(d, std::move(a)); }
void Append(Destination d, audio::Action a) { Push(d, std::move(a)); }
void Append(Destination d, bone::Action a) { Push(d, std::move(a)); }
void Append(Destination d, timeline::Action a) { Push(d, std::move(a)); }
void Append(Destination d, io::Action a) { Push(d, std::move(a)); }
void Append(Destination d, animation::Action a) { Push(d, std::move(a)); }
void Append(Destination d, mesh::Action a) { Push(d, std::move(a)); }
void Append(Destination d, nodes::Action a) { Push(d, std::move(a)); }
void Append(Destination d, scene::Action a) { Push(d, std::move(a)); }
void Emit(Action a, Phase phase) {
    if (Begin(phase)) Push(Destination::Gesture, std::move(a));
}
void Emit(std::vector<Action> actions, Phase phase) {
    if (!actions.empty() && Begin(phase)) Emitted->first = std::move(actions);
}
void Commit() { CommitRequested = true; }
void Cancel() { CancelRequested = true; }
void Fail(state::Registry &r, std::string message, bool unsupported_extensions) {
    auto &errors = r.Resources.get<Errors>();
    errors.OnlyUnsupportedExtensions = unsupported_extensions && (errors.Messages.empty() || errors.OnlyUnsupportedExtensions);
    errors.Messages.push_back(std::move(message));
}

Drained Drain() {
    auto emitted = std::exchange(Emitted, {});
    if (emitted && emitted->first.empty()) emitted.reset();
    return {std::move(emitted), std::exchange(CommitRequested, false), std::exchange(CancelRequested, false)};
}
std::vector<Action> DrainSteps() { return std::exchange(Steps, {}); }
} // namespace action
