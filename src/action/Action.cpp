#include "action/Action.h"
#include "action/Emit.h"
#include "action/Log.h"

#include <zpp_bits.h>

using namespace action;

namespace {
std::optional<Action> Staged;

std::optional<std::ofstream> LogStream;
std::optional<WriteBehindLog<Action>> Log;

// Serialize action into a reused buffer and append its bytes to the log stream.
// Runs only on the writer thread.
void SerializeAction(const Action &a, std::ostream &out) {
    static thread_local std::vector<std::byte> buffer;
    zpp::bits::out archive{buffer};
    if (zpp::bits::failure(SerializeVariant(archive, a))) return; // e.g. a null owning pointer; drop rather than write a partial record
    out.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(archive.position()));
}
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) {
    if (!Staged) Staged.emplace(std::move(a)); // First action emitted in the frame wins.
}

void StartLog() {
    LogStream.emplace(OpenLogStream());
    Log.emplace(*LogStream, &SerializeAction);
}
void StopLog() {
    if (Log) Log->Stop();
    Log.reset();
    LogStream.reset();
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    if (!Staged) return;

    // Apply the action, then record it to the log.
    auto recorded = std::visit(
        [&]<typename T>(T &&a) -> Action {
            using L = std::decay_t<T>;
            auto apply_keep = [&]<typename DomainV>() -> Action {
                DomainV dv{std::move(a)};
                Apply(r, viewport, dv);
                return Action{std::in_place_type<L>, std::move(std::get<L>(dv))};
            };
            if constexpr (is_variant_member_v<L, selection::Action>) return apply_keep.template operator()<selection::Action>();
            else if constexpr (is_variant_member_v<L, object::Action>) return apply_keep.template operator()<object::Action>();
            else if constexpr (is_variant_member_v<L, view::Action>) return apply_keep.template operator()<view::Action>();
            else if constexpr (is_variant_member_v<L, action::physics::Action>) return apply_keep.template operator()<action::physics::Action>();
            else if constexpr (is_variant_member_v<L, audio::Action>) return apply_keep.template operator()<audio::Action>();
            else if constexpr (is_variant_member_v<L, bone::Action>) return apply_keep.template operator()<bone::Action>();
            else if constexpr (is_variant_member_v<L, timeline::Action>) return apply_keep.template operator()<timeline::Action>();
            else if constexpr (is_variant_member_v<L, io::Action>) return apply_keep.template operator()<io::Action>();
            else return apply_keep.template operator()<Core>();
        },
        std::move(*Staged)
    );
    if (Log) Log->Enqueue(std::move(recorded));
    Staged.reset();
}

std::size_t ActionSize() { return sizeof(Action); }
} // namespace action

namespace {
// Instantiate Emit for all action types
using EmitPtr = void (*)();
template<std::size_t... I>
std::array<EmitPtr, sizeof...(I)> AllEmits(std::index_sequence<I...>) {
    return {reinterpret_cast<EmitPtr>(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&Emit))...};
}
const auto _ = AllEmits(std::make_index_sequence<std::variant_size_v<Action>>{});
} // namespace
