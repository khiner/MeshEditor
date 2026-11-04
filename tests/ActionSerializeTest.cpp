#include "Path.h"
#include "action/LogSerialize.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioSamples.h"
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshComponents.h"
#include "mesh/TetBuffers.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRoles.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportScene.h"

#include <boost/ut.hpp>

#include <zpp_bits.h>

#include <filesystem>
#include <sstream>

using namespace action;

namespace {
std::vector<Action> AllDefaultActions() {
    std::vector<Action> all;
    [&]<size_t... Ds>(std::index_sequence<Ds...>) {
        ([&] {
            constexpr size_t D = Ds;
            using DV = std::variant_alternative_t<D, Action>;
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (all.emplace_back(std::in_place_index<D>, DV{std::in_place_index<Is>}), ...);
            }(std::make_index_sequence<std::variant_size_v<DV>>{});
        }(),
         ...);
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
    return all;
}

std::pair<size_t, size_t> IndexOf(const Action &a) {
    return {a.index(), std::visit([](const auto &dv) { return dv.index(); }, a)};
}

template<class> constexpr bool IsUniquePtr = false;
template<class T, class D> constexpr bool IsUniquePtr<std::unique_ptr<T, D>> = true;
template<class> constexpr bool IsVector = false;
template<class T, class A> constexpr bool IsVector<std::vector<T, A>> = true;
template<class> constexpr bool IsArray = false;
template<class T, size_t N> constexpr bool IsArray<std::array<T, N>> = true;
template<class> constexpr bool IsOptional = false;
template<class T> constexpr bool IsOptional<std::optional<T>> = true;
template<class> constexpr bool IsVariant = false;
template<class... Ts> constexpr bool IsVariant<std::variant<Ts...>> = true;
template<class T>
concept Components = requires(T v) { v.x; v.y; };
template<class T>
concept Matrix = requires(T v) { v.Columns; };
template<class T>
concept Map = requires { typename T::key_type; typename T::mapped_type; };
template<class T>
concept Set = requires { typename T::key_type; } && !Map<T>;

// Writes a distinct value into every member of a payload, so a round trip that drops or defaults a member shows.
// A variant takes its last alternative and a null owning pointer takes a filled value, since zpp::bits serializes neither a null pointer nor an empty choice.
struct Filler {
    uint32_t Next{1};
    static constexpr auto kind() { return zpp::bits::kind::in; }
    zpp::bits::errc operator()(auto &...values) {
        (Fill(values), ...);
        return {};
    }
    void Fill(auto &value) {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::same_as<T, bool>) value = true;
        else if constexpr (std::is_enum_v<T>) value = T(std::underlying_type_t<T>(Next++));
        else if constexpr (std::is_arithmetic_v<T>) value = T(Next++);
        else if constexpr (std::same_as<T, std::string> || std::same_as<T, std::filesystem::path>) value = std::to_string(Next++);
        else if constexpr (IsVector<T>) {
            value.resize(2);
            for (auto &element : value) Fill(element);
        } else if constexpr (Map<T>) {
            for (int i = 0; i < 2; ++i) {
                typename T::key_type key{};
                typename T::mapped_type mapped{};
                Fill(key);
                Fill(mapped);
                value.emplace(std::move(key), std::move(mapped));
            }
        } else if constexpr (Set<T>) {
            for (int i = 0; i < 2; ++i) {
                typename T::key_type key{};
                Fill(key);
                value.emplace(std::move(key));
            }
        } else if constexpr (IsArray<T>) {
            for (auto &element : value) Fill(element);
        } else if constexpr (IsOptional<T>) {
            value.emplace();
            Fill(*value);
        } else if constexpr (IsUniquePtr<T>) {
            if (!value) value = std::make_unique<typename T::element_type>();
            Fill(*value);
        } else if constexpr (IsVariant<T>) {
            value.template emplace<std::variant_size_v<T> - 1>();
            std::visit([&](auto &alternative) { Fill(alternative); }, value);
        } else if constexpr (Components<T>) {
            value.x = Next++;
            value.y = Next++;
            if constexpr (requires { value.z; }) value.z = Next++;
            if constexpr (requires { value.w; }) value.w = Next++;
        } else if constexpr (Matrix<T>) {
            for (auto &column : value.Columns) Fill(column);
        } else if constexpr (std::is_empty_v<T>) {
        } else if constexpr (requires { serialize(*this, value); }) {
            (void)serialize(*this, value);
        } else {
            zpp::bits::visit_members(value, [&](auto &...members) { (Fill(members), ...); });
        }
    }
};

template<class T> constexpr bool Fillable = [] {
    if constexpr (!requires { sizeof(T); }) return false;
    else if constexpr (!std::is_default_constructible_v<T>) return false;
    else if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::same_as<T, std::string> || std::same_as<T, std::filesystem::path>) return true;
    else if constexpr (IsVector<T> || IsArray<T> || IsOptional<T> || Set<T>) return Fillable<typename T::value_type>;
    else if constexpr (Map<T>) return Fillable<typename T::key_type> && Fillable<typename T::mapped_type>;
    else if constexpr (IsUniquePtr<T>) return Fillable<typename T::element_type>;
    else if constexpr (IsVariant<T>) return Fillable<std::variant_alternative_t<std::variant_size_v<T> - 1, T>>;
    else if constexpr (Components<T> || Matrix<T> || std::is_empty_v<T> || requires(Filler &f, T &v) { serialize(f, v); }) return true;
    else if constexpr (std::is_aggregate_v<T>) return zpp::bits::access::visit_members_types<T>([]<class... M>() { return std::bool_constant<(Fillable<std::remove_cvref_t<M>> && ...)>{}; })();
    else return false;
}();

// Restores the references a decode follows, which a generic fill leaves dangling.
void Repair(Armature &armature) {
    armature.Bones[0].ParentBoneId = InvalidBoneId;
    armature.Bones[1].ParentBoneId = armature.Bones[0].Id;
}
void Repair(auto &) {}

void Fill(Action &a) {
    Filler filler;
    std::visit([&](auto &domain) { std::visit([&](auto &leaf) { filler.Fill(leaf); }, domain); }, a);
}

std::stringstream MakeStream() { return std::stringstream{std::ios::in | std::ios::out | std::ios::binary}; }

template<state::TypeId Base, state::TypeId Count, class... Ts> void ForEachType(state::Schema<Base, Count, Ts...>, auto &&fn) { (fn.template operator()<Ts>(), ...); }
} // namespace

const boost::ut::suite ActionSerialize = [] {
    using namespace boost::ut;

    "every action round-trips through the log with every member filled"_test = [] {
        for (auto &a : AllDefaultActions()) {
            Fill(a);
            auto out = MakeStream();
            SerializeAction(a, out);
            const auto encoded = out.str();
            std::vector<Action> decoded;
            StreamActions(out, [&](Action &&d) { decoded.emplace_back(std::move(d)); });
            expect(decoded.size() == 1u) << IndexOf(a).first << IndexOf(a).second;
            if (decoded.size() != 1u) continue;
            auto again = MakeStream();
            SerializeAction(decoded[0], again);
            expect(again.str() == encoded) << IndexOf(a).first << IndexOf(a).second;
        }
    };

    "a stream reads back in order up to a limit and stops at a truncated record"_test = [] {
        auto out = MakeStream();
        std::vector<std::pair<size_t, size_t>> written;
        size_t last_record = 0;
        for (auto &a : AllDefaultActions()) {
            Fill(a);
            last_record = out.str().size();
            SerializeAction(a, out);
            written.emplace_back(IndexOf(a));
        }
        const auto all = out.str();
        const auto read = [&](std::string bytes, uint64_t limit = std::numeric_limits<uint64_t>::max()) {
            auto in = MakeStream();
            in << bytes;
            std::vector<std::pair<size_t, size_t>> indices;
            StreamActions(in, [&](Action &&d) { indices.emplace_back(IndexOf(d)); }, limit);
            return indices;
        };
        expect(read(all, 0).empty());
        expect(bool(read(all, 1) == std::vector{written.front()}));
        expect(bool(read(all) == written));
        for (size_t length = last_record; length < all.size(); ++length) expect(read(all.substr(0, length)).size() == written.size() - 1u) << length;
    };

    // Each record also appends after a prefix within reserved capacity, leaving the prefix and capacity as they were.
    "every recorded component round-trips through its snapshot entry with every member filled"_test = [] {
        constexpr size_t PrefixSize = 1024;
        constexpr auto PrefixByte = std::byte{42};
        state::Registry r;
        size_t checked = 0;
        const auto check = [&]<class C>() {
            const auto &entry = snapshot::SnapshotTable()[state::Type<C>()];
            if (entry.Kind == snapshot::Role::Derived) return;
            ++checked;
            if constexpr (!Fillable<C>) {
                expect(false) << entry.Name << "has a member the filler cannot fill";
            } else {
                C value{};
                Filler{}.Fill(value);
                Repair(value);
                const auto encode = [&](const C &v, std::vector<std::byte> &out) {
                    if (entry.Serialize) entry.Serialize(&v, out);
                };
                std::vector<std::byte> encoded;
                encode(value, encoded);
                std::vector<std::byte> bytes(PrefixSize, PrefixByte);
                bytes.reserve(PrefixSize + encoded.size());
                const auto capacity = bytes.capacity();
                encode(value, bytes);
                const auto record = std::span{bytes}.subspan(PrefixSize);
                expect(bytes.capacity() == capacity && std::ranges::all_of(std::span{bytes}.first(PrefixSize), [&](auto b) { return b == PrefixByte; }) && std::ranges::equal(record, encoded)) << entry.Name;
                const auto e = r.create();
                entry.Emplace(r, e, record);
                const auto *emplaced = r.try_get<C>(e);
                expect(emplaced != nullptr && entry.Equal(&value, emplaced)) << entry.Name;
            }
        };
        std::apply(
            [&](auto... schemas) { (ForEachType(schemas, check), ...); },
            std::tuple<action::ActionSchema, AnimationSchema, ArmatureSchema, AudioSchema, GltfSchema, GpuSchema, MeshSchema, PhysicsSchema, ProcedureSchema, RenderSchema, SceneSchema, SelectionSchema, TimelineSchema, UiSchema, ViewportSchema>{}
        );
        expect(checked == snapshot::TypesOf(snapshot::Role::Persistent).size() + snapshot::TypesOf(snapshot::Role::Workspace).size()) << "a domain schema with recorded types is missing from the list";
    };
};
