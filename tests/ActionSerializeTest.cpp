#include "Variant.h"
#include "action/Action.h"
#include "action/LogSerialize.h"

#include <boost/ut.hpp>

#include <sstream>

using namespace action;

namespace {
// A merged Action holding alternative `idx`, default-constructed.
// (Can't use CreateVariantByIndex - it builds a constexpr table and copies, but Action is move-only bc of unique_ptrs)
Action DefaultAt(std::size_t idx) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> Action {
        Action a;
        ((Is == idx ? (void)a.emplace<Is>() : void()), ...);
        return a;
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
}

// Owning-pointer alternatives default to a null pointer, which zpp::bits won't serialize.
// Give them a payload - an enqueued action is never null.
void EnsureSerializable(Action &a) {
    std::visit(
        overloaded{
            [](object::AddEmpty &x) { x.Info = std::make_unique<ObjectCreateInfo>(); },
            [](object::AddArmature &x) { x.Info = std::make_unique<ObjectCreateInfo>(); },
            [](object::AddCamera &x) { x.Info = std::make_unique<ObjectCreateInfo>(); },
            [](object::AddLight &x) { x.Info = std::make_unique<ObjectCreateInfo>(); },
            [](object::AddMeshPrimitive &x) { x.Info = std::make_unique<MeshInstanceCreateInfo>(); },
            [](object::ImportMesh &x) { x.Info = std::make_unique<MeshInstanceCreateInfo>(); },
            [](view::DragGizmo &x) { x.Value = std::make_unique<PendingTransform>(); },
            [](view::DragGizmoMeshEdit &x) { x.Value = std::make_unique<PendingTransform>(); },
            [](action::physics::SetJointVecItem<PhysicsJointLimit> &x) { x.Value = std::make_unique<PhysicsJointLimit>(); },
            [](action::physics::SetJointVecItem<PhysicsJointDrive> &x) { x.Value = std::make_unique<PhysicsJointDrive>(); },
            [](Replace<PhysicsMotion> &x) { x.Value = std::make_unique<PhysicsMotion>(); },
            [](ReplaceActive<PhysicsMotion> &x) { x.Value = std::make_unique<PhysicsMotion>(); },
            [](audio::OpenModalForm &x) { x.Info = std::make_unique<ModalModelCreateInfo>(); },
            [](audio::SetModalFormMaterial &x) { x.Material = std::make_unique<AcousticMaterial>(); },
            [](auto &) {},
        },
        a
    );
}

std::stringstream MakeStream() { return std::stringstream{std::ios::in | std::ios::out | std::ios::binary}; }
} // namespace

int main() {
    using namespace boost::ut;

    "every action round-trips through the log"_test = [] {
        for (std::size_t i = 0; i < std::variant_size_v<Action>; ++i) {
            Action a = DefaultAt(i);
            EnsureSerializable(a);

            auto out = MakeStream();
            SerializeAction(a, out);
            const auto encoded = out.str();

            std::vector<Action> decoded;
            StreamActions(out, [&](Action &&d) { decoded.push_back(std::move(d)); });
            expect(decoded.size() == 1u);
            if (decoded.size() != 1u) continue;
            expect(decoded[0].index() == i);

            // Re-encoding the decoded action reproduces the same bytes.
            auto reout = MakeStream();
            SerializeAction(decoded[0], reout);
            const bool stable = reout.str() == encoded;
            expect(stable);
        }
    };

    "back-to-back records read back in order"_test = [] {
        auto out = MakeStream();
        std::vector<std::size_t> written;
        for (std::size_t i = 0; i < std::variant_size_v<Action>; ++i) {
            Action a = DefaultAt(i);
            EnsureSerializable(a);
            SerializeAction(a, out);
            written.push_back(i);
        }
        std::vector<std::size_t> read;
        StreamActions(out, [&](Action &&d) { read.push_back(d.index()); });
        const bool same = read == written;
        expect(same);
    };
}
