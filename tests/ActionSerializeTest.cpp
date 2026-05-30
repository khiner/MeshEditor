#include "Variant.h"
#include "action/Action.h"

#include <zpp_bits.h>

#include <boost/ut.hpp>

#include <cstddef>
#include <memory>
#include <vector>

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
} // namespace

int main() {
    using namespace boost::ut;

    "every action alternative round-trips"_test = [] {
        for (std::size_t i = 0; i < std::variant_size_v<Action>; ++i) {
            Action a = DefaultAt(i);
            EnsureSerializable(a);

            std::vector<std::byte> encoded;
            expect(!zpp::bits::failure(SerializeVariant(zpp::bits::out{encoded}, a)));

            Action decoded;
            expect(!zpp::bits::failure(DeserializeVariant(zpp::bits::in{encoded}, decoded)));
            expect(decoded.index() == i);

            std::vector<std::byte> reencoded;
            expect(!zpp::bits::failure(SerializeVariant(zpp::bits::out{reencoded}, decoded)));
            const bool stable = encoded == reencoded; // bool so ut doesn't try to print std::byte operands
            expect(stable);
        }
    };
}
