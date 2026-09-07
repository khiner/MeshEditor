#include "selection/Selection.h"
#include "mesh/MeshComponents.h"
#include "render/Instance.h"
#include "scene/Entity.h"

#include <boost/ut.hpp>
#include <entt/entity/registry.hpp>

#include <algorithm>
#include <array>

namespace {
const boost::ut::suite selection_tests = [] {
    using namespace boost::ut;
    "primary edit instance is independent of component insertion order"_test = [] {
        std::array order{0, 1, 2};
        do {
            entt::registry r;
            const auto mesh = r.create();
            const std::array instances{r.create(), r.create(), r.create()};
            for (const auto i : order) {
                const auto e = instances[i];
                r.emplace<Instance>(e, mesh);
                r.emplace<Selected>(e);
                r.emplace<ObjectKind>(e, ObjectType::Mesh);
                r.emplace<RenderInstance>(e);
            }
            expect(selection::ComputePrimaryEditInstances(r).at(mesh) == instances[0]);
            r.emplace<Active>(instances[2]);
            expect(selection::ComputePrimaryEditInstances(r).at(mesh) == instances[2]);
            r.emplace<ScaleLocked>(instances[2]);
            auto primaries = selection::ComputePrimaryEditInstanceMaps(r);
            expect(primaries.All.at(mesh) == instances[2]);
            expect(primaries.Transformable.at(mesh) == instances[0]);
            expect(selection::ComputePrimaryEditInstances(r, false) == primaries.Transformable);
            r.remove<Selected>(instances[0]);
            expect(selection::ComputePrimaryEditInstances(r, false).at(mesh) == instances[1]);
        } while (std::next_permutation(order.begin(), order.end()));
    };
};
} // namespace
