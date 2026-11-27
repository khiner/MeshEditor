#include "mesh/MeshComponents.h"
#include "render/ObjectGeometry.h"
#include "scene/Entity.h"
#include "selection/SelectionState.h"

#include "state/Registry.h"
#include <boost/ut.hpp>

namespace {
const boost::ut::suite selection_tests = [] {
    using namespace boost::ut;
    "primary edit object is independent of component insertion order"_test = [] {
        std::array order{0, 1, 2};
        do {
            state::Registry r;
            r.emplace<Scene>(r.create());
            const auto mesh = r.create();
            const std::array objects{r.create(), r.create(), r.create()};
            for (const auto i : order) {
                const auto e = objects[i];
                r.emplace<ObjectGeometry>(e, mesh);
                r.emplace<Selected>(e);
                EmplaceObject(r, e, ObjectType::Mesh);
                r.emplace<RenderInstance>(e);
            }
            expect(ComputeEditPrimaries(r).All.at(mesh) == objects[0]);
            r.emplace<Active>(objects[2]);
            expect(ComputeEditPrimaries(r).All.at(mesh) == objects[2]);
            r.emplace<ScaleLocked>(objects[2]);
            const auto primaries = ComputeEditPrimaries(r);
            expect(primaries.All.at(mesh) == objects[2]);
            expect(primaries.Transformable.at(mesh) == objects[0]);
            r.remove<Selected>(objects[0]);
            expect(ComputeEditPrimaries(r).Transformable.at(mesh) == objects[1]);
        } while (std::next_permutation(order.begin(), order.end()));
    };
};
} // namespace
