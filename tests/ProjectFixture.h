#pragma once

#include "TestPaths.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/MeshCreate.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"

#include <boost/ut.hpp>

// Compares rendered images, which --no-render skips.
inline bool Render{true};

// An engine whose project lives in a scratch directory, auditing the project after each action and commit.
struct Fixture : Engine {
    Fixture() : Engine{true} {}
    explicit Fixture(const TestDir &dir) : Fixture{} {
        boost::ut::expect(P->New(dir));
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
    }
    std::vector<std::byte> Workspace() {
        return workspace::Serialize(workspace::Capture(R, R.Resources.emplace<PanelsState>()));
    }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::println("audit: {}", why);
        boost::ut::expect(valid);
        for (const auto &message : R.Resources.get<action::Errors>().Messages) std::println("error: {}", message);
        boost::ut::expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    uint32_t Do(auto a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        return node.value();
    }
    void Commit(std::string label) {
        P->Settle();
        P->History.Commit(std::move(label), {});
        Audit();
    }
    state::Entity Add(auto a) {
        Do(std::move(a));
        return FindActiveEntity(R);
    }
    state::Entity AddCube() { return Add(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()}); }
    state::Entity AddEmpty(std::string name) {
        return Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Name = std::move(name)})});
    }
    std::vector<std::byte> Image() {
        // An opened document's viewport takes the fixture's extent.
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
        SubmitViewports(R);
        WaitForRender(R);
        const auto &image = R.get<const Viewport>(FirstViewport(R)).Targets.Screen->FinalColorImage;
        return ReadbackImageRgba8(R.Resources.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};
