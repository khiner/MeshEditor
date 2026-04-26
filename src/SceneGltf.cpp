#include "Scene.h"

#include "AnimationTimeline.h"
#include "Timer.h"
#include "gltf/EcsScene.h"
#include "physics/PhysicsTypes.h"
#include "physics/PhysicsWorld.h"

#include <entt/entity/registry.hpp>

std::expected<std::pair<entt::entity, entt::entity>, std::string> Scene::AddGltfScene(const std::filesystem::path &path) {
    const Timer timer{"AddGltfScene"};
    gltf::PopulateContext ctx{
        .R = R,
        .SceneEntity = SceneEntity,
        .Slots = *Stores.Slots,
        .Buffers = *Stores.Buffers,
        .Meshes = *Stores.Meshes,
        .Textures = *Stores.Textures,
        .Environments = *Stores.Environments,
    };
    auto result = gltf::LoadGltfFile(path, ctx);
    if (!result) return std::unexpected{std::move(result.error())};

    // TODO: drive RecomputeSceneScale reactively from track<changes::PhysicsShape> and drop this imperative call.
    if (!R.view<ColliderShape>().empty()) Physics->RecomputeSceneScale(R);

    if (result->FirstCameraObject != entt::null) {
        LookThrough = LookThroughState{result->FirstCameraObject, R.get<ViewCamera>(SceneEntity)};
        SnapToCamera(result->FirstCameraObject);
    }
    if (result->ImportedAnimation) {
        ApplyTimelineAction(timeline_action::JumpToStart{});
        LastEvaluatedFrame = -1;
    }
    ProfileNextProcessComponentEvents = true;
    return std::pair{result->FirstMesh, result->Active};
}
