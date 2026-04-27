#include "Scene.h"

#include "AnimationTimeline.h"
#include "Timer.h"
#include "gltf/GltfScene.h"
#include "physics/PhysicsTypes.h"
#include "physics/PhysicsWorld.h"
#include "scene_impl/SceneBuffers.h"

#include <entt/entity/registry.hpp>

std::expected<std::pair<entt::entity, entt::entity>, std::string> Scene::AddGltfScene(const std::filesystem::path &path) {
    const Timer timer{"AddGltfScene"};
    auto result = gltf::LoadGltf(path, {R, SceneEntity, *Stores.Slots, *Stores.Buffers, *Stores.Meshes, *Stores.Textures, *Stores.Environments});
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

std::expected<void, std::string> Scene::SaveGltf(const std::filesystem::path &path) {
    return gltf::SaveGltf(path, {R, SceneEntity, *Stores.Buffers, *Stores.Meshes, *Stores.Textures, &Vk, &Stores.Buffers->Ctx});
}
