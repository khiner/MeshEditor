#include "audio/AudioStores.h"
#include "audio/SoundVertices.h"
#include "mesh/MeshStore.h"
#include <entt/entity/registry.hpp>
void RegisterAudioStoreHandlers(entt::registry &r) {
    r.on_destroy<SoundVertices>().connect<[](entt::registry &r, entt::entity e) {
        r.ctx().get<MeshStore>().ReleaseSoundVertices(r.get<SoundVertices>(e).Vertices);
    }>();
}
