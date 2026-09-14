#include "audio/AudioStores.h"
#include "audio/SoundVertices.h"
#include "mesh/MeshStore.h"
#include "state/Scene.h"
void RegisterAudioStoreHandlers(state::Scene &r) {
    r.on_destroy<SoundVertices>().connect<[](state::Scene &r, state::Entity e) {
        r.ctx().get<MeshStore>().ReleaseSoundVertices(r.get<SoundVertices>(e).Vertices);
    }>();
}
