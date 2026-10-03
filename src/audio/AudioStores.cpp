#include "audio/AudioStores.h"
#include "audio/SoundVertices.h"
#include "object/PendingSync.h"
#include "state/Scene.h"
void RegisterAudioStoreHandlers(state::Scene &r) {
    r.on_destroy<SoundVertices, [](state::Scene &r, state::Entity e) {
        r.Context.emplace<PendingObjectRemovals>().SoundVertexRanges.push_back(r.get<const SoundVertices>(e).Vertices);
    }>();
}
