#include "audio/AudioStores.h"
#include "audio/SoundVertices.h"
#include "object/PendingSync.h"
#include "state/Registry.h"
void RegisterAudioStoreHandlers(state::Registry &r) {
    r.on_destroy<SoundVertices, [](state::Registry &r, state::Entity e) {
        r.Resources.emplace<PendingObjectRemovals>().SoundVertexRanges.push_back(r.get<const SoundVertices>(e).Vertices);
    }>();
}
