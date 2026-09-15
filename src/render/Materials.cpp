#include "project/store/History.h"
#include "project/store/Records.h"
#include "render/MaterialComponents.h"

MaterialStore::MaterialStore() = default;
MaterialStore::~MaterialStore() = default;

void MaterialStore::AppendNames(std::vector<std::string> names) {
    if (Tracked) Tracked->Write(Names.size(), names.size());
    Names.insert(Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));
}

void MaterialStore::ResizeNames(size_t size) {
    if (Tracked) Tracked->Write(std::min(size, Names.size()), std::max(size, Names.size()) - std::min(size, Names.size()));
    Names.resize(size);
}

void MaterialStore::Track(store::History &history) {
    Tracked = std::make_unique<store::Records>(Names);
    history.Track(*Tracked, "material.names", 0);
}
