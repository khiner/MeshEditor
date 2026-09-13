#include "project/VectorHistory.h"
#include "render/MaterialComponents.h"

MaterialStore::MaterialStore() = default;
MaterialStore::~MaterialStore() = default;

void MaterialStore::AppendNames(std::vector<std::string> names) {
    if (Tracked) Tracked->Trie.Write(Names.size(), names.size());
    Names.insert(Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));
}

void MaterialStore::ResizeNames(size_t size) {
    if (Tracked) Tracked->Trie.Write(std::min(size, Names.size()), std::max(size, Names.size()) - std::min(size, Names.size()));
    Names.resize(size);
}

void MaterialStore::Track(store::History &history) {
    Tracked = std::make_unique<project::VectorHistory<std::string>>(Names, history, "material.names");
}
