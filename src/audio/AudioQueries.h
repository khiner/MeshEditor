#pragma once
#include "audio/AudioSystem.h"
#include "audio/ModalAudio.h"
const std::vector<float> &GetSampleFrames(const entt::registry &, const std::filesystem::path &);
uint32_t GetActiveVertexIndex(const entt::registry &, entt::entity);
std::optional<std::filesystem::path> ActiveSamplePath(const entt::registry &, entt::entity);
void UpdateListenerGains(const entt::registry &, ModalBank &, entt::entity);
