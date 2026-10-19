#pragma once
#include "audio/AudioSystem.h"
#include "audio/ModalAudio.h"
const std::vector<float> &GetSampleFrames(const state::Registry &, const std::filesystem::path &);
uint32_t GetActiveVertexIndex(const state::Registry &, state::Entity);
std::optional<std::filesystem::path> ActiveSamplePath(const state::Registry &, state::Entity);
void UpdateListenerGains(const state::Registry &, ModalBank &);
