#pragma once
#include "audio/AudioSystem.h"
#include "audio/ModalAudio.h"
const std::vector<float> &GetSampleFrames(const state::Scene &, const std::filesystem::path &);
uint32_t GetActiveVertexIndex(const state::Scene &, state::Entity);
std::optional<std::filesystem::path> ActiveSamplePath(const state::Scene &, state::Entity);
void UpdateListenerGains(const state::Scene &, ModalBank &, state::Entity);
