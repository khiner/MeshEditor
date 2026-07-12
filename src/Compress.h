#pragma once

#include <filesystem>

bool Compress(const std::filesystem::path &src, const std::filesystem::path &dst);
bool Decompress(const std::filesystem::path &src, const std::filesystem::path &dst);
