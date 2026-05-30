#include "action/Log.h"

#include <chrono>
#include <filesystem>
#include <string>

namespace action {
std::ofstream OpenLogStream() {
    std::filesystem::create_directories("wbl");
    const auto unix_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return std::ofstream{std::filesystem::path{"wbl"} / (std::to_string(unix_sec) + ".wbl"), std::ios::binary | std::ios::app};
}
} // namespace action
