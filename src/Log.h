#pragma once

#include <format>
#include <iostream>
#include <source_location>

struct Log {
    static void Info(const std::string &message, const std::source_location &location = std::source_location::current()) {
        std::cout << std::format("[{}:{}] {}", location.file_name(), location.line(), message) << "\n";
    }
    static void Error(const std::string &message, const std::source_location &location = std::source_location::current()) {
        std::cerr << std::format("[{}:{}] {}", location.file_name(), location.line(), message) << "\n";
    }
};
