#pragma once

#include "materialx/Library.h"

namespace mx {
// The Metal source of the given definitions and the definitions they use, in library order, each under its own include guard.
// It holds `declarations` first, which the definitions may use.
std::string MetalSource(const Library &, std::span<const uint32_t> definitions, std::string_view declarations = {});

// A Metal statement that evaluates the implementation on argument expressions given in declaration input order.
// It assigns the outputs to the variables in `results`, in declaration output order.
std::string Call(const Declaration &, const InlineExpression &, std::span<const std::string> arguments, std::span<const std::string> results);
std::string Call(const Declaration &, const FunctionCall &, std::span<const std::string> arguments, std::span<const std::string> results);
} // namespace mx
