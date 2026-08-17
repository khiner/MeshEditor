#pragma once

#include <boost/ut.hpp>

// Runs every suite the translation unit declared, and returns what main returns.
// Running them here keeps the code under test running while its statics are still alive, where the runner's destructor would reach library code after that library's statics are gone.
// The returned result is how the run reports its failures, the destructor setting an exit code only for a run it started itself.
inline int RunSuites() { return boost::ut::cfg<>.run(); }
