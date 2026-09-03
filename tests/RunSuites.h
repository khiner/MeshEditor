#pragma once

#include <boost/ut.hpp>

inline int RunSuites() { return boost::ut::cfg<>.run(); }
