#pragma once

#include <boost/ut.hpp>

// Returns the Boost.UT status after running every suite declared by the translation unit.
// Explicit execution completes before static destruction, avoiding calls into library code after library statics are destroyed.
inline int RunSuites() { return boost::ut::cfg<>.run(); }
