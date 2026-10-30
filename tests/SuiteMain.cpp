#include "RunSuites.h"

// Runs every suite the executable links, filtered by the test name pattern given as the first argument.
int main(int argc, const char **argv) {
    if (argc > 1) boost::ut::cfg<boost::ut::override> = {.filter = argv[1]};
    return RunSuites();
}
