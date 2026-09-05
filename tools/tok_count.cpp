#include "b70/tokenizer.hpp"
#include <cstdio>
using namespace b70;
int main(int argc, char** argv) {
    Tokenizer tk; std::string err;
    if (!tk.load(argv[1], err)) { std::printf("load: %s\n", err.c_str()); return 1; }
    for (int i = 2; i < argc; ++i)
        std::printf("%zu\t%s\n", tk.encode(argv[i]).size(), argv[i]);
    return 0;
}
