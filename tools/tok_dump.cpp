// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/tokenizer.hpp"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tok_dump TOKENIZER_DIR INPUT_FILE\n");
        return 2;
    }
    b70::Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(argv[1], error)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    for (int32_t id : tokenizer.encode(text)) std::printf("%d\n", id);
    return 0;
}
