// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/tokenizer.hpp"
#include "b70/gguf.hpp"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    const bool gguf=argc==4 && std::string(argv[1])=="--gguf";
    if (argc != 3 && !gguf) {
        std::fprintf(stderr, "usage: tok_dump TOKENIZER_DIR INPUT_FILE\n"
                             "       tok_dump --gguf GGUF_OR_HEADER_SLICE INPUT_FILE\n");
        return 2;
    }
    b70::Tokenizer tokenizer;
    std::string error;
    b70::GGUFFile file;
    if (gguf && !file.open(argv[2],error,true)) {
        std::fprintf(stderr,"GGUF load failed: %s\n",error.c_str());return 1;
    }
    if (!(gguf ? tokenizer.load_gguf(file,error) : tokenizer.load(argv[1], error))) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    std::ifstream input(argv[gguf?3:2], std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", argv[gguf?3:2]);
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    for (int32_t id : tokenizer.encode(text)) std::printf("%d\n", id);
    return 0;
}
