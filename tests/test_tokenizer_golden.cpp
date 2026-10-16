// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/json.hpp"
#include "b70/tokenizer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("GRIMOIRE_TOKENIZER_DIR");
    const std::string dir = env && *env ? env : "/tmp/toktest";
    std::ifstream model(dir + "/tokenizer.json", std::ios::binary);
    if (!model) {
        std::printf("SKIP tokenizer HF goldens: %s/tokenizer.json absent\n", dir.c_str());
        return 0;
    }
    model.close();
    b70::Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(dir, error)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    std::ifstream f("tests/tokenizer_golden/ornith_hf_golden.json", std::ios::binary);
    if (!f) { std::fprintf(stderr, "HF golden fixture absent\n"); return 1; }
    const std::string raw((std::istreambuf_iterator<char>(f)), {});
    const b70::Json cases = b70::Json::parse(raw).at("cases");
    for (const auto& name : cases.keys) {
        const b70::Json& test = cases.at(name);
        const std::string input = test.at("text").text();
        std::vector<int32_t> expected;
        for (const b70::Json& id : test.at("ids").array) expected.push_back(id.integer());
        const std::vector<int32_t> actual = tokenizer.encode(input);
        size_t i = 0;
        while (i < expected.size() && i < actual.size() && expected[i] == actual[i]) ++i;
        if (i != expected.size() || i != actual.size()) {
            const std::vector<int32_t> prefix(expected.begin(), expected.begin() + std::min(i, expected.size()));
            const size_t byte = tokenizer.decode(prefix).size();
            const size_t left = byte > 40 ? byte - 40 : 0;
            const std::string context = input.substr(left, 80);
            std::fprintf(stderr,
                "FAIL %s: first divergence at index %zu, HF %d, GRIMOIRE %d "
                "(HF %zu tokens, GRIMOIRE %zu); text near byte %zu: %s\n",
                name.c_str(), i, i < expected.size() ? expected[i] : -1,
                i < actual.size() ? actual[i] : -1, expected.size(), actual.size(),
                byte, b70::json_dump_string(context).c_str());
            return 1;
        }
        std::printf("PASS %s: %zu HF ids identical\n", name.c_str(), actual.size());
    }
    return 0;
}
