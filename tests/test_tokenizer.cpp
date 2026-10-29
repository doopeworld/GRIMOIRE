// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Round-trip and byte-table tests. No model needed.
#include "b70/tokenizer.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
using namespace b70;
static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL "); std::printf(__VA_ARGS__); \
    std::printf("\n"); ++fails; } } while(0)

int main(int argc, char** argv) {
    std::printf("=== tokenizer ===\n\n");
    {
        char dir[] = "/tmp/grimoire-tokenizer-XXXXXX";
        CHECK(::mkdtemp(dir) != nullptr, "cannot create tokenizer fixture");
        const std::string root(dir);
        { std::ofstream f(root + "/tokenizer.json");
          f << R"({"model":{"vocab":{"a":0},"merges":[]},"added_tokens":[{"id":1,"content":"<s>","special":true}]})"; }
        { std::ofstream f(root + "/tokenizer_config.json");
          f << R"({"add_bos_token":true,"bos_token":"<s>"})"; }
        Tokenizer yes; std::string err;
        CHECK(yes.load(root,err) && yes.add_bos_token() && yes.bos()==1,
              "configured BOS was not resolved: %s", err.c_str());
        CHECK(yes.encode("a") == std::vector<int32_t>{0},
              "raw encode unexpectedly inserted BOS");
        { std::ofstream f(root + "/tokenizer_config.json");
          f << R"({"add_bos_token":false,"bos_token":"<s>"})"; }
        Tokenizer no; err.clear();
        CHECK(no.load(root,err) && !no.add_bos_token(),
              "configured no-BOS was ignored: %s", err.c_str());
        ::unlink((root + "/tokenizer.json").c_str());
        ::unlink((root + "/tokenizer_config.json").c_str());
        ::rmdir(root.c_str());
    }
    if (argc < 2) {
        std::printf("  no model dir given; skipping (usage: test_tokenizer <dir>)\n");
        return fails ? 1 : 0;
    }
    Tokenizer tk;
    std::string err;
    if (!tk.load(argv[1], err)) { std::printf("  LOAD FAILED: %s\n", err.c_str()); return 1; }
    std::printf("  vocab %zu  bos %d  eos %d\n", tk.vocab_size(), tk.bos(), tk.eos());

    const char* cases[] = {
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
        "  multiple   spaces  ",
        "unicode: cafe\xc3\xa9 \xe4\xb8\xad\xe6\x96\x87 \xf0\x9f\x98\x80",
        "1234567890",
        "",
    };
    for (const char* c : cases) {
        auto ids = tk.encode(c);
        std::string back = tk.decode(ids);
        const bool ok = (back == c);
        std::printf("  %-46s %3zu tok  %s\n",
                    (std::string("\"") + c).substr(0, 46).c_str(), ids.size(),
                    ok ? "roundtrip ok" : "ROUNDTRIP MISMATCH");
        if (!ok) {
            std::printf("      in : %s\n      out: %s\n", c, back.c_str());
            ++fails;
        }
    }

    // Byte-level completeness: every byte value must survive a round trip.
    {
        std::string all;
        for (int b = 1; b < 256; ++b) all += char(b);
        auto ids = tk.encode(all);
        std::string back = tk.decode(ids);
        CHECK(back == all, "byte-level roundtrip lost data (%zu -> %zu bytes)",
              all.size(), back.size());
        std::printf("  all 255 byte values roundtrip: %s\n",
                    back == all ? "ok" : "FAILED");
    }

    // Chat template must produce the ChatML control tokens.
    {
        const std::string p = tk.apply_chat_template("Hi");
        auto ids = tk.encode(p);
        CHECK(!ids.empty(), "chat template encoded to nothing");
        const int32_t im_start = tk.special_id("<|im_start|>");
        if (im_start >= 0) {
            CHECK(!ids.empty() && ids.front() == im_start,
                  "chat template did not start with <|im_start|>");
            CHECK(tk.encode("<|im_start|>") == std::vector<int32_t>{im_start},
                  "special token was split by the pre-tokenizer");
        }
        std::printf("  chat template: %zu tokens, first id %d\n",
                    ids.size(), ids.empty() ? -1 : ids[0]);
    }

    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
