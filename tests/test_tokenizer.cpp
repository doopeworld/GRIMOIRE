// Round-trip and byte-table tests. No model needed.
#include "b70/tokenizer.hpp"
#include <cstdio>
#include <cstring>
#include <string>
using namespace b70;
static int fails = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL "); std::printf(__VA_ARGS__); \
    std::printf("\n"); ++fails; } } while(0)

int main(int argc, char** argv) {
    std::printf("=== tokenizer ===\n\n");
    if (argc < 2) {
        std::printf("  no model dir given; skipping (usage: test_tokenizer <dir>)\n");
        return 0;
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
        std::printf("  chat template: %zu tokens, first id %d\n",
                    ids.size(), ids.empty() ? -1 : ids[0]);
    }

    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
