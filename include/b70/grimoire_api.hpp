#pragma once
#include "b70/formats.hpp"
#include "b70/generation.hpp"
#include <string>
namespace b70 {
struct Grimoire;
Grimoire* grimoire_new();
void grimoire_delete(Grimoire*);
bool grimoire_load(Grimoire&, const std::string&, Fmt, int, std::string&);
int grimoire_serve_generate(Grimoire&, const std::vector<int32_t>&, int, int,
    std::vector<int32_t>&, int, const std::function<bool(int32_t)>& = {}, FinishReason* = nullptr);
// Occupied prefix slots.  For gates; see the note at the definition.
int grimoire_prefix_slots_used(Grimoire&);
}
