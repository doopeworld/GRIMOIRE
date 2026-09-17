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
// Several conversations at once.  The answers are identical to serving
// them one at a time; what changes is how much weight traffic it took.
// Falls back to the serial path, with a reason on stderr, when the
// engine cannot batch this model or this device.
int grimoire_serve_generate_batch(Grimoire&,
    const std::vector<std::vector<int32_t>>& prompts,
    int n_predict, int eos_id,
    std::vector<std::vector<int32_t>>& out_ids, int eot_id);
}
