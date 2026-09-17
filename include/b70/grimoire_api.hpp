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

// A resident batching scheduler: one thread owns the engine, request
// threads hand it prompts and are handed tokens back as they are
// produced.  Requests that overlap in time are STEPPED TOGETHER instead
// of queued behind each other, which for a decode step -- where the cost
// is reading the weights, not the one token -- is most of the bill.
//
// It degrades to one request at a time, by the ordinary path and with
// speculation intact, whenever the engine cannot batch the model or the
// device.  Callers do not have to ask which: they submit either way.
struct GrimoireScheduler;
GrimoireScheduler* grimoire_scheduler_new(Grimoire&, int width);
void grimoire_scheduler_delete(GrimoireScheduler*);
// How many requests it will step together.  One means it could not batch.
int grimoire_scheduler_width(GrimoireScheduler&);
// Blocking, and safe to call from several threads at once.  on_token is
// called on the CALLING thread for each token; returning false from it
// cancels the request and releases its slot.
int grimoire_scheduler_generate(GrimoireScheduler&,
    const std::vector<int32_t>& prompt, int n_predict, int eos_id, int eot_id,
    std::vector<int32_t>& out, const std::function<bool(int32_t)>& on_token = {},
    FinishReason* finish = nullptr);
}
