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
// Pipeline-parallel serving: every stage except the first runs this
// instead of listening on a socket.  It waits for the front end to
// forward a request and then makes the SAME call the front end makes, so
// the per-token messages the stages exchange line up by construction.
// Returns when the front end closes the pipe or shuts down.
void grimoire_pp_worker_loop(Grimoire&);
// True when this process is a pipeline stage that is NOT the front end.
bool grimoire_is_pp_worker(Grimoire&);
// The front end tells the rest of the pipeline what it is about to run.
// Call it immediately before generating. A no-op without a pipeline, and
// on any stage that is not the front end, so callers need not ask.
bool grimoire_pp_broadcast_request(Grimoire&, const std::vector<int32_t>& prompt,
                                   int n_predict, int eos_id, int eot_id);
// Bring the workers down with the front end.
void grimoire_pp_shutdown(Grimoire&);
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
