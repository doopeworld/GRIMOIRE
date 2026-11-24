#!/usr/bin/env python3
"""OpenAI-compatible front end for the Inkling 2 Mini GRIMOIRE engine (`inkling2_gpu MODEL serve SOCKET ...`).

- GET /v1/models, GET /health, POST /v1/chat/completions (stream and non-stream).
- The prompt is chat_template.jinja rendered for text messages (same strings, same order: the thinking-effort
  system line before the first non-system message, default effort 0.9; no generation prompt), encoded with the
  checkpoint's tokenizer.json (HF tokenizers, add_special_tokens=False).
- Output: byte-level BPE decoded per token; special tokens carry no text.  <|content_thinking|> .. <|end_message|>
  is streamed as `reasoning_content`, <|content_text|> .. as `content` (the checkpoint's response_template).
- One engine, one request at a time (requests queue on a lock); temperature 0 = greedy, else temperature / top-k /
  top-p sampled on the engine (exact speculative sampling on the MTP verify rows; top-k above 32 or off -> 32).
"""
import argparse
import codecs
import json
import os
import random
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--sock", required=True)
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--port", type=int, default=6911)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--name", default="inkling-2-mini")
ap.add_argument("--log", default="")
A = ap.parse_args()

TOK = Tokenizer.from_file(os.path.join(A.model_dir, "tokenizer.json"))
TJ = json.load(open(os.path.join(A.model_dir, "tokenizer.json"), encoding="utf-8"))


def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_U2B = {v: k for k, v in _b2u().items()}
ID2BYTES = {}
for _s, _i in TJ["model"]["vocab"].items():
    try:
        ID2BYTES[_i] = bytes(_U2B[c] for c in _s)
    except KeyError:
        ID2BYTES[_i] = _s.encode("utf-8")
SPECIAL = {a["id"] for a in TJ.get("added_tokens", [])}
MSG_MODEL, C_TEXT, C_THINK, END_MSG = 200001, 200004, 200008, 200010
C_CONTENT = {200004, 200022, 200057, 200049}   # text, tool error, invoke tool text, invoke tool json
ROLE = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"}
EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


def _effort_text(eff):
    if eff is None:
        eff = 0.9
    if isinstance(eff, str):
        key = eff.strip()
        if key not in EFFORT:
            raise ValueError("Unknown reasoning_effort: " + eff)
        num = EFFORT[key]
    else:
        num = float(eff)
    if num < 0.0 or num > 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return "<|message_system|><|content_text|>Thinking effort level: " + ("0" if num == 0.0 else str(num)) + "<|end_message|>"


def render(messages, effort=None):
    out = []
    emitted = False
    for m in messages:
        role = m.get("role")
        if role not in ROLE:
            raise ValueError("Unknown message role: %s" % role)
        if not emitted and role != "system":
            out.append(_effort_text(effort))
            emitted = True
        rtok = ROLE[role]
        if role in ("tool", "assistant") and m.get("name"):
            rtok += m["name"]
        rc = m.get("reasoning_content")
        if role == "assistant" and rc:
            out.append(rtok + "<|content_thinking|>" + rc + "<|end_message|>")
        c = m.get("content")
        if isinstance(c, str):
            if c or not (role == "assistant" and (m.get("tool_calls") or rc)):
                out.append(rtok + "<|content_text|>" + c + "<|end_message|>")
        elif c:
            for part in c:
                if isinstance(part, str):
                    out.append(rtok + "<|content_text|>" + part + "<|end_message|>")
                elif part.get("type") in (None, "text", "input_text"):
                    t = part.get("text") if isinstance(part.get("text"), str) else ""
                    out.append(rtok + "<|content_text|>" + t + "<|end_message|>")
                else:
                    raise ValueError("unsupported content part: %s" % part.get("type"))
        if role == "assistant" and m.get("tool_calls"):
            raise ValueError("assistant tool_calls are not supported by this front end")
        if role == "assistant":
            out.append("<|content_model_end_sampling|>")
    if not emitted:
        out.append(_effort_text(effort))
    return "".join(out)


class Detok:
    """Token ids -> (reasoning, content) text, following the response_template channels."""

    def __init__(self):
        self.chan = None
        self.hdr = False
        self.dec = {"r": codecs.getincrementaldecoder("utf-8")("replace"), "c": codecs.getincrementaldecoder("utf-8")("replace")}

    def feed(self, ids):
        r, c = [], []
        for t in ids:
            if t in SPECIAL:
                if t == MSG_MODEL:
                    self.chan, self.hdr = None, True
                elif t == C_THINK:
                    self.chan, self.hdr = "r", False
                elif t in C_CONTENT:
                    self.chan, self.hdr = "c", False
                elif t == END_MSG:
                    self.chan, self.hdr = None, False
                continue
            if self.hdr:   # speaker name between <|message_model|> and its content marker
                continue
            ch = self.chan or "c"
            txt = self.dec[ch].decode(ID2BYTES.get(t, b""), False)
            (r if ch == "r" else c).append(txt)
        return "".join(r), "".join(c)


class Engine:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.s = None
        self.f = None

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        self.s, self.f = s, s.makefile("rb")

    def run(self, rid, ids, max_new, temp, top_p, top_k, seed):
        """Yields ('tok', [ids]) per decode step, then ('end', fields) or ('err', message)."""
        if self.s is None:
            self.connect()
        line = "GEN %s %d %.6g %.6g %d %d %d %s\n" % (rid, max_new, temp, top_p, top_k, seed, len(ids), " ".join(map(str, ids)))
        self.s.sendall(line.encode())
        while True:
            raw = self.f.readline()
            if not raw:
                self.s = None
                yield ("err", "engine closed the connection")
                return
            p = raw.decode().split()
            if p[0] == "TOK" and p[1] == rid:
                yield ("tok", [int(x) for x in p[3:3 + int(p[2])]])
            elif p[0] == "END" and p[1] == rid:
                e = {"reason": p[2], "prompt": int(p[3]), "gen": int(p[4]), "prefill_ms": float(p[5]), "decode_ms": float(p[6]), "steps": int(p[7])}
                if len(p) > 8:
                    e["hist"] = [int(x) for x in p[8].split(",")]   # decode steps producing 1..8 tokens
                yield ("end", e)
                return
            elif p[0] == "ERR":
                yield ("err", raw.decode().split(" ", 2)[2].strip() if len(p) > 2 else "engine error")
                return


ENG = Engine(A.sock)
LOGF = open(A.log, "a") if A.log else None
WARNED = set()


def log(rec):
    if LOGF:
        LOGF.write(json.dumps(rec) + "\n")
        LOGF.flush()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": A.name, "object": "model", "owned_by": "grimoire", "created": 0, "max_model_len": A.ctx}]})
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            ctk = body.get("chat_template_kwargs") or {}
            effort = ctk.get("reasoning_effort", body.get("reasoning_effort"))
            text = render(body.get("messages") or [], effort)
            ids = TOK.encode(text, add_special_tokens=False).ids
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": {"message": "bad request: %s" % e, "type": "invalid_request_error"}})
            return
        if len(ids) + 9 > A.ctx:
            self._json(400, {"error": {"message": "This model's maximum context length is %d tokens. However, you requested %d tokens in the messages. Please reduce the length of the messages." % (A.ctx, len(ids)), "type": "invalid_request_error"}})
            return
        temp = body.get("temperature")
        temp = 1.0 if temp is None else float(temp)
        top_p = body.get("top_p")
        top_p = 1.0 if top_p is None else float(top_p)
        top_k = body.get("top_k")
        top_k = -1 if top_k is None else int(top_k)
        if temp > 0 and (top_k <= 0 or top_k > 32) and "topk" not in WARNED:
            WARNED.add("topk")
            print("note: top_k %d -> 32 candidates (engine keeps 32 per row)" % top_k, flush=True)
        mx = body.get("max_completion_tokens") or body.get("max_tokens") or (A.ctx - len(ids) - 9)
        seed = body.get("seed")
        seed = random.getrandbits(62) if seed is None else int(seed) & ((1 << 62) - 1)
        stream = bool(body.get("stream"))
        inc_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        det = Detok()
        t_arrive = time.perf_counter()
        gone = False

        def chunk(delta, finish=None):
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": A.name,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}

        def send(obj):
            nonlocal gone
            if gone:
                return
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
            try:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            except OSError:
                gone = True

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            send(chunk({"role": "assistant", "content": ""}))
        rtxt, ctxt = [], []
        end = None
        err = None
        with ENG.lock:
            t_start = time.perf_counter()
            for kind, val in ENG.run(rid, ids, int(mx), temp, top_p, top_k, seed):
                if kind == "tok":
                    r, c = det.feed(val)
                    if r or c:
                        if stream:
                            d = {}
                            if r:
                                d["reasoning_content"] = r
                            if c:
                                d["content"] = c
                            send(chunk(d))
                        rtxt.append(r)
                        ctxt.append(c)
                elif kind == "end":
                    end = val
                else:
                    err = val
        usage = None
        if end:
            usage = {"prompt_tokens": end["prompt"], "completion_tokens": end["gen"], "total_tokens": end["prompt"] + end["gen"]}
        log({"id": rid, "queue_ms": round((t_start - t_arrive) * 1000, 1), "end": end, "err": err, "temp": temp, "top_p": top_p, "top_k": top_k, "max": int(mx)})
        if stream:
            if err:
                send({"error": {"message": err}})
            else:
                send(chunk({}, "stop" if end["reason"] == "stop" else "length"))
                if inc_usage:
                    send({"id": rid, "object": "chat.completion.chunk", "created": created, "model": A.name, "choices": [], "usage": usage})
            send_done = b"data: [DONE]\n\n"
            if not gone:
                try:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(send_done), send_done))
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass
            return
        if err:
            self._json(500, {"error": {"message": err}})
            return
        msg = {"role": "assistant", "content": "".join(ctxt)}
        if any(rtxt):
            msg["reasoning_content"] = "".join(rtxt)
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": A.name,
                         "choices": [{"index": 0, "message": msg, "finish_reason": "stop" if end["reason"] == "stop" else "length"}],
                         "usage": usage})


def main():
    for _ in range(600):
        try:
            ENG.connect()
            break
        except OSError:
            time.sleep(1)
    else:
        print("engine socket not reachable: " + A.sock, flush=True)
        sys.exit(1)
    srv = ThreadingHTTPServer((A.host, A.port), H)
    srv.daemon_threads = True
    print("inkling server on http://%s:%d/v1 (model %s, ctx %d)" % (A.host, A.port, A.name, A.ctx), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
