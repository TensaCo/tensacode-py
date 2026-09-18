"""A local teacher model, served over HTTP, for skill acquisition only.

    python -m examples.browser_agents.assistant.teacher [--model Qwen/Qwen3.5-9B] [--port 8790]

The assistant never calls a model for requests it already knows. When it meets one it
does not know, it may ask this teacher for the next step, then compiles the successful
steps into a skill it replays without the model. Runs in an environment with torch and
transformers (the ``local-model`` extra). Greedy decoding.
"""

from __future__ import annotations

import argparse
import http.server
import json
import queue
import socketserver
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--max-batch", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.perf_counter() - t0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pending: "queue.Queue[dict]" = queue.Queue()

    def generate(jobs: list[dict]) -> None:
        texts = [j["text"] for j in jobs]
        want = max(int(j["req"].get("max_new_tokens", 400)) for j in jobs)
        with torch.inference_mode():
            start = time.perf_counter()
            batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**batch, max_new_tokens=want, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            seconds = time.perf_counter() - start
        for j, row in zip(jobs, out):
            new = row[batch["input_ids"].shape[1]:]
            j["result"] = {"text": tokenizer.decode(new, skip_special_tokens=True).strip(), "model": args.model,
                           "prompt_tokens": int(batch["input_ids"].shape[1]), "new_tokens": int(len(new)), "seconds": round(seconds, 2), "batch": len(jobs)}
            j["done"].set()

    def worker() -> None:
        """Batch whatever arrives within a short window, so several runs can share the model."""
        while True:
            jobs = [pending.get()]
            deadline = time.perf_counter() + 0.05
            while len(jobs) < args.max_batch and (left := deadline - time.perf_counter()) > 0:
                try:
                    jobs.append(pending.get(timeout=left))
                except queue.Empty:
                    break
            try:
                generate(jobs)
            except Exception as exc:  # noqa: BLE001
                for j in jobs:
                    j["result"] = {"error": f"{type(exc).__name__}: {exc}"}
                    j["done"].set()

    threading.Thread(target=worker, daemon=True).start()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            self._json(200, {"model": args.model, "load_seconds": round(load_s, 1)})

        def do_POST(self) -> None:  # noqa: N802
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            messages = [{"role": "system", "content": req.get("system", "")}, {"role": "user", "content": req["user"]}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(req.get("think", False)))
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            job = {"text": text, "req": req, "done": threading.Event(), "result": None}
            pending.put(job)
            job["done"].wait()
            result = job["result"] or {"error": "no result"}
            self._json(500 if "error" in result else 200, result)

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    print(f"teacher {args.model} loaded in {load_s:.0f}s on :{args.port}", flush=True)
    Server(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
