# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow>=15", "huggingface_hub>=0.24"]
# ///
"""Build the frozen held-out prompt sets from public datasets.

Run:      uv run eval/heldout/build.py            (download, sample, write, update MANIFEST)
Verify:   uv run eval/heldout/build.py --verify   (recompute hashes, compare to MANIFEST)

Every source is pinned to a git/Hub revision, and every random choice uses
``random.Random(f"{SEED}:...")`` (string seeds hash deterministically), so a re-run
reproduces byte-identical split files. The data is written to the user cache and
is never committed; only this script, MANIFEST.json and README.md live in the repo.

This script was written without reading the agent's code (src/, examples/, tests/).
It prints only counts and hashes, never items, so running it does not expose the
test split.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

SEED = 20260918
BUILD_DATE = "2026-09-18"
PER_CATEGORY = 60
SPLITS = (("dev", 0.2), ("calibration", 0.2), ("test", 0.6))

CACHE = Path(os.environ.get("TENSORCODE_HELDOUT_DIR", Path.home() / ".cache" / "tensorcode" / "heldout"))
RAW = CACHE / "raw"
IMAGES = CACHE / "images"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"

# ---------------------------------------------------------------- pinned sources
SOURCES = {
    "osworld": {
        "name": "OSWorld evaluation_examples (Ubuntu tasks)",
        "url": "https://github.com/xlang-ai/OSWorld",
        "revision": "b138d348256078fa634fc3b73567a7337c793e6b",
        "license": "Apache-2.0",
    },
    "nl2bash": {
        "name": "NL2Bash (data/bash/all.nl + all.cm)",
        "url": "https://github.com/TellinaTool/nl2bash",
        "revision": "d6b9f5bdff45621d190134e31ab63b7bf7002190",
        "license": "GPL-3.0 (repository license); not redistributed here",
    },
    "screenqa": {
        "name": "ScreenQA-Short (RICO screenshots), HF mirror bevaya/RICO-ScreenQA-Short, split=test",
        "url": "https://huggingface.co/datasets/bevaya/RICO-ScreenQA-Short (original: https://github.com/google-research-datasets/screen_qa)",
        "revision": "d432b8e9e191447b4d04d99e2740838d7319dac5",
        "license": "CC-BY-4.0 (ScreenQA annotations); RICO screenshots under RICO terms",
    },
    "vqav2": {
        "name": "VQA v2, HF mirror lmms-lab-encoder/VQAv2, split=validation",
        "url": "https://huggingface.co/datasets/lmms-lab-encoder/VQAv2 (original: https://visualqa.org)",
        "revision": "32665d35052eb4a6d4414851c3c829a72754915a",
        "license": "CC-BY-4.0 (annotations); COCO images under Flickr terms of use",
    },
    "longmemeval": {
        "name": "LongMemEval oracle (evidence sessions only), HF xiaowu0162/longmemeval-cleaned",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (paper repo: https://github.com/xiaowu0162/LongMemEval)",
        "revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "license": "MIT",
    },
    "nq_open": {
        "name": "Natural Questions open, HF google-research-datasets/nq_open, split=validation",
        "url": "https://huggingface.co/datasets/google-research-datasets/nq_open",
        "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28",
        "license": "CC-BY-SA-3.0",
    },
    "web_questions": {
        "name": "WebQuestions, HF stanfordnlp/web_questions, split=test",
        "url": "https://huggingface.co/datasets/stanfordnlp/web_questions",
        "revision": "0e473cbe21d1e91ec18da343644498be6a3f5454",
        "license": "CC-BY-4.0 per original release (HF card says 'unknown')",
    },
    "gsm8k": {
        "name": "GSM8K main, HF openai/gsm8k, split=test",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
    },
    "dolly": {
        "name": "databricks-dolly-15k",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "CC-BY-SA-3.0",
    },
    "ambignq": {
        "name": "AmbigNQ light, HF sewon/ambig_qa, split=validation (multipleQAs only)",
        "url": "https://huggingface.co/datasets/sewon/ambig_qa",
        "revision": "e969d0132f4dd28c2939d55be34f1788c00ccfe7",
        "license": "CC-BY-SA-3.0",
    },
    "clariq": {
        "name": "ClariQ train+dev topics with clarification_need >= 3",
        "url": "https://github.com/aliannejadi/ClariQ",
        "revision": "46885a544581a0af8aff0681d29e4971807e2912",
        "license": "no explicit license in repository (research data; TREC Web Track topics); not redistributed here",
    },
    "mind2web": {
        "name": "Mind2Web test_task/test_website/test_domain, HF osunlp/Multimodal-Mind2Web",
        "url": "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web",
        "revision": "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
        "license": "OpenRAIL (HF card of the multimodal mirror); original Mind2Web annotations CC-BY-4.0",
    },
    "owner": {
        "name": "Owner's own prompt (verbatim), docs/revival/28-general-agent.md",
        "url": "n/a",
        "revision": "n/a",
        "license": "owner-provided",
    },
}

OWNER_PROMPT = (
    "you are designing an autonomous factory that must manufacture arbitrary small electromechanical devices from raw stock.\n\n"
    "available processes initially include cnc milling, turning, laser cutting, fdm printing, pcb milling, pick-and-place, "
    "solder reflow, robotic assembly, and inspection.\n\n"
    "the factory receives a target artifact only as a functional specification, e.g.:\n\n"
    "\"build a self-contained device under 250 g that can travel 100 m across uneven indoor terrain, carry a 50 g payload, "
    "autonomously navigate around obstacles, and cost under $40 in consumed materials.\"\n\n"
    "design:\n\n"
    "the artifact itself\n"
    "its mechanical/electrical/software architecture\n"
    "a manufacturable bom\n"
    "the complete process graph converting raw materials into the artifact\n"
    "inspection/evaluation procedures\n"
    "recovery strategies when fabrication steps fail\n"
    "a representation that would let the same planning system generalize to radically different artifacts\n\n"
    "do not prematurely optimize one design. explicitly identify the major uncertainties and tell me what information "
    "or experiment you would request next."
)


# ---------------------------------------------------------------- helpers
def rng(*parts) -> random.Random:
    return random.Random(":".join([str(SEED), *map(str, parts)]))


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "tensorcode-heldout-build"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
    return dest


def gh_raw(repo: str, rev: str, path: str) -> Path:
    return fetch(f"https://raw.githubusercontent.com/{repo}/{rev}/{path}", RAW / repo.replace("/", "__") / rev / path)


def hf_file(repo: str, rev: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev, cache_dir=str(RAW / "hf")))


def hf_parquet_files(repo: str, rev: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def hf_open(repo: str, rev: str, path: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    return pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}@{rev}/{path}", block_size=2**20))


def hf_read_rows(repo: str, rev: str, path: str, rows: list[int], columns: list[str]) -> dict[int, dict]:
    """Read selected global row indices of one parquet file, touching only their row groups."""
    pf = hf_open(repo, rev, path)
    out, start = {}, 0
    want = sorted(rows)
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        local = [r - start for r in want if start <= r < start + n]
        if local:
            tbl = pf.read_row_group(g, columns=columns)
            for i in local:
                out[start + i] = tbl.slice(i, 1).to_pylist()[0]
        start += n
    return out


def hf_scan(repo: str, rev: str, path: str, columns: list[str]) -> list[dict]:
    return hf_open(repo, rev, path).read(columns=columns).to_pylist()


def hf_scan_many(repo: str, rev: str, paths: list[str], columns: list[str]) -> list[list[dict]]:
    """hf_scan over several files in parallel (latency-bound); results keep the order of `paths`."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda p: hf_scan(repo, rev, p, columns), paths))


def image_ext(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def save_image(category: str, item_id: str, data: bytes) -> str:
    rel = Path("images") / category / (item_id + image_ext(data))
    dest = CACHE / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return rel.as_posix()


def stratified(cands: list[dict], n: int, key, rng_: random.Random, mode: str = "equal") -> list[dict]:
    """Stratified sample without replacement.

    mode='equal': n split evenly across strata (capped by stratum size; the
    leftover is redistributed evenly). mode='proportional': largest-remainder
    allocation proportional to stratum size. Strata are visited in sorted order
    so the result is deterministic.
    """
    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        strata[str(key(c))].append(c)
    names = sorted(strata)
    for s in names:
        strata[s].sort(key=lambda c: str(c["source_id"]))
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    alloc = dict.fromkeys(names, 0)
    if mode == "proportional":
        quotas = {s: n * len(strata[s]) / total for s in names}
        alloc = {s: int(quotas[s]) for s in names}
        rest = n - sum(alloc.values())
        for s in sorted(names, key=lambda s: (-(quotas[s] - alloc[s]), s))[:rest]:
            alloc[s] += 1
    else:
        remaining = n
        open_ = list(names)
        while remaining and open_:
            share, extra = divmod(remaining, len(open_))
            order = sorted(open_)
            rng_.shuffle(order)  # who gets the +1 remainder is random but seeded
            for i, s in enumerate(order):
                alloc[s] += share + (1 if i < extra else 0)
            over = 0
            for s in list(open_):
                if alloc[s] >= len(strata[s]):
                    over += alloc[s] - len(strata[s])
                    alloc[s] = len(strata[s])
                    open_.remove(s)
            remaining = over
    picked = []
    for s in names:
        picked.extend(rng_.sample(strata[s], alloc[s]))
    return picked


def item_id(category: str, source: str, source_id) -> str:
    h = hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()[:10]
    return f"{category}-{h}"


def rec(category, source, source_id, text, reference=None, image=None, notes="", **extra) -> dict:
    d = {
        "id": item_id(category, source, source_id),
        "category": category,
        "source": source,
        "source_id": str(source_id),
        "license": SOURCES[source]["license"],
        "text": text,
        "image": image,
        "reference": reference,
        "notes": notes,
    }
    d.update(extra)
    return d


# ---------------------------------------------------------------- OSWorld
def osworld_index() -> dict[str, list[str]]:
    rev = SOURCES["osworld"]["revision"]
    p = gh_raw("xlang-ai/OSWorld", rev, "evaluation_examples/test_all.json")
    return json.loads(p.read_text())


def osworld_task(domain: str, tid: str) -> dict:
    rev = SOURCES["osworld"]["revision"]
    p = gh_raw("xlang-ai/OSWorld", rev, f"evaluation_examples/examples/{domain}/{tid}.json")
    return json.loads(p.read_text())


def osworld_record(category: str, domain: str, tid: str, notes: str) -> dict:
    t = osworld_task(domain, tid)
    ref = {k: t.get(k) for k in ("evaluator", "config", "related_apps", "snapshot") if k in t}
    ref["domain"] = domain
    return rec(category, "osworld", f"{domain}/{tid}", t["instruction"], reference=ref, notes=notes)


def build_desktop_gui() -> list[dict]:
    idx = osworld_index()
    cands = [{"source_id": f"{d}/{t}", "d": d, "t": t} for d, ts in idx.items() if d != "multi_apps" for t in ts]
    picked = stratified(cands, PER_CATEGORY, lambda c: c["d"], rng("desktop_gui"), "equal")
    return [
        osworld_record(
            "desktop_gui",
            c["d"],
            c["t"],
            "OSWorld single-app Ubuntu task; stratified equally across app domains. reference.evaluator is OSWorld's "
            "own checker (needs the real VM/files); evaluator.func == 'infeasible' means the correct behaviour is to "
            "report the task cannot be done.",
        )
        for c in picked
    ]


# ---------------------------------------------------------------- NL2Bash
def build_shell_files() -> list[dict]:
    rev = SOURCES["nl2bash"]["revision"]
    nl = gh_raw("TellinaTool/nl2bash", rev, "data/bash/all.nl").read_text(encoding="utf-8").splitlines()
    cm = gh_raw("TellinaTool/nl2bash", rev, "data/bash/all.cm").read_text(encoding="utf-8").splitlines()
    assert len(nl) == len(cm), (len(nl), len(cm))
    seen, cands = set(), []
    for i, (q, c) in enumerate(zip(nl, cm)):
        q, c = q.strip(), c.strip()
        if not q or not c or q.lower() in seen:
            continue
        seen.add(q.lower())
        toks = [t for t in c.split() if "=" not in t and t not in ("sudo", "env", "nohup", "time")]
        util = toks[0].split("/")[-1] if toks else "?"
        cands.append({"source_id": f"line{i + 1}", "q": q, "c": c, "util": util})
    picked = stratified(cands, PER_CATEGORY, lambda c: c["util"], rng("shell_files"), "proportional")
    return [
        rec(
            "shell_files",
            "nl2bash",
            c["source_id"],
            c["q"],
            reference={"bash": c["c"], "head_utility": c["util"]},
            notes="NL2Bash description (1-based line in all.nl/all.cm); stratified proportionally by the command's "
            "head utility. The gold command is one valid answer, not the only one; grade by resulting world state.",
        )
        for c in picked
    ]


# ---------------------------------------------------------------- ScreenQA
def build_screen_questions() -> list[dict]:
    src = SOURCES["screenqa"]
    repo, rev = "bevaya/RICO-ScreenQA-Short", src["revision"]
    files = hf_parquet_files(repo, rev, "data/test-")
    cands = []
    for f, rows in zip(files, hf_scan_many(repo, rev, files, ["screen_id", "question"])):
        for i, r in enumerate(rows):
            cands.append({"source_id": f"{f.split('/')[-1]}#{i}", "f": f, "row": i, "screen": r["screen_id"]})
    # one question per screen, so 60 items cover 60 different screens
    by_screen = {}
    for c in cands:
        by_screen.setdefault(c["screen"], []).append(c)
    r_ = rng("screen_questions")
    one_per = [r_.choice(sorted(v, key=lambda c: c["row"])) for _, v in sorted(by_screen.items())]
    picked = stratified(one_per, PER_CATEGORY, lambda c: "all", r_, "equal")
    out = []
    for f in sorted({c["f"] for c in picked}):
        rows = [c for c in picked if c["f"] == f]
        got = hf_read_rows(repo, rev, f, [c["row"] for c in rows], ["screen_id", "question", "ground_truth", "file_name", "image"])
        for c in rows:
            r = got[c["row"]]
            iid = item_id("screen_questions", "screenqa", c["source_id"])
            img = save_image("screen_questions", iid, r["image"]["bytes"])
            out.append(
                rec(
                    "screen_questions",
                    "screenqa",
                    c["source_id"],
                    r["question"],
                    reference={"answers": r["ground_truth"], "screen_id": r["screen_id"], "rico_file": r["file_name"]},
                    image=img,
                    notes="ScreenQA-Short question about an Android (RICO) screenshot; one question per screen; "
                    "reference.answers lists acceptable short answers (empty list = not answerable from the screen).",
                )
            )
    return out


# ---------------------------------------------------------------- VQA v2
def build_image_questions() -> list[dict]:
    repo, rev = "lmms-lab-encoder/VQAv2", SOURCES["vqav2"]["revision"]
    files = hf_parquet_files(repo, rev, "data/validation-")
    cands = []
    for f, rows in zip(files, hf_scan_many(repo, rev, files, ["question_id", "image_id", "answer_type"])):
        for i, r in enumerate(rows):
            cands.append({"source_id": str(r["question_id"]), "f": f, "row": i, "img": r["image_id"], "at": r["answer_type"]})
    by_img = {}
    for c in cands:
        by_img.setdefault(c["img"], []).append(c)
    r_ = rng("image_questions")
    one_per = [r_.choice(sorted(v, key=lambda c: c["source_id"])) for _, v in sorted(by_img.items())]
    picked = stratified(one_per, PER_CATEGORY, lambda c: c["at"], r_, "equal")
    out = []
    for f in sorted({c["f"] for c in picked}):
        rows = [c for c in picked if c["f"] == f]
        got = hf_read_rows(
            repo, rev, f, [c["row"] for c in rows],
            ["question_id", "image_id", "question", "answer_type", "question_type", "multiple_choice_answer", "answers", "image"],
        )
        for c in rows:
            r = got[c["row"]]
            iid = item_id("image_questions", "vqav2", c["source_id"])
            img = save_image("image_questions", iid, r["image"]["bytes"])
            out.append(
                rec(
                    "image_questions",
                    "vqav2",
                    c["source_id"],
                    r["question"],
                    reference={
                        "answer": r["multiple_choice_answer"],
                        "answers": [a["answer"] for a in r["answers"]],
                        "answer_type": r["answer_type"],
                        "question_type": r["question_type"],
                        "coco_image_id": r["image_id"],
                    },
                    image=img,
                    notes="VQA v2 val question on a COCO photo; one question per image; stratified equally by "
                    "answer_type (yes/no, number, other). Standard VQA accuracy: min(#humans agreeing / 3, 1).",
                )
            )
    return out


# ---------------------------------------------------------------- LongMemEval
LME_TYPES = ("single-session-user", "single-session-preference", "multi-session", "knowledge-update", "temporal-reasoning")


def build_conversation_facts() -> list[dict]:
    p = hf_file("xiaowu0162/longmemeval-cleaned", SOURCES["longmemeval"]["revision"], "longmemeval_oracle.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    cands = [
        {"source_id": d["question_id"], "d": d}
        for d in data
        if d["question_type"] in LME_TYPES
    ]
    picked = stratified(cands, PER_CATEGORY, lambda c: c["d"]["question_type"], rng("conversation_facts"), "equal")
    out = []
    for c in picked:
        d = c["d"]
        history, evidence = [], []
        for s_i, (date, sess) in enumerate(zip(d["haystack_dates"], d["haystack_sessions"])):
            msgs = []
            for m_i, m in enumerate(sess):
                msgs.append({"role": m["role"], "content": m["content"]})
                if m.get("has_answer"):
                    evidence.append([s_i, m_i])
            history.append({"date": date, "messages": msgs})
        out.append(
            rec(
                "conversation_facts",
                "longmemeval",
                d["question_id"],
                d["question"],
                reference={
                    "answer": d["answer"],
                    "question_type": d["question_type"],
                    "question_date": d.get("question_date"),
                    "evidence_turns": evidence,
                    "abstention": d["question_id"].endswith("_abs"),
                },
                notes="LongMemEval oracle item: replay `history` (earlier chat sessions; the user turns state facts "
                "about themselves) before sending `text`. Assistant turns in history come from the dataset. Items "
                "whose id ends in _abs are unanswerable (correct reply says the fact was never given).",
                history=history,
            )
        )
    return out


# ---------------------------------------------------------------- general knowledge
def build_general_knowledge() -> list[dict]:
    nq = hf_scan("google-research-datasets/nq_open", SOURCES["nq_open"]["revision"], "nq_open/validation-00000-of-00001.parquet", ["question", "answer"])
    wq = hf_scan("stanfordnlp/web_questions", SOURCES["web_questions"]["revision"], "data/test-00000-of-00001.parquet", ["url", "question", "answers"])
    cands = [{"source_id": f"validation#{i}", "src": "nq_open", "q": r["question"], "a": r["answer"]} for i, r in enumerate(nq)]
    cands += [{"source_id": f"test#{i}", "src": "web_questions", "q": r["question"], "a": r["answers"], "url": r["url"]} for i, r in enumerate(wq)]
    picked = stratified(cands, PER_CATEGORY, lambda c: c["src"], rng("general_knowledge"), "equal")
    return [
        rec(
            "general_knowledge",
            c["src"],
            c["source_id"],
            c["q"],
            reference={"answers": c["a"], **({"freebase_url": c["url"]} if "url" in c else {})},
            notes="Short open-domain factual question (lower-cased as in the source); any listed answer counts. "
            "Answers reflect the source's snapshot date and may be stale.",
        )
        for c in picked
    ]


# ---------------------------------------------------------------- GSM8K
def build_arithmetic() -> list[dict]:
    rows = hf_scan("openai/gsm8k", SOURCES["gsm8k"]["revision"], "main/test-00000-of-00001.parquet", ["question", "answer"])
    cands = [{"source_id": f"test#{i}", **r} for i, r in enumerate(rows)]
    picked = stratified(cands, PER_CATEGORY, lambda c: "all", rng("arithmetic"), "equal")
    return [
        rec(
            "arithmetic",
            "gsm8k",
            c["source_id"],
            c["question"],
            reference={"final": c["answer"].split("####")[-1].strip().replace(",", ""), "solution": c["answer"]},
            notes="GSM8K test word problem; grade the final number.",
        )
        for c in picked
    ]


# ---------------------------------------------------------------- Dolly
DOLLY_CATS = ("brainstorming", "creative_writing", "general_qa", "summarization")


def build_open_ended() -> list[dict]:
    p = hf_file("databricks/databricks-dolly-15k", SOURCES["dolly"]["revision"], "databricks-dolly-15k.jsonl")
    cands = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        d = json.loads(line)
        if d["category"] in DOLLY_CATS:
            cands.append({"source_id": f"line{i + 1}", **d})
    picked = stratified(cands, PER_CATEGORY, lambda c: c["category"], rng("open_ended"), "equal")
    out = []
    for c in picked:
        text = c["instruction"].strip()
        if c["context"].strip():
            text += "\n\n" + c["context"].strip()
        out.append(
            rec(
                "open_ended",
                "dolly",
                c["source_id"],
                text,
                reference={"dolly_category": c["category"], "example_response": c["response"]},
                notes="Open-ended request (brainstorming / creative writing / advice-style general QA / summarization "
                "of the included passage); stratified equally by Dolly category. example_response is one human "
                "answer, not a gold standard.",
            )
        )
    return out


# ---------------------------------------------------------------- ambiguous
def build_ambiguous() -> list[dict]:
    rows = hf_scan("sewon/ambig_qa", SOURCES["ambignq"]["revision"], "light/validation-00000-of-00001.parquet", ["id", "question", "annotations"])
    amb = [
        {"source_id": r["id"], "src": "ambignq", "q": r["question"], "ann": r["annotations"]}
        for r in rows
        if "multipleQAs" in (r["annotations"]["type"] or [])
    ]
    rev = SOURCES["clariq"]["revision"]
    topics = {}
    for split in ("train", "dev"):
        lines = gh_raw("aliannejadi/ClariQ", rev, f"data/{split}.tsv").read_text(encoding="utf-8").splitlines()
        head = lines[0].split("\t")
        for ln in lines[1:]:
            row = dict(zip(head, ln.split("\t")))
            t = topics.setdefault(
                row["topic_id"],
                {"source_id": f"topic{row['topic_id']}", "src": "clariq", "q": row["initial_request"], "need": int(row["clarification_need"]), "facets": {}},
            )
            t["facets"][row["facet_id"]] = row["facet_desc"]
    clq = [t for _, t in sorted(topics.items(), key=lambda kv: int(kv[0])) if t["need"] >= 3]
    picked = stratified(amb + clq, PER_CATEGORY, lambda c: c["src"], rng("ambiguous"), "equal")
    out = []
    for c in picked:
        if c["src"] == "ambignq":
            ref = {"interpretations": [p for p in c["ann"]["qaPairs"] if p["question"]]}
            notes = "AmbigNQ question with multiple plausible readings; a good reply asks which one or answers each."
        else:
            ref = {"clarification_need": c["need"], "facets": [c["facets"][k] for k in sorted(c["facets"])]}
            notes = "ClariQ search request rated 3-4 on the 1-4 clarification-need scale; facets are what the user might mean."
        out.append(rec("ambiguous", c["src"], c["source_id"], c["q"], reference=ref, notes=notes))
    return out


# ---------------------------------------------------------------- multi-step
def build_multi_step() -> list[dict]:
    repo, rev = "osunlp/Multimodal-Mind2Web", SOURCES["mind2web"]["revision"]
    tasks = {}
    for split in ("test_task", "test_website", "test_domain"):
        files = hf_parquet_files(repo, rev, f"data/{split}-")
        cols = ["annotation_id", "confirmed_task", "website", "domain", "subdomain", "action_reprs"]
        for rows in hf_scan_many(repo, rev, files, cols):
            for r in rows:
                tasks.setdefault(r["annotation_id"], {"source_id": f"{split}/{r['annotation_id']}", "src": "mind2web", "split": split, **r})
    m2w = sorted(tasks.values(), key=lambda t: t["source_id"])
    idx = osworld_index()
    osw = [{"source_id": f"multi_apps/{t}", "src": "osworld", "t": t} for t in idx["multi_apps"]]
    half = PER_CATEGORY // 2
    r_ = rng("multi_step")
    picked_m = stratified(m2w, half, lambda c: c["split"], r_, "equal")
    picked_o = stratified(osw, PER_CATEGORY - half, lambda c: "all", r_, "equal")
    out = [
        rec(
            "multi_step",
            "mind2web",
            c["source_id"],
            c["confirmed_task"],
            reference={"action_reprs": c["action_reprs"], "website": c["website"], "domain": c["domain"], "subdomain": c["subdomain"]},
            notes="Mind2Web web task (needs several browser actions); action_reprs is the annotated action sequence "
            "on the live site at collection time. Stratified equally across the three test splits.",
        )
        for c in picked_m
    ]
    out += [
        osworld_record(
            "multi_step",
            "multi_apps",
            c["t"],
            "OSWorld multi-application Ubuntu task (several apps / steps). reference.evaluator is OSWorld's checker.",
        )
        for c in picked_o
    ]
    return out


BUILDERS = {
    "desktop_gui": build_desktop_gui,
    "shell_files": build_shell_files,
    "screen_questions": build_screen_questions,
    "image_questions": build_image_questions,
    "conversation_facts": build_conversation_facts,
    "general_knowledge": build_general_knowledge,
    "arithmetic": build_arithmetic,
    "open_ended": build_open_ended,
    "ambiguous": build_ambiguous,
    "multi_step": build_multi_step,
}


# ---------------------------------------------------------------- splitting / manifest
def split_items(category: str, items: list[dict]) -> dict[str, list[dict]]:
    items = sorted(items, key=lambda d: d["id"])
    ids = [d["id"] for d in items]
    assert len(set(ids)) == len(ids), f"duplicate ids in {category}"
    # split each source separately so every split keeps the category's source mix
    r_ = rng("split", category)
    out = {"dev": [], "calibration": [], "test": []}
    for src in sorted({d["source"] for d in items}):
        group = [d for d in items if d["source"] == src]
        r_.shuffle(group)
        n = len(group)
        n_dev = round(n * SPLITS[0][1])
        n_cal = round(n * SPLITS[1][1])
        out["dev"] += group[:n_dev]
        out["calibration"] += group[n_dev : n_dev + n_cal]
        out["test"] += group[n_dev + n_cal :]
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def images_digest(split_rows: list[dict]) -> str:
    """sha256 over sorted 'relpath sha256' lines of the images a split references."""
    lines = sorted(f"{r['image']} {sha256_file(CACHE / r['image'])}" for r in split_rows if r.get("image"))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_hashes() -> dict:
    out = {}
    for name, _ in SPLITS:
        p = CACHE / f"{name}.jsonl"
        rows = read_jsonl(p)
        out[name] = {
            "file": p.name,
            "sha256": sha256_file(p),
            "images_sha256": images_digest(rows),
            "n_items": len(rows),
            "counts": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
        }
    return out


def build() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.rmtree(IMAGES, ignore_errors=True)  # only images this build samples may remain
    splits = {name: [] for name, _ in SPLITS}
    for cat, fn in BUILDERS.items():
        items = fn()
        parts = split_items(cat, items)
        for k, v in parts.items():
            splits[k].extend(v)
        print(f"{cat:20s} total={len(items):3d} " + " ".join(f"{k}={len(v)}" for k, v in parts.items()), file=sys.stderr)
    splits["dev"].append(
        rec("owner", "owner", "factory-2026-09-18", OWNER_PROMPT, notes="Owner's own prompt (verbatim); dev only.")
    )
    for name, rows in splits.items():
        with open(CACHE / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "build_date": BUILD_DATE,
        "seed": SEED,
        "per_category_target": PER_CATEGORY,
        "split_fractions": dict(SPLITS),
        "data_dir": "~/.cache/tensorcode/heldout",
        "script": "eval/heldout/build.py",
        "splits": split_hashes(),
        "sources": SOURCES,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", MANIFEST, file=sys.stderr)


def verify() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    now = split_hashes()
    bad = 0
    for name, want in manifest["splits"].items():
        got = now[name]
        ok = all(got[k] == want[k] for k in ("sha256", "images_sha256", "n_items", "counts"))
        bad += not ok
        print(f"{name:12s} {'OK' if ok else 'MISMATCH'} sha256={got['sha256'][:16]} n={got['n_items']}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="recompute split hashes and compare with MANIFEST.json")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else build())
