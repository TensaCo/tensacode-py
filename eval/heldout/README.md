# Held-out prompt sets (frozen 2026-09-18)

Public-dataset prompts for evaluating the general agent (docs/revival/28-general-agent.md §28.4).
They were assembled by an author who did not read `src/`, `examples/` or `tests/`. No item was written
for this project, except the owner's factory prompt (dev split only).

- `build.py`: the exact script that downloads every pinned source, samples with seed `20260918`, and writes
  the splits. `uv run eval/heldout/build.py` rebuilds. `uv run eval/heldout/build.py --verify` checks the
  cached files against the manifest.
- `MANIFEST.json`: per-split counts by category, the sha256 of each split file and of the images it
  references, and every source with its URL, pinned revision and license.
- Data (not committed): `~/.cache/tensorcode/heldout/{dev,calibration,test}.jsonl` and `images/`.
  Downloaded raw sources are cached under `raw/`.

## Rules

1. **The test split is not read by the builder.** Don't open, print, grep or summarise `test.jsonl`, and
   don't look at per-item test results. Only aggregate numbers leave a test run.
2. **Any code change after the test split is read retires it.** The same applies to calibration once it has
   been used to set a threshold that is then changed. A retired split stays in the results history. To make
   a new one, change `SEED` in `build.py`, record the new manifest, and keep the old one.
3. **Frozen.** Don't edit items, filters or the seed in place. `--verify` must report OK before any run
   counts. A change to `build.py` gives a new named set.
4. **Results are append-only.** Every result row records the manifest's split sha256. Nothing is
   overwritten.
5. **Use of splits.** Develop freely on dev. Use calibration only to set thresholds and abstention points.
   Test is for headline numbers, run rarely.
6. Grade by world state or by the source's reference, never by the agent's own report.

## Record format

One JSON object per line: `id`, `category`, `source`, `source_id`, `license`, `text` (the user message
exactly as typed), `image` (path relative to the data dir, or null), `reference` (the source's gold answer,
command or evaluator, or null), and `notes`. `conversation_facts` items add `history`: earlier chat sessions
to replay before `text`.

## Categories

Each category has 60 items, split 20/20/60 (12 dev, 12 calibration, 36 test) by a seeded shuffle. The
shuffle runs within each source, so every split keeps the category's source mix.

| category | tests | source (stratification) |
|---|---|---|
| desktop_gui | single-app Ubuntu desktop tasks, including infeasible ones | OSWorld `examples/*` except multi_apps (equal per app domain) |
| shell_files | file and shell requests in English; gold bash is kept as a reference | NL2Bash (proportional by the command's head utility) |
| screen_questions | questions about what a UI screenshot shows | ScreenQA-Short test on RICO Android screens (one question per screen) |
| image_questions | questions about natural photos | VQA v2 val (one per image; equal across yes/no, number, other) |
| conversation_facts | remembering facts the user stated in earlier sessions, knowledge updates, temporal reasoning, abstention | LongMemEval oracle (equal across 5 user-centred question types) |
| general_knowledge | short factual questions | NQ-open val, WebQuestions test (30 each) |
| arithmetic | math word problems | GSM8K test |
| open_ended | brainstorming, creative writing, advice-style questions, summarising a given passage | Dolly-15k (equal across 4 categories) |
| ambiguous | requests that need clarification | AmbigNQ val multipleQAs, ClariQ need ≥ 3 (30 each) |
| multi_step | tasks that take several steps | Mind2Web test splits (30; equal per split), OSWorld multi_apps (30) |
| owner | the owner's factory-design prompt | owner (dev only) |

Caveats:
- OSWorld and Mind2Web references describe the original VM or live website. The computerworld engine won't
  reproduce them exactly, so treat those references as intent and not as a grader.
- Screen questions are about mobile screenshots, not Ubuntu screens.
- NQ and WebQuestions answers are snapshots and may be out of date.
- Images are only the sampled ones, saved under `images/<category>/`.

## Licenses

| source | license |
|---|---|
| OSWorld | Apache-2.0 |
| NL2Bash | GPL-3.0 |
| ScreenQA | CC-BY-4.0 (RICO screenshots under RICO terms) |
| VQA v2 | CC-BY-4.0 annotations (COCO images under Flickr terms) |
| LongMemEval | MIT |
| NQ-open | CC-BY-SA-3.0 |
| WebQuestions | CC-BY-4.0 |
| GSM8K | MIT |
| Dolly-15k | CC-BY-SA-3.0 |
| AmbigNQ | CC-BY-SA-3.0 |
| Mind2Web | CC-BY-4.0 (the HF multimodal mirror's card says OpenRAIL) |
| ClariQ | no explicit license |

None of this data is committed. It stays in the user cache, so share-alike, GPL and unlicensed terms don't
put redistribution obligations on this repository.
