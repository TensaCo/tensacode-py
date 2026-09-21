# Pretrained cognitive tools implementation plan

Goal: replace provider-owned tools with complete trainable models, preferred Hugging Face distribution, shared cognitive workspace, durable learning, and explicit symbolic graph stubs.

Approved specification: the ten-part architecture proposal in this conversation, with the user's correction that graph operations remain largely unimplemented symbolic interfaces. Work only on main; commit verified milestones. Developer docs describe current capabilities, not speculative milestones.

## Contracts and ownership

- `_internal/pretrained.py`: PretrainedTool(nn.Module), dictionary config, versioned safetensors/config save/load, Hub/local loading, operation bindings, separate session persistence. Constructors do not download.
- `_internal/workspace.py`: Workspace(nn.Module) transforms encoded token tensors [batch,tokens,width] plus mask into source-linked learned slots and response-conditioning tensors. Explicit configuration, no authored semantic seeds. Evidence and interpretations stay distinct.
- `tools/chatbot.py`: Chatbot(PretrainedTool) owns tokenizer, seq2seq input encoder, Workspace and llm.Decode language decoder. Config reconstructs every component. Fresh sessions, transactional turns, teacher-forced training path and generation. `from_foundation` is an explicit bootstrap for training; `from_pretrained` loads complete TensorCode tools.
- `ops/llm/decode.py`: add genuine locally differentiable language realization without removing message text extraction used independently.
- `tools/investigator.py`, `tools/planner.py`: concrete trainable model interfaces over shared workspace, sourced evidence, explicit predictions and revisions; no external actions by default.
- `ops/graph`: preserve representation records; replace executable graph reasoning/learning operations with explicit NotImplementedError symbolic interfaces including TextEncode/TextDecode. Retire graph benchmarks and implementation tests that no longer apply.
- `training`: stable owned-operation bindings, traced teacher-forcing objective and persistence, resumable optimizer/RNG/progress checkpoint utilities.
- `examples`: collect/train/save/load CLI workflows and real-data checkpoint training/evaluation with before/after and workspace ablations.

## Milestones

- [x] Common pretrained artifact lifecycle: local roundtrip, revision/offline loading, wrong-class/version/config rejection, no session data in model assets.
- [x] Workspace and owned tool architecture: model parameters constructed up front, source preservation, masked evidence invariance, gradients, generation from changed workspace, independent sessions and rollback.
- [x] Graph symbolic interface migration: stubs fail explicitly; no hidden old execution path; docs and examples match.
- [x] Complete tools and training: explicit feedback through local gradient paths, save/reload experiences, optimizer/RNG resume, fresh-process model parity.
- [x] Real data and pretrained delivery: train a compatible pretrained foundation/workspace on sourced data, evaluate held-out records and ablations, save complete checkpoints, publish if authenticated and authorized. Report actual limitations if hosting or training resources block a deliverable.
- [x] Documentation and integration: update imports/examples, validate all tests, package builds, CLI/link checks, independent review, coherent commits and push.

## Review focus

Checkpoint configurations must reconstruct without network or executable remote code. Shared weights must retain aliases. Raw evidence and generated interpretations must remain distinguishable. State updates must be transactional. Training labels must never enter inference inputs. Learned workspace claims require active behavior and ablation measurements, not schema tests alone. Hosted pretrained tools must have real artifacts, provenance and measured scope.

## Capability acceptance still open

- [ ] Demonstrate a consistent held-out benefit from recurrent workspace updates. Current HotpotQA answer EM/F1 is identical when bypassing them; ranking ablations are mixed.
- [ ] Produce useful scene grounding. Both random and CLIP-backed VSR experiments failed; the published Scene checkpoint is explicitly experimental.
- [ ] Learn hypothesis/candidate generation and causal plan utility. Current rankers consume supplied candidates, and the Planner release learns supporting-document relevance.
- [ ] Validate general multi-turn conversational competence and calibrated uncertainty. The released Chatbot is a narrow oracle-context QA fine-tune of FLAN.

Implementation and reproducible training/distribution milestones are complete. These capability requirements are not marked achieved by schemas, unit tests, or pretrained loading. See docs/validation.md and docs/results/pretrained-releases.json for measurements and exact public artifacts.
