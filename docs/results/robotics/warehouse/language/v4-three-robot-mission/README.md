# Actual model instruction for three humanoids and six jobs

Pinned Qwen3-4B-Instruct-2507 generated the requested mission correctly on its first attempt: three A transfers in parallel, followed by three B transfers in parallel. Its 89-token raw output, native prompt hash, pinned model revision, source hashes, and complete controlled-language certificate are retained in proposal.json. No model repair or authored AST replacement was used.

`instruction.json` is the strict typed tree derived from that actual output, suitable for `physics_jobs.py --instruction-json`. Its SHA-256 is `ab94f46212154a3d4c082eb1ba09cbb34743c967123dcb6ff4b96d1c4eb954e5`. `execution-input.json` contains the exact mission argument array, including the original English text for the physical receipt.

The certificate establishes agreement within the declared controlled grammar. The physical executor must still ground every object/destination in fresh sensor observations, validate its capability scope, and check dependencies/resources/motion feasibility. This artifact alone is not a robot execution or success result. Physical evidence should bind this exact instruction hash.

Inference used the peer GB10 with private MPS verified at 32/48 multiprocessors, two CPU cores, and a 16 GiB memory limit. Peer Python/driver differ from host experiments; no cross-host latency claim is made. The entire frozen source archive is hash-verified against the launch receipt.
