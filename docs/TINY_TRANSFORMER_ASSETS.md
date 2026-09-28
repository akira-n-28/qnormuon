# Local offline asset inspection (2026-09-28)

Inspection used directory metadata at conservative depth (at most three levels
under obvious projects), small manifests, and 12-byte binary headers. It did
not read dataset contents on the frontend, scan full caches, load checkpoints,
or download anything. Project roots inspected were modded-nanogpt,
parameter-golf, autoresearch, TWA, Guido-1, tesi, and models. The pg directory
is an environment, not a dataset. Log files are not considered training data.
This is a bounded inventory, not an exhaustive claim about the home directory.

| Candidate | Format / approximate size | Split availability | Tokenizer / offline suitability |
| --- | --- | --- | --- |
| `/home/prignano/parameter-golf/data/datasets/fineweb10B_sp1024/` | 10 visible training shards, each 200,001,024 bytes; validation 124,044,716 bytes | Named train/validation shards | Selected: pretokenized uint16 IDs, vocabulary 1024; no runtime tokenizer needed |
| `/home/prignano/parameter-golf/data/tokenizers/fineweb_1024_bpe.model` | SentencePiece, 254,483 bytes | Shared tokenizer | Local manifest identifies vocabulary 1024, BOS 1, EOS 2; file exists, not loaded by the training loop |
| `/home/prignano/modded-nanogpt/data/fineweb10B/` | 9 visible training shards and one validation shard, each 200,001,024 bytes | Both present | Same header convention; different tokenizer provenance not fully audited for this smoke; not selected |
| `/home/prignano/tesi/distill/data/` | Small JSONL candidates include `distill_kept_smoke.jsonl` (46,291 bytes), `miniset_full.jsonl` (110,021), `problems.jsonl` (23,465); also much larger unrelated files | Train/validation separation not established | Would need field/schema and split audit plus explicit tokenization; not selected |
| `/home/prignano/tesi/grpo_h100/data/` | `gsm8k.jsonl` 772,535 bytes; `math500.jsonl` 119,067 | Evaluation-oriented assets; training split not established | Not appropriate to silently repurpose held-out math evaluations for training |
| `/home/prignano/tesi/emma5_model/bpe.model`, model tokenizer JSON files | Local tokenizer files, approximately 1–13 MB | Not applicable | No matched small training corpus verified; not selected; no model weights opened |
| Harness synthetic stream | Deterministic periodic sequence with 10% seeded random replacements | Independent train/validation seeds | No tokenizer or external files; implemented for tests and optional offline use |

Selected binary format: a 1,024-byte header (256 little-endian int32 values),
magic `20240520`, version `1`, token count in the third integer, followed by
little-endian uint16 IDs. The harness checks the exact file size against the
header and bounds IDs by vocabulary size. It loads only a small prefix inside
the compute allocation, not the full shard.

Smoke train source:
`/home/prignano/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin`.
Loaded **1,048,576 tokens**, 2 MiB of source token bytes.
Validation source: corresponding `fineweb_val_000000.bin`.
Loaded **65,536 tokens**, 128 KiB of source token bytes.
The manifest reports a source revision and seed, but those are local manifest
claims; upstream identity and full-corpus deduplication were not independently
verified. Reproducibility of the consumed prefixes is established by content
hashes recorded in each run's provenance. Train and validation files are
separate; semantic duplicate overlap has not been audited.

Train-prefix SHA256 (uint16 encoding):
`afce8cb11a4012770ee331ad3e3ca931ccc761c7716a680dfe28efc83e525e99`.
Validation-prefix SHA256:
`09545e7dae1baab9e2c244bfc5d88103b759a56bc1d3a2aca83cb09b9a45db6a`.

The shared `/home` mount is GlusterFS with approximately 12 TB aggregate free.
A personal quota remains **unknown**; aggregate capacity is not a quota.
The run directory is explicitly `/home/prignano/qnormuon-runs/tiny-transformer`,
outside source. The smoke writes one checkpoint per completed method, compact
JSON/JSONL metrics, and small SLURM logs. No dataset copy or preprocessing cache
is written. The runner requires at least 2 GiB aggregate free space; this does
not establish personal quota availability. Actual checkpoint creation was
validated on the compute node.
