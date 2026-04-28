# `agent-skills-py-proof`

A minimal Python implementation of the [agent-skills v0.2 specification](https://github.com/MauricioPerera/agent-skills/blob/main/SPEC.md). **Sole purpose: empirically prove the spec is sufficient for a second, independent implementation.**

This is **not** a general-purpose Python skill bank. It implements a deliberately small subset (parse, sync, query, bench) — enough to validate the retrieval contract end-to-end. Exec, audit, rerank, signing, and author tooling are **out of scope** here; they live in the [reference TypeScript CLI](https://github.com/MauricioPerera/agent-skills-cli).

## The empirical claim

Spec v0.2.0 was published on the back of 11 minor releases of the reference TypeScript CLI. Until now, the spec had only one production implementation. The risk: **patterns documented in the spec may have leaked details that only matter to that implementation**, and a second one might not be able to reproduce its retrieval behaviour from the spec alone.

This repo refutes that risk with a single experiment:

> Implement a Python skill bank from the spec only. Ingest the same public skill pack as the TS CLI. Run the same bench truth file. Get the same numbers.

**Result**: bit-for-bit identical retrieval scores, identical bench accuracy, identical failure mode.

| Metric | TS CLI ([`agent-skills-cli`](https://github.com/MauricioPerera/agent-skills-cli) v0.11.0) | Python (`bank.py` in this repo) |
|---|---:|---:|
| Top-1 accuracy | 34/35 (97.1 %) | **34/35 (97.1 %)** ✓ |
| Top-3 accuracy | 35/35 (100 %) | **35/35 (100 %)** ✓ |
| Top-5 accuracy | 35/35 (100 %) | **35/35 (100 %)** ✓ |
| Mean top-1 score | 0.551 | **0.551** ✓ |
| Mean margin (top-1 → top-2) | +0.175 | **+0.175** ✓ |
| Sole failure | *"read what's at this https url"* → got `read-file` (rank 2, score 0.450) | **identical** ✓ |

Both runs:
- Sync `github.com/MauricioPerera/agent-skills-pack@main` (7 skills).
- Use Ollama `embeddinggemma` (768-dim) as the embedding provider.
- Run pure cosine retrieval (no rerank, no applicable_when filter — those are spec-optional and orthogonal to the claim).
- Bench against [`agent-skills-pack/bench-truth.jsonl`](https://github.com/MauricioPerera/agent-skills-pack/blob/main/bench-truth.jsonl) (35 paraphrases × 7 skills).

## Why this matters

A spec that one implementation can satisfy is just documentation of that implementation. A spec that **two independent implementations satisfy with identical observable behaviour** is a real specification. The numbers above are the empirical step from "documented" to "specified".

What this rules out:
- The TS CLI's retrieval relies on a TS-specific data structure → false (Python `list[float]` produces the same cosine).
- The embedding-text composition has unspecified behaviour → false (SPEC §4.2's *"title . use_when . description . examples[].intent . tags"* recipe was sufficient to match scores).
- The bench format is under-specified → false (SPEC §4.6 was enough; the Python parser handles JSONL with `#` comments, JSON-array, fail-fast on unknown short ids — all from the spec).

What this does **not** rule out (out of scope, but worth flagging):
- That `exec` would behave the same. Different shell, different OS, different timeout semantics. This proof doesn't claim execution parity — only retrieval.
- That signature verification would behave the same. Spec §5.1 is more permissive (Level 3a *or* 3b); two implementations could legitimately differ in which they support.
- That rerank would behave the same. Rerank is optional per spec; this proof uses no rerank.

## What's in `bank.py`

A single ~510-line file (627 with comments and blank lines) with three subcommands and four pure functions. Standard library + `pyyaml` + `requests`.

Implemented sections of the spec:

| Section | What it covers | LOC |
|---|---|---:|
| §2.1 + §2.2 | SKILL.md frontmatter parse + required-field validation | ~30 |
| §3.2 | `skills-index.json` consumption | ~20 |
| §4.2 | Embedding-text composition (`title . use_when . description . examples[].intent . tags`) | ~15 |
| §4.3 | Pure cosine retrieval (no rerank, no filter) | ~15 |
| §4.6 | JSONL/JSON-array truth-file parsing + bench accuracy reporting | ~80 |
| §4.7 | Embedding provider abstraction (Ollama only) — name + dim + embed | ~40 |
| §7.1 | Sync: GitHub ref → SHA → jsDelivr CDN → embed → store | ~80 |

Out of scope here (see SPEC for the contracts):

- §2.6 — substitution / shell quoting (`exec`-only)
- §4.3.1 — rerank patterns (orthogonal)
- §4.4 — execution contract (`exec`-only)
- §4.5 — audit log (`exec`-only)
- §5.x — signature verification

## Try it

```bash
# Prereqs: Python 3.10+, Ollama running locally with embeddinggemma pulled.
ollama pull embeddinggemma
pip install pyyaml requests

git clone https://github.com/MauricioPerera/agent-skills-py-proof
cd agent-skills-py-proof

# Sync the public pack (~85 s — Ollama embedding is the bottleneck)
python bank.py sync github.com/MauricioPerera/agent-skills-pack@main

# Single query
python bank.py query "fetch the contents of a URL"
# Top 1 skills for: "fetch the contents of a URL"
#   model: ollama:embeddinggemma
#   1. [0.629] github.com/.../http-get
#       HTTP GET request

# Full bench (~95 s on Ollama)
curl -fsSL https://raw.githubusercontent.com/MauricioPerera/agent-skills-pack/main/bench-truth.jsonl > bench-truth.jsonl
python bank.py bench bench-truth.jsonl
```

Add `--json` to any command for machine-readable output.

The bank state lives at `~/.config/agent-skills-py/` (XDG-style) by default. Override with `--bank-dir <path>`. **The Python bank uses a separate state directory from the TS CLI** so you can run both side-by-side and compare.

## Reproducing the parity claim

Run the same sync + bench against both implementations on the same machine, with the same Ollama model, against the same pack. The numbers in the table above should match within ±0 (true bit equality, not just close).

```bash
# TS CLI (after install per https://github.com/MauricioPerera/agent-skills-cli):
EMBEDDING_PROVIDER=ollama OLLAMA_MODEL=embeddinggemma agent-skills sync github.com/MauricioPerera/agent-skills-pack@main
EMBEDDING_PROVIDER=ollama OLLAMA_MODEL=embeddinggemma agent-skills bench bench-truth.jsonl

# Python proof:
python bank.py sync github.com/MauricioPerera/agent-skills-pack@main
python bank.py bench bench-truth.jsonl
```

Both should report **34/35 top-1, 35/35 top-3, mean top-1 = 0.551, margin = +0.175, sole failure on the same paraphrase**.

## What this changes

If you want to write a third implementation — Rust, Go, Java, Common Lisp — you now have:

1. The spec ([SPEC.md](https://github.com/MauricioPerera/agent-skills/blob/main/SPEC.md)) as the contract.
2. The TS CLI as a comprehensive, production-grade reference.
3. **This Python file as a 510-line reading list** that demonstrates which spec sections matter for retrieval, in what order, and with what numerical guarantees.

If your implementation produces different scores on the same setup, **either you're doing something different or you've found a gap in the spec**. Either way, that's actionable.

## License

[MIT](./LICENSE) — copy-paste, fork, port. The point is more implementations.

## Sister projects

- [`agent-skills`](https://github.com/MauricioPerera/agent-skills) — canonical specification (v0.2.0).
- [`agent-skills-cli`](https://github.com/MauricioPerera/agent-skills-cli) — reference TypeScript CLI (v0.11.0).
- [`agent-skills-pack`](https://github.com/MauricioPerera/agent-skills-pack) — example skill pack with `bench-truth.jsonl`.
