# Kimi Systems Lab

Reproducible experiments for inspecting, reconstructing, and reasoning about open-weight model
systems, with the boundary of each result made explicit.

Every experiment ships the thing that produced its numbers. If a figure appears in a report,
article, or presentation, the script that computed it is here, along with its raw output.

## Experiments

| Experiment | What it checks | Evidence class |
| --- | --- | --- |
| `delta-rule-correction/` | Why a memory that can only add cannot replace a stale fact. | Teaching miniature |
| `million-token-attention/` | The dense KV-cache and attention-arithmetic bill at one million tokens. | Derived |
| `flashattention-boundary/` | Which dense-attention cost tiling removes—and which ones remain. | Derived |

Further experiments are added only when their inputs and limits can be checked locally.

## Running any of it

Python 3, standard library only. No numpy, no torch, no GPU, no install.

```bash
python3 delta-rule-correction/delta_rule_correction.py
python3 million-token-attention/attention_cost_at_1m.py
python3 flashattention-boundary/flashattention_boundary.py
```

Each script asserts its own headline figures, so a transcription error fails loudly. The
`raw-output.txt` file in each folder is the captured output of the committed script.

The scripts are written in plain Python rather than with a tensor library on purpose: the point
is that you can check the claim in two seconds without setting anything up.

## Evidence classes

The lab does not treat all numbers as the same kind of thing. Every claim carries one of these
labels, and they are not interchangeable:

| Class | Meaning |
|---|---|
| **Released** | A primary release, configuration, or licence fact, retained with source and date. |
| **Measured by us** | A run of mine, with hardware, software, configuration, workload, date, raw output, and exclusions. |
| **Derived** | Arithmetic from named inputs, with formula, units, and assumptions stated. |
| **Configured** | A setting or declared value. It is never presented as observed behaviour. |
| **Teaching miniature** | A deliberately reduced model or test that preserves a named relationship but establishes nothing about Kimi K3's performance. |
| **Reconstructed** | A falsifiable hypothesis about unpublished wiring, constrained by public artifacts. It must state its assumptions, alternatives, and a falsifier; it is never a route around private-source restrictions. |

A claim also states its provenance: primary release, my own run, or derivation. **Someone else's
experiment is never labelled "measured by us."**

## Cost tiers

| Tier | Environment |
|---:|---|
| 0 | Source card or derivation. No run implied. |
| 1 | CPU miniature — a bounded mechanism reproduction. |
| 2 | Real checkpoint, no GPU — inspecting a released config, index, or tensor. |
| 3 | Single GPU, with a predeclared cost ceiling. |
| 4 | Multi-GPU or full serving. Only after funding, a receipt, and a stop-loss budget. |

Everything published here so far is Tier 0 or Tier 1 — arithmetic and CPU miniatures.

## What is not here

No Kimi K3 benchmark, latency, throughput, cost, or quality figure. Nothing in this repository
was run on a GPU, and no released checkpoint has been downloaded, loaded, or served.

Where an experiment prices a **hypothetical dense transformer** using Kimi K3's attention dimensions,
that is stated plainly in the experiment's own README. Kimi K3 is not built that way — that is the point
of the arithmetic, and those totals are never Kimi K3's.

## Sources

Every release fact traces to Moonshot's own published artifacts:

- Kimi K3 model repository and configuration — https://huggingface.co/moonshotai/Kimi-K3
- Kimi K3 announcement and technical report — https://www.kimi.com/news/kimi-k3-open-source

Mechanisms are cited to the primary literature that introduced them, named in each experiment's README.
Arithmetic is recomputed here rather than quoted, and asserted in the script so it can be
checked rather than trusted.

## Licence

MIT — see [LICENSE](LICENSE).
