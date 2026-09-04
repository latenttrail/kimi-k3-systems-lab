# Post 2 — One million tokens of dense attention costs 4.8 TB of cache.

**Evidence class:** Derived · **Cost tier:** 0 (arithmetic) · **Provenance:** my own derivation

## The claim

Kimi K3's released config declares `max_position_embeddings` = 1,048,576. Read the rest of that
same config as a conventional dense transformer — 93 layers, 96 heads, 128 dimensions per head,
bfloat16, full attention everywhere — and the declared context window is arithmetically
unservable. The config refutes the naive reading of itself.

```
$ python3 attention_cost_at_1m.py

  96 x 128 x 2 x 2 = 49,152 bytes  = 48 KiB  per layer per token
  x 93 layers      = 4,571,136 bytes  = 4.36 MiB  per token, whole stack

     context T |       KV cache |   vs 8x B300 node
  -------------+----------------+------------------
       131,072 |       599.1 GB |             0.26x
     1,048,576 |     4,793.2 GB |             2.08x

  total attention arithmetic : 5.026e+18 FLOPs = 5.03 exaFLOPs
```

## Inputs

All from Kimi K3's `config.json`:

| Key | Value |
|---|---|
| `num_hidden_layers` | 93 |
| `num_attention_heads` | 96 |
| `head_dim` | 128 |
| `max_position_embeddings` | 1,048,576 (2²⁰ exactly) |
| declared dtype | bfloat16 |

The node comparison is 8 × 288 GB HBM = 2.30 TB.

## Units

`GB` is decimal (10⁹ bytes), matching how GPU memory is marketed; `TB` is 10¹². `KiB`, `MiB` and
`TiB` are binary. The 1M-token cache is 4,793.2 GB decimal, which is 4.36 TiB binary. Both are
stated because either alone invites a wrong comparison.

Every conversion in the script is a literal division by a named constant, so you can check the
unit as well as the number.

## Boundary

**The model priced here is not Kimi K3.** It is a hypothetical fully-dense 93-layer transformer
using K3's attention dimensions — the design K3's authors walked away from. Every total belongs to
that hypothetical.

This is not a hedge attached to the end. It is the argument: the arithmetic is what makes K3's
actual attention stack necessary.

No run, no GPU, no benchmark, no latency, no throughput, no cost. The FLOP figure is an arithmetic
floor, not a runtime.

## Run it

```bash
python3 attention_cost_at_1m.py
```

Standard library only. The script asserts every headline figure, so a transcription error fails
loudly rather than being published. Inputs come from the released configuration; the arithmetic is
recomputed here rather than quoted.
