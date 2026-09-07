"""
Kimi K3 Systems Lab - Post 02 proof object
==========================================

Claim under test
----------------
Kimi K3's released config declares max_position_embeddings = 1,048,576. If you
read the rest of that same config as a conventional dense transformer - 93
layers, 96 heads, 128 dimensions per head, full attention everywhere - the
declared context window is arithmetically unservable. The config refutes the
naive reading of itself.

Evidence class : Derived (Tier 0). Pure arithmetic from released config values.
Provenance     : Ankit derivation from the published Kimi K3 configuration,
                 recomputed and asserted here rather than quoted.

Scope boundary
--------------
THE HYPOTHETICAL DENSE MODEL PRICED HERE IS NOT KIMI K3.
It is K3's attention dimensions wired as a textbook dense transformer - the
design K3's authors walked away from. Every total below is the bill for that
hypothetical, not a measurement of K3, and not a measurement of anything.
No run, no GPU, no benchmark, no latency, no throughput.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 attention_cost_at_1m.py
"""

# ---------------------------------------------------------------------------
# RELEASED inputs - Kimi K3 config.json
# Re-verify these six keys against the primary artifact before publishing:
#   https://huggingface.co/moonshotai/Kimi-K3  ->  config.json
# ---------------------------------------------------------------------------
N_LAYERS      = 93          # num_hidden_layers
N_HEADS       = 96          # num_attention_heads
HEAD_DIM      = 128         # head_dim
T_MAX         = 1_048_576   # max_position_embeddings  (2**20 exactly)

# ---------------------------------------------------------------------------
# DERIVED / stated assumptions
# ---------------------------------------------------------------------------
BYTES_BF16    = 2           # declared dtype for non-expert weights
BYTES_FP32    = 4
# A realistic single-node ceiling, NOT a vendor recommendation. 288 GB is the
# published HBM capacity of one NVIDIA B300 SXM; times eight is arithmetic.
# Moonshot's own deployment guidance is NOT cited here and must not be inferred:
# an earlier draft called this 'the recommended node', which attributed a
# recommendation to the model's authors that they did not make.
NODE_HBM_BYTES = 8 * 288 * 10**9   # 8 x 288 GB = 2.30 TB

GB  = 10**9      # decimal GB, the unit GPU memory is marketed in
TB  = 10**12
MiB = 2**20
TiB = 2**40


def rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
rule("INPUTS  (released config values, treated as given)")
print(f"""
  num_hidden_layers        {N_LAYERS:>12,}
  num_attention_heads      {N_HEADS:>12,}
  head_dim                 {HEAD_DIM:>12,}
  max_position_embeddings  {T_MAX:>12,}   (= 2**20)
  activation dtype         {'bfloat16':>12}   ({BYTES_BF16} bytes/scalar)
""")
assert T_MAX == 2 ** 20


# ---------------------------------------------------------------------------
rule("COST 1  -  the score matrix is quadratic in sequence length")
print("""
  Standard attention scores every query against every key, so each head forms
  a T x T matrix. Byte counts below are exact: T * T * bytes_per_scalar.
""")
print(f"  {'T':>10} | {'fp32 per head':>16} | {'bf16 per head':>16}")
print(f"  {'-'*10}-+-{'-'*16}-+-{'-'*16}")
for T in (1_024, 2_048, 4_096, 8_192, 131_072, T_MAX):
    fp32 = T * T * BYTES_FP32
    bf16 = T * T * BYTES_BF16
    def h(b):
        return f"{b/TiB:.2f} TiB" if b >= TiB else f"{b/MiB:,.1f} MiB"
    mark = "  <-- 1M context" if T == T_MAX else ""
    print(f"  {T:>10,} | {h(fp32):>16} | {h(bf16):>16}{mark}")

score_1m_bf16 = T_MAX * T_MAX * BYTES_BF16
print(f"""
  Each doubling of T quadruples the matrix: 4 -> 16 -> 64 -> 256 MiB.
  At T = 1,048,576 one head in bf16 wants {score_1m_bf16/TiB:.1f} TiB.
  That is ONE head, of {N_HEADS}, in ONE layer, of {N_LAYERS}.
""")
assert abs(score_1m_bf16 / TiB - 2.0) < 1e-9


# ---------------------------------------------------------------------------
rule("COST 2  -  the KV cache is linear in T, with a very large constant")
print("""
  Generation caches one key and one value vector per head, per layer, per
  token. Unit check runs left to right:

    heads x head_dim x 2 tensors x bytes  =  bytes per layer per token
""")
per_token_per_layer = N_HEADS * HEAD_DIM * 2 * BYTES_BF16
per_token           = per_token_per_layer * N_LAYERS

print(f"    {N_HEADS} x {HEAD_DIM} x 2 x {BYTES_BF16} = {per_token_per_layer:,} bytes"
      f"  = {per_token_per_layer/2**10:.0f} KiB  per layer per token")
print(f"    x {N_LAYERS} layers      = {per_token:,} bytes"
      f"  = {per_token/MiB:.2f} MiB  per token, whole stack")
print()
print(f"  {'context T':>12} | {'KV cache':>14} | {'vs 8x B300 node':>17}")
print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*17}")
for T in (8_192, 131_072, T_MAX):
    total = per_token * T
    print(f"  {T:>12,} | {total/GB:>11,.1f} GB | {total/NODE_HBM_BYTES:>16.2f}x")

cache_1m  = per_token * T_MAX
cache_128 = per_token * 131_072
ratio     = cache_1m / NODE_HBM_BYTES

print(f"""
  Hand-check of the two headline figures:
    128K context : {cache_128:,} bytes = {cache_128/GB:,.1f} GB
    1M   context : {cache_1m:,} bytes = {cache_1m/GB:,.1f} GB = {cache_1m/TB:.2f} TB

  An 8-GPU node at 288 GB each is {NODE_HBM_BYTES/TB:.2f} TB of HBM,
  and that node has to hold the 2.8T-parameter model as well.

  A single 1M-token sequence would need {ratio:.2f}x the node's ENTIRE memory,
  weights included. Batch size one. Before any weights are loaded.
""")
assert abs(per_token_per_layer - 49_152) == 0
assert abs(per_token / MiB - 4.36) < 0.01
assert abs(cache_128 / GB - 599.1) < 0.1
# 4,793.2, not 4,792.8. The old expected value was wrong and a 0.5 GB tolerance
# let it pass, which is the exact failure an assertion is supposed to prevent.
assert abs(cache_1m / GB - 4_793.2) < 0.05
# Where the cache alone fills the node. This is the number the declared context
# window has to be read against: the wall arrives well before 1M tokens, and the
# model weights have not been loaded at this point.
tokens_to_fill = NODE_HBM_BYTES / per_token
print(f"""
  The KV cache alone fills the whole node at {tokens_to_fill:,.0f} tokens.
  That is {tokens_to_fill/T_MAX*100:.0f}% of the declared {T_MAX:,}-token window,
  and it assumes zero bytes are spent on the model itself.
""")

assert abs(ratio - 2.08) < 0.01
assert 500_000 < tokens_to_fill < 510_000, "node-fill crossover moved"


# ---------------------------------------------------------------------------
rule("COST 3  -  the arithmetic floor, in FLOPs")
print("""
  Scoring T queries against T keys costs 2 * T^2 * head_dim FLOPs per head
  (one multiply and one add per product). Applying the weights to V costs the
  same shape of work again.
""")
flops_qk = 2 * T_MAX * T_MAX * HEAD_DIM * N_HEADS * N_LAYERS
flops_av = flops_qk
flops    = flops_qk + flops_av

print(f"    scores (Q @ K^T) : {flops_qk:.3e} FLOPs")
print(f"    weights @ V      : {flops_av:.3e} FLOPs")
print(f"    total            : {flops:.3e} FLOPs   = {flops/1e18:.2f} exaFLOPs")
print(f"""
  {flops/1e18:.2f} exaFLOPs for the attention arithmetic of ONE forward pass over
  a million tokens, before a single MLP or projection is counted.

  Growth between 128K and 1M context is quadratic: 8x the tokens, {(T_MAX/131072)**2:.0f}x the work.
""")
assert abs(flops / 1e18 - 5.026) < 0.01


# ---------------------------------------------------------------------------
rule("RESULT")
print(f"""
  Reading K3's declared 1M context through a dense-attention architecture gives
  two independent walls, from the same config:

    memory   {cache_1m/TB:.1f} TB of KV cache for one sequence  ({ratio:.2f}x the whole node)
    compute  {flops/1e18:.2f} exaFLOPs of attention per forward pass

  Neither is a benchmark. Both are arithmetic, and arithmetic is enough to
  rule the design out. A context-length claim is not an architecture until
  someone has done this sum.

  Again: this prices the dense model K3's designers REFUSED to build.
  K3's own answer - a fixed-size state on 69 layers, a 512-wide compressed
  latent on the other 24 - is a later post, and needs its own arithmetic.

  All assertions passed.
""")
