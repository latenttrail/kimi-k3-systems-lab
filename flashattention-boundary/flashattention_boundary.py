"""
Kimi K3 Systems Lab - Post 03 proof object
==========================================

Claim under test
----------------
FlashAttention-style tiling removes the STORAGE cost of the T x T score matrix.
It does not remove the ARITHMETIC, and it does not touch the KV cache at all.
Of the three walls that a million-token dense context runs into, tiling
removes exactly one - and it is not the one that binds during serving.

Evidence class : Derived (Tier 0). Companion derivation to Post 02.
Provenance     : Ankit derivation. The storage/compute distinction follows from
                 FlashAttention (Dao et al., 2022), which keeps the score matrix
                 out of HBM without changing the operation count.

Scope boundary
--------------
NO RUNTIME CLAIM IS MADE HERE. Not a benchmark, not a speedup, not a latency,
not a throughput number. FLOPs are an arithmetic floor, and wall-clock time is
not FLOPs - a tiled kernel is genuinely faster in practice, largely because it
moves far less data between HBM and on-chip memory. That is a real effect and
this derivation does not measure it. What it shows is narrower and harder:
the arithmetic floor is INVARIANT under tiling.

As in Post 02, the model priced here is a hypothetical dense 93-layer
transformer using K3's attention dimensions. It is not Kimi K3.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 flashattention_boundary.py
"""

N_LAYERS   = 93
N_HEADS    = 96
HEAD_DIM   = 128
T          = 1_048_576
BYTES_BF16 = 2

GB, TB, MiB, TiB = 10**9, 10**12, 2**20, 2**40
NODE_HBM_BYTES = 8 * 288 * 10**9


def rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
rule("THE THREE COSTS OF DENSE ATTENTION AT T = 1,048,576")

score_per_head  = T * T * BYTES_BF16
score_all       = score_per_head * N_HEADS            # one layer, all heads
flops           = 4 * T * T * HEAD_DIM * N_HEADS * N_LAYERS
kv_per_token    = N_HEADS * HEAD_DIM * 2 * BYTES_BF16 * N_LAYERS
kv_cache        = kv_per_token * T

print(f"""
  A  score-matrix storage   {score_per_head/TiB:>8.1f} TiB per head
                            {score_all/TB:>8,.0f} TB for one layer's {N_HEADS} heads
  B  attention arithmetic   {flops/1e18:>8.2f} exaFLOPs per forward pass
  C  KV cache               {kv_cache/TB:>8.2f} TB for one sequence
                            ({kv_cache/NODE_HBM_BYTES:.2f}x the 8x B300 node's total HBM)
""")


# ---------------------------------------------------------------------------
rule("WHAT TILING CHANGES")
print("""
  A tiled kernel never forms the whole T x T matrix. It walks blocks of queries
  against blocks of keys, keeps each block's scores in on-chip memory, folds
  them into a running softmax, and discards them. The output is identical.
""")

BLOCK = 128
tile_bytes = BLOCK * BLOCK * BYTES_BF16
n_tiles    = (T // BLOCK) ** 2

print(f"  With a {BLOCK} x {BLOCK} block:")
print(f"    live score memory per block   {tile_bytes/2**10:>10,.0f} KiB")
print(f"    blocks that must be visited   {n_tiles:>10,}")
print(f"    score entries computed        {T*T:>10.3e}   (unchanged)")
print(f"""
  Peak score storage falls from {score_per_head/TiB:.1f} TiB to {tile_bytes/2**10:.0f} KiB per head - a
  reduction of about {score_per_head/tile_bytes:,.0f}x. Every one of the {T*T:.2e} score entries
  is still computed. Nothing was skipped; the results were consumed and thrown
  away instead of stored.
""")


# ---------------------------------------------------------------------------
rule("THE LEDGER")
print(f"""
  cost                        before tiling      after tiling      removed?
  --------------------------------------------------------------------------
  A  score storage/head       {score_per_head/TiB:>9.1f} TiB   {tile_bytes/2**10:>9.0f} KiB   YES
  B  attention FLOPs          {flops/1e18:>9.2f} EFLOP {flops/1e18:>9.2f} EFLOP  NO
  C  KV cache                 {kv_cache/TB:>9.2f} TB    {kv_cache/TB:>9.2f} TB     NOT ADDRESSED

  Tiling is an answer to exactly one row.
""")

assert abs(score_per_head / TiB - 2.0) < 1e-9
assert abs(flops / 1e18 - 5.026) < 0.01
# ---------------------------------------------------------------------------
# The distinction the taxonomy above was missing.
#
# The score matrix is TRANSIENT: it exists during one forward pass and is gone.
# The KV cache is PERSISTENT: it must survive every token the model generates,
# because each new token reads the whole thing back.
#
# Tiling is a kernel optimisation, so it can only ever touch the transient half.
# At K3's declared context the transient wall is the SMALLER of the two, which
# is why removing it does not make a million tokens servable.

ratio      = kv_cache / score_per_head
parity_at  = kv_per_token / 2      # where T*T*2 == T*per_token

print(f"""
========================================================================
TRANSIENT AGAINST PERSISTENT
========================================================================

  score matrix, one head   {score_per_head/TB:8.2f} TB   transient, one forward pass
  KV cache, one sequence   {kv_cache/TB:8.2f} TB   persistent, re-read every token
  {'-'*24} {'-'*11}
  the wall tiling leaves is {ratio:.2f}x the wall it removes

  The two are equal only at {parity_at:,.0f} tokens, which is {parity_at/T:.2f}x
  K3's declared window. Below that point, the cost a kernel cannot touch is
  always the larger one.
""")

assert abs(kv_cache / TB - 4.79) < 0.01
assert 2.1 < ratio < 2.3, "transient/persistent ratio moved"
assert 2_280_000 < parity_at < 2_290_000, "parity point moved"


# ---------------------------------------------------------------------------
rule("WHY ROW C IS THE ONE THAT DECIDES THE ARCHITECTURE")
print(f"""
  Row A and row B are prefill costs - the one-time pass over a long prompt.
  Row C is a per-token cost that persists for the whole conversation, and it
  gets worse in a way the other two do not: to emit each new token, attention
  must read the entire cache back. At 1M context that is a {kv_cache/TB:.1f} TB read per
  generated token.

  So a system can adopt the best tiled kernel available and still be unable to
  serve this context, because the wall it hits is the cache it must stream, not
  the matrix it declined to store.

  This derivation does not identify which architectural mechanism should reduce
  row C for a real model. That requires separate released architectural evidence;
  it cannot be inferred from this dense counterfactual alone.
""")


# ---------------------------------------------------------------------------
rule("RESULT")
print(f"""
  Tiling cut peak score storage by a factor of about {score_per_head/tile_bytes/1e6:,.0f} million.
  It cut the arithmetic by nothing at all.

  Avoiding materialization is not avoiding computation.

  A kernel optimization changes where a cost is paid. An architecture change is
  what removes one.

  All assertions passed.
""")
