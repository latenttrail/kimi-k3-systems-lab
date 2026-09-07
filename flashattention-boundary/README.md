# FlashAttention removes the storage wall, not the work

**Evidence class:** Derived · **Cost tier:** 0 (arithmetic) · **Provenance:** my own derivation
**Companion to [the million-token attention derivation](../million-token-attention/).** Same
released inputs, same hypothetical dense model.

## The claim

Tiling removes the storage cost of the T × T score matrix. It does not remove the arithmetic, and
it does not touch the KV cache at all. Of the three walls a million-token dense context runs into,
tiling removes exactly one — and it is not the one that binds during serving.

```
$ python3 flashattention_boundary.py

  cost                        before tiling      after tiling      removed?
  --------------------------------------------------------------------------
  A  score storage/head             2.0 TiB          32 KiB   YES
  B  attention FLOPs               5.03 EFLOP      5.03 EFLOP  NO
  C  KV cache                      4.79 TB         4.79 TB     NOT ADDRESSED
```

Peak score storage falls by a factor of about 67 million. Every one of the 1.1 × 10¹² score
entries is still computed — nothing was skipped, the results were consumed and discarded instead
of stored.

## Why row C is the one that decides the architecture

Rows A and B are prefill costs, paid once over a long prompt. Row C is paid again for every token
generated, because the whole cache is read back to emit each one.

So a system can adopt the best tiled kernel available and still be unable to serve the context —
blocked by the cache it must stream, not by the matrix it declined to store.

## Boundary: no runtime claim is made here

This needs saying plainly, because the shape of the argument invites the opposite reading.

**A tiled attention kernel is genuinely, substantially faster in practice**, largely because it
moves far less data between HBM and on-chip memory. That effect is real, it is the main reason
FlashAttention matters, and this derivation does not measure it. Nothing here should be read as
"FlashAttention does not help."

What the derivation shows is narrower and harder to argue with: **the arithmetic floor is
invariant under tiling.** Wall-clock time is not FLOPs.

The 128 × 128 block size is a conventional illustrative choice, not a released parameter. The
order of the reduction, not the block size, is the claim.

## Run it

```bash
python3 flashattention_boundary.py
```

Standard library only. The distinction follows from Dao et al., [*FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135): it is an
exact tiled attention algorithm designed to reduce traffic between HBM and on-chip SRAM. This
experiment does not benchmark that speedup.
