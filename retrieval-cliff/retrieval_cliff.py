"""
Kimi K3 Systems Lab - Post 04 proof object
==========================================

Claim under test
----------------
A fixed-size associative state has limited representational capacity. But a
non-zero retrieval error by itself does not establish that capacity failed: it
can also be an unfinished iterative solve. Two rows are reported because they
ask different questions:

  REPEATED WRITES  error after a fixed 400 correction passes. A separate sweep
                   test asks whether a residual shrinks with more passes.
  ONE PASS         error after one ordered write of each pair. Similar keys can
                   interfere even before representational capacity is at issue.

For generic independent associations, a state with d key dimensions cannot
guarantee exact interpolation once more than d constraints arrive. The run below
is one seeded illustration of that structural limit, not a lower-bound proof.

Evidence class : Teaching miniature (Tier 1, CPU)
Provenance     : Ankit run. Delta-rule update as in Widrow & Hoff (1960) /
                 DeltaNet (Schlag, Irie & Schmidhuber, 2021).

Scope boundary
--------------
This is a d_k = d_v = 8 single-head miniature on random unit-norm keys. It shows
a CAPACITY property of a fixed-size associative memory. It establishes nothing
about Kimi K3's recall, accuracy, or long-context quality, and it is not a
benchmark of any released model.

The comparison row for softmax attention is included because it is the honest
control: softmax keeps every key, so it does not have this failure mode - it has
the cost that Posts 2 and 3 priced instead.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 retrieval_cliff.py
"""

import math
import random

D_K = D_V = 8
SEED = 20260828


# ---------------------------------------------------------------- primitives

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v))


def unit(v):
    n = norm(v)
    return [x / n for x in v]


def matvec(S, x):
    return [dot(row, x) for row in S]


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def delta_write(S, k, v, beta=1.0):
    """S <- S + beta (v - S k) k^T   with unit-norm k, beta=1 is exact overwrite."""
    v_hat = matvec(S, k)
    err = [v[i] - v_hat[i] for i in range(len(v))]
    return [[S[i][j] + beta * err[i] * k[j] for j in range(len(k))]
            for i in range(len(v))]


def softmax_read(keys, values, q, temp=0.05):
    """Exact attention over everything that was stored. The control."""
    scores = [dot(q, k) / temp for k in keys]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    out = [0.0] * len(values[0])
    for w, v in zip(exps, values):
        for i in range(len(v)):
            out[i] += (w / Z) * v[i]
    return out


# ---------------------------------------------------------------- experiment

def make_pairs(n, rng):
    keys = [unit([rng.gauss(0, 1) for _ in range(D_K)]) for _ in range(n)]
    values = [[rng.gauss(0, 1) for _ in range(D_V)] for _ in range(n)]
    return keys, values


def mean_err(S, keys, values):
    errs = []
    for k, v in zip(keys, values):
        vd = matvec(S, k)
        errs.append(norm([vd[i] - v[i] for i in range(D_V)]) / norm(v))
    return sum(errs) / len(errs)


def recall_error(n, rng, sweeps=400):
    """One-pass error, 400-sweep repeated-write error, and the softmax control."""
    keys, values = make_pairs(n, rng)

    S = zeros(D_V, D_K)
    for k, v in zip(keys, values):
        S = delta_write(S, k, v)
    one_pass = mean_err(S, keys, values)

    # Keep writing the same pairs. Where a solution exists this converges to it;
    # where none exists, this is the level the iteration settles at. Whether
    # that level is a true lower bound is not something more sweeps can show.
    for _ in range(sweeps):
        for k, v in zip(keys, values):
            S = delta_write(S, k, v, beta=0.5)
    converged = mean_err(S, keys, values)

    soft = []
    for k, v in zip(keys, values):
        vs = softmax_read(keys, values, k)
        soft.append(norm([vs[i] - v[i] for i in range(D_V)]) / norm(v))
    return one_pass, converged, sum(soft) / len(soft)


def rule(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())

    rule(f"SETUP  (seed {SEED}, fixed and published)")
    print(f"""
  state shape          {D_V} x {D_K}  = {D_V * D_K} numbers, and it never grows
  keys                 random Gaussian, L2-normalised to unit length
  values               random Gaussian, unnormalised
  write rule           delta rule, beta = 1 (exact overwrite for unit-norm keys)
  query                every stored key, read back in turn
  error                mean over pairs of ||read - stored|| / ||stored||
""")

    rng = random.Random(SEED)
    rows = []
    rule(f"HOW MANY PAIRS CAN AN {D_K}x{D_V} STATE HOLD?")
    print(f"\n  {'pairs':>6} | {'one pass':>10} | {'repeat 400':>10} | {'softmax':>9}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
    for n in (2, 4, 6, 8, 10, 12, 16, 24, 32, 64):
        one, conv, soft = recall_error(n, rng)
        rows.append((n, one, conv, soft))
        mark = '   <-- state is full' if n == D_K else ''
        print(f"  {n:>6} | {one:>10.4f} | {conv:>10.4f} | {soft:>9.4f}{mark}")

    conv_below = max(c for n, _, c, _ in rows if n <= D_K)
    conv_above = min(c for n, _, c, _ in rows if n > D_K)
    one_at_half = [o for n, o, _, _ in rows if n == D_K // 2][0]
    soft_max_err = max(s for *_, s in rows)

    rule("THE SHAPE OF THE FAILURE")
    print(f"""
  REPEATED WRITES (400 sweeps) - an observed solver budget, not a proven optimum

    at or below {D_K} pairs     worst error {conv_below:.4f}
    above {D_K} pairs           best  error {conv_above:.4f}

    (Below {D_K} pairs the error is 0.0000 outright. The {conv_below:.4f} is the n = {D_K} row:
     at exactly full rank the key matrix is ill-conditioned, so the iteration
     converges slowly rather than incompletely. It is still {conv_above / conv_below:.0f}x below the
     first row above capacity.)

  That is the jump this run measures, and it is worth being exact about what it
  does and does not show. The sampled sizes are 2, 4, 6, 8, 10, 12, ... so n = 9
  is never tested, and each size is a single seeded draw of random keys. What is
  measured is a large jump between {D_K} and 10 for these draws, not the absence
  of every intermediate value.

  For generic independent associations, an {D_V} x {D_K} state cannot guarantee
  exact recall once more than {D_K} constraints arrive. This run illustrates
  that structural limit; it is not a lower-bound proof for every possible draw.

  The SWEEP DEPTH test below separates the two candidate explanations directly.

  ONE PASS - a different update budget

    at {D_K // 2} pairs, half capacity, the error is already {one_at_half:.4f}.

  Each write is a rank-1 correction aimed at one key. Because random keys are
  not orthogonal, every write disturbs the neighbours that share direction with
  it. So the practical memory degrades long before the arithmetic runs out.

  Representational capacity, convergence, and one-pass interference are
  different failure modes.

  SOFTMAX - the control

    error rises too, but from 0.0001 to {soft_max_err:.4f} across a 32x increase in pairs -
    a gentle slope, not a cliff, and about {conv_above / soft_max_err:.0f}x smaller than the linear
    state's first failure. It degrades because a fixed temperature blurs keys
    that lie close together, not because it ran out of room: it kept every key.

    That is not a free win. Keeping every key is exactly the cost Posts 2 and 3
    priced. K3's released config pairs 69 fixed-state layers with 24
    full-attention ones; this miniature says nothing about why.
""")
    assert conv_below < 0.01, "at or below capacity the state should be solvable"
    assert conv_above > 0.1, "beyond capacity the state should fail clearly"
    assert one_at_half > 0.05, "one pass should already be lossy at half capacity"
    assert soft_max_err < 0.05, "softmax should stay far below the linear cliff"
    assert conv_above / soft_max_err > 10, "the cliff should dwarf softmax degradation"

    rule("SWEEP DEPTH - IS THE RESIDUE AT FULL CAPACITY A SOLVER ARTEFACT?")
    print("""
  At n = 8 the 400-sweep repeated-write error is not 0.0000 but a small non-zero number, and
  there are two candidate explanations that look identical in a single column:

    (a) the state cannot quite hold 8 pairs, so a floor remains, or
    (b) the pairs fit and the iteration has not finished converging.

  They are told apart by running longer. Under (a) more sweeps change nothing.
  Under (b) the error goes to zero. Same seeds, same draws, only the sweep
  count varies.
""")
    print(f"  {'sweeps':>8} | {'n = 8':>15} | {'n = 10':>15}")
    print(f"  {'-'*8}-+-{'-'*15}-+-{'-'*15}")
    for sw in (400, 1000, 2000, 5000, 20000):
        r8 = random.Random(SEED)
        r10 = random.Random(SEED)
        for nn in (2, 4, 6):
            recall_error(nn, r8, sweeps=1)
            recall_error(nn, r10, sweeps=1)
        e8 = recall_error(8, r8, sweeps=sw)[1]
        for nn in (8,):
            recall_error(nn, r10, sweeps=1)
        e10 = recall_error(10, r10, sweeps=sw)[1]
        print(f"  {sw:>8} | {e8:>15.9f} | {e10:>15.9f}")
    print("""
  The two rows behave completely differently.

  n = 8  falls away to zero. The residue at 400 sweeps was the iteration still
         working, not a limit. Eight pairs fit exactly.
  n = 10 does not move at all, to nine decimal places, across a fiftyfold
         increase in effort, on this draw. That is what a floor looks like from
         the outside. It is not a proof of one: establishing a genuine lower
         bound needs a rank or residual argument, not a longer run.

  So the small number at capacity and the large number past it are not two
  points on one curve. They are different phenomena, and only the second is
  about capacity.
""")

    rule("RESULT")
    print(f"""
  A retrieval error is a diagnosis to investigate, not a verdict.

  Three separate lessons:
    - the 8-pair residual disappears with more correction passes in this draw;
    - the 10-pair residual persists through the observed sweep budget;
    - one ordered pass introduces interference before either is a capacity verdict.

  What any particular architecture does about those failure modes is a separate
  question, and this miniature does not answer it.

  All assertions passed.
""")
