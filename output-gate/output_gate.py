"""
Kimi K3 Systems Lab - Post 06 proof object
==========================================

Claim under test
----------------
Softmax attention cannot abstain. Its weights are non-negative and sum to one,
so its output is always a weighted average of the values it is attending to -
and a weighted average of non-zero vectors is not zero. Whatever the question,
the layer must return something.

That is a representational limit, not a training problem. No choice of query,
key or temperature makes a softmax layer output nothing.

An output gate removes the limit: multiply the layer's result by a learned
per-channel sigmoid, and the layer can scale its own contribution to zero.

Evidence class : Teaching miniature (Tier 1, CPU)
Provenance     : Ankit run.

Scope boundary
--------------
A d = 32 single-head miniature with random values, no trained weights. It
demonstrates a property of the softmax function and of a gated output, not the
behaviour of any released model. Kimi K3's config lists 24 full-attention
layers described as gated; the exact form of that gate is not established here.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 output_gate.py
"""

import math
import random

D = 32
SEED = 20260828
TRIALS = 2000


def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(v): return math.sqrt(dot(v, v))
def unit(v):
    n = norm(v)
    return [x / n for x in v]


def softmax(scores):
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    Z = sum(e)
    return [x / Z for x in e]


def attend(q, keys, values, temp):
    w = softmax([dot(q, k) / temp for k in keys])
    out = [0.0] * len(values[0])
    for wi, v in zip(w, values):
        for i in range(len(v)):
            out[i] += wi * v[i]
    return out, w


def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())
    rng = random.Random(SEED)

    n_keys = 12
    keys = [unit([rng.gauss(0, 1) for _ in range(D)]) for _ in range(n_keys)]
    values = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n_keys)]
    mean_val_norm = sum(norm(v) for v in values) / n_keys

    # ------------------------------------------------------------------
    rule("TRY TO MAKE SOFTMAX ATTENTION OUTPUT NOTHING")
    print(f"""
  {n_keys} keys, {D} dimensions, random values with mean norm {mean_val_norm:.2f}.
  {TRIALS:,} random queries per temperature. Temperature is swept across four
  orders of magnitude, which is the only lever a query has over the weights.
""")
    print(f"  {'temperature':>12} | {'min ||out||':>12} | {'mean ||out||':>13} | {'weights sum':>12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*13}-+-{'-'*12}")

    overall_min = float('inf')
    for temp in (0.01, 0.1, 1.0, 10.0, 100.0):
        norms, wsums = [], []
        for _ in range(TRIALS):
            q = unit([rng.gauss(0, 1) for _ in range(D)])
            out, w = attend(q, keys, values, temp)
            norms.append(norm(out))
            wsums.append(sum(w))
        overall_min = min(overall_min, min(norms))
        print(f"  {temp:>12.2f} | {min(norms):>12.4f} | {sum(norms)/len(norms):>13.4f} | {sum(wsums)/len(wsums):>12.6f}")

    print(f"""
  The weights sum to 1.000000 at every temperature - that is what softmax is.
  Across {TRIALS * 5:,} queries the smallest output norm seen was {overall_min:.4f}.

  It never reaches zero, and it cannot: the output is a convex combination of
  the value vectors, so it is trapped inside their convex hull. Zero is outside
  that hull unless the values happen to surround the origin exactly.
""")
    assert overall_min > 0.01, "softmax output should never collapse to zero"

    # ------------------------------------------------------------------
    rule("ONE MULTIPLICATION REMOVES THE LIMIT")
    print("""
  Add a per-channel output gate: out <- sigmoid(g) * attention_out.
  The gate is a learned scalar per channel; here it is swept directly.
""")
    q = unit([rng.gauss(0, 1) for _ in range(D)])
    base, _ = attend(q, keys, values, 1.0)
    base_norm = norm(base)

    print(f"  {'gate pre-activation':>20} | {'sigmoid':>9} | {'||out||':>10} | {'vs ungated':>12}")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*10}-+-{'-'*12}")
    for g in (4.0, 0.0, -2.0, -5.0, -8.0, -12.0):
        s = sigmoid(g)
        n = base_norm * s
        print(f"  {g:>20.1f} | {s:>9.6f} | {n:>10.6f} | {base_norm/n:>11.1f}x")

    collapse = base_norm / (base_norm * sigmoid(-12.0))
    print(f"""
  At a gate pre-activation of -12 the layer's contribution is {collapse:,.0f}x smaller
  than ungated. The layer has, in effect, said nothing for this token.

  Note what did NOT change: the attention weights still sum to 1, and the layer
  still computed its average. The gate does not stop the work. It decides
  whether the result is allowed to reach the residual stream.
""")
    assert collapse > 1000, "a closed gate should collapse the contribution by orders of magnitude"

    rule("RESULT")
    print(f"""
  Softmax must answer. Its floor across {TRIALS * 5:,} queries was {overall_min:.4f}, never zero.
  A gated output can decline, collapsing its contribution by {collapse:,.0f}x.

  The design consequence is not about attention quality. It is that a system
  forced to answer every time has no way to express "nothing here is relevant" -
  and in a deep stack, a layer with nothing to add is better off adding nothing
  than adding its best guess.

  A system that must answer needs a way to abstain.

  All assertions passed.
""")
