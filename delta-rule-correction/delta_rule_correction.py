"""
Kimi K3 Systems Lab - Experiment 01: delta-rule correction
==========================================

Claim under test
----------------
An additive linear-attention memory cannot correct a fact. Writing a new value
under a key it has already seen leaves it answering with the SUM of the old and
new values. The delta rule writes the ERROR instead of the value, and answers
with the current value.

Evidence class : Teaching miniature (Tier 1, CPU)
Provenance     : Ankit run. The delta rule is Widrow & Hoff (1960); its use as a
                 linear-attention update is DeltaNet (Schlag, Irie & Schmidhuber,
                 2021), and the gated form is Gated DeltaNet (Yang et al., 2024).

Scope boundary
--------------
This is a 4x4 single-head miniature with hand-chosen unit-norm keys. It
demonstrates a WRITE-POLICY property of the two update rules. It establishes
NOTHING about Kimi K3's accuracy, throughput, latency, or quality. No trained
weights are involved.

Dependencies
------------
Python 3 standard library only. No numpy, no torch, no GPU.
Written in plain Python so the result can be reproduced by anyone in about two
seconds with no install.

Run
---
    python3 delta_rule_correction.py
"""

D_K = D_V = 4


# ---------------------------------------------------------------- primitives

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def outer(v, k):
    """v k^T  ->  (d_v, d_k) rank-1 matrix."""
    return [[vi * kj for kj in k] for vi in v]


def matvec(S, x):
    """S @ x  ->  (d_v,).  S is (d_v, d_k), x is (d_k,)."""
    return [sum(row[j] * x[j] for j in range(len(x))) for row in S]


def add_scaled(S, M, beta):
    """S + beta * M, elementwise."""
    return [[S[i][j] + beta * M[i][j] for j in range(len(S[0]))]
            for i in range(len(S))]


# ------------------------------------------------------------- the two rules

def linear_attn_step(S, k, v):
    """Additive linear attention: store the value itself."""
    return add_scaled(S, outer(v, k), 1.0)


def delta_step(S, k, v, beta):
    """Delta rule: predict, take the error, store only the error."""
    v_hat = matvec(S, k)                       # what memory currently answers
    error = [v[i] - v_hat[i] for i in range(len(v))]
    return add_scaled(S, outer(error, k), beta)


def read(S, q):
    return matvec(S, q)


# ------------------------------------------------------------------ printing

def fmt(vec):
    return "[" + ", ".join(f"{x:6.2f}" for x in vec) + "]"


def show_state(label, S):
    print(f"  {label}")
    for row in S:
        print("    " + " ".join(f"{x:6.2f}" for x in row))


def rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------------ scenario

#  A single unit-norm key standing for the entity "max_batch".
#  The scenario is a model reading a runbook: a setting is stated, then changed.
#  This illustration is our own construction. The MECHANISM it demonstrates is
#  the published delta rule, cited above.
#  Two different values for it, arriving in order.
k_setting = [1.0, 0.0, 0.0, 0.0]      # "max_batch"      (unit norm: |k| = 1)
v_first   = [8.0, 0.0, 0.0, 0.0]      # "max_batch is 8"
v_second  = [20.0, 0.0, 0.0, 0.0]     # "max_batch was raised to 20"


def scenario_correction():
    rule("SCENARIO 1 - a setting is changed:  'max_batch is 8' then 'raised to 20'")

    print("\nAdditive linear attention  ->  S = S + v k^T")
    S = zeros(D_V, D_K)
    S = linear_attn_step(S, k_setting, v_first)
    show_state("after writing 8:", S)
    S = linear_attn_step(S, k_setting, v_second)
    show_state("after writing 20:", S)
    additive = read(S, k_setting)
    print(f"\n  query 'max_batch' -> {fmt(additive)}   <-- answers {additive[0]:.1f}")
    print("  8 and 20 were both written. Neither was ever 28.")

    print("\nDelta rule  ->  S = S + beta * (v - S k) k^T,  beta = 1.0")
    S = zeros(D_V, D_K)
    S = delta_step(S, k_setting, v_first, beta=1.0)
    show_state("after writing 8:", S)
    v_hat = matvec(S, k_setting)
    err = [v_second[i] - v_hat[i] for i in range(D_V)]
    print(f"    predicted for this key: {fmt(v_hat)}")
    print(f"    error it writes:        {fmt(err)}   <-- 20 - 8 = 12, not 20")
    S = delta_step(S, k_setting, v_second, beta=1.0)
    show_state("after writing 20:", S)
    delta = read(S, k_setting)
    print(f"\n  query 'max_batch' -> {fmt(delta)}   <-- answers {delta[0]:.1f}")

    assert abs(additive[0] - 28.0) < 1e-9, "additive rule should answer 28.0"
    assert abs(delta[0] - 20.0) < 1e-9, "delta rule should answer 20.0"
    return additive[0], delta[0]


def scenario_repetition(n=5):
    rule(f"SCENARIO 2 - the same fact is simply repeated {n} times")
    print("\n  Natural text repeats entities constantly. Nothing is being corrected here.\n")
    print(f"  {'writes':>8} | {'additive':>10} | {'delta':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}")

    S_lin, S_del = zeros(D_V, D_K), zeros(D_V, D_K)
    last = (0.0, 0.0)
    for i in range(1, n + 1):
        S_lin = linear_attn_step(S_lin, k_setting, v_first)
        S_del = delta_step(S_del, k_setting, v_first, beta=1.0)
        a, d = read(S_lin, k_setting)[0], read(S_del, k_setting)[0]
        print(f"  {i:>8} | {a:>10.2f} | {d:>8.2f}")
        last = (a, d)

    print("\n  Additive memory inflates the value it already holds, once per mention.")
    print("  The delta rule's error is zero after the first write, so it writes nothing.")
    assert abs(last[0] - 8.0 * n) < 1e-9
    assert abs(last[1] - 8.0) < 1e-9
    return last


def scenario_beta():
    rule("SCENARIO 3 - beta is a write strength, not a switch")
    print("\n  In KDA beta is produced per token by a learned projection through a")
    print("  sigmoid, so it lies strictly between 0 and 1. The model chooses, token")
    print("  by token, how hard to overwrite. Endpoints shown for the mechanism only.\n")
    print(f"  {'beta':>6} | {'answer after correction':>24}")
    print(f"  {'-'*6}-+-{'-'*24}")
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        S = zeros(D_V, D_K)
        S = delta_step(S, k_setting, v_first, beta=1.0)
        S = delta_step(S, k_setting, v_second, beta=beta)
        print(f"  {beta:>6.2f} | {read(S, k_setting)[0]:>24.2f}")
    print("\n  beta = 0 keeps the old fact. beta = 1 replaces it exactly.")
    print("  Exact replacement at beta = 1 depends on the key being unit-norm.")


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())
    additive, delta = scenario_correction()
    scenario_repetition()
    scenario_beta()

    rule("RESULT")
    print(f"""
  additive linear attention answered  {additive:.1f}
  delta rule answered                 {delta:.1f}

  Both memories are the same size: {D_V} x {D_K} numbers. The difference is not
  capacity. It is the write policy.

  A memory that can only add cannot correct itself.

  All assertions passed.
""")
