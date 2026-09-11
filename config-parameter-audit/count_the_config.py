"""
Kimi K3 parameter-count derivation
==================================

Claim under test
----------------
A config file is a blueprint, not an inventory - and the difference is one key
you have not noticed yet.

Counting Kimi K3's parameters the obvious way gives 5.47 trillion against a
published 2.8 trillion: nearly double. The gap is not an arithmetic error. It is
a released architectural distinction: the routed experts use a 3,584-wide latent
space rather than the model's 7,168-wide hidden state.

Evidence class : Released + Derived (Tier 0)
Provenance     : Ankit derivation. Config read first-hand from
                 https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
                 Architecture mapping read first-hand from Moonshot AI's Kimi K3
                 Technical Report, §2.3:
                 https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf

Scope boundary
--------------
This counts the model the config DESCRIBES. It is not a tensor inventory of the
released checkpoint - no shard was opened. Where the two disagree, the checkpoint
wins.

Dependencies
------------
Python 3 standard library only.

Run
---
    python3 count_the_config.py
"""

# ---------------------------------------------------------------------------
# RELEASED - every value below was read from config.json directly.
# ---------------------------------------------------------------------------
HIDDEN               = 7168      # text_config.hidden_size
LAYERS               = 93        # num_hidden_layers
VOCAB                = 163840    # vocab_size
HEADS                = 96        # num_attention_heads
HEAD_DIM             = 128       # linear_attn_config.head_dim
KV_LORA_RANK         = 512       # kv_lora_rank
Q_LORA_RANK          = 1536      # q_lora_rank
QK_NOPE              = 128       # qk_nope_head_dim
QK_ROPE              = 64        # qk_rope_head_dim
V_HEAD_DIM           = 128       # v_head_dim
DENSE_INTERMEDIATE   = 33792     # intermediate_size
FIRST_K_DENSE        = 1         # first_k_dense_replace
N_FULL_ATTN          = 24        # len(linear_attn_config.full_attn_layers)
N_KDA                = 69        # len(linear_attn_config.kda_layers)

N_EXPERTS            = 896       # num_experts
N_ACTIVE_EXPERTS     = 16        # num_experts_per_token
N_SHARED_EXPERTS     = 2         # num_shared_experts
MOE_INTERMEDIATE     = 3072      # moe_intermediate_size
ROUTED_EXPERT_HIDDEN = 3584      # routed_expert_hidden_size   <-- the key

TOTAL_PARAMS_PUB  = 2.8e12
B, T = 1e9, 1e12


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def embeddings():
    return 2 * VOCAB * HIDDEN                      # tie_word_embeddings: false


def mla_attention():
    return (HIDDEN * Q_LORA_RANK
            + Q_LORA_RANK * HEADS * (QK_NOPE + QK_ROPE)
            + HIDDEN * (KV_LORA_RANK + QK_ROPE)
            + KV_LORA_RANK * HEADS * (QK_NOPE + V_HEAD_DIM)
            + HEADS * V_HEAD_DIM * HIDDEN)


def kda_attention():
    return mla_attention()          # ASSUMED: comparable projection budget


def dense_mlp():
    return 3 * HIDDEN * DENSE_INTERMEDIATE


def expert(width):
    return 3 * width * MOE_INTERMEDIATE            # gate, up, down


if __name__ == "__main__":
    print(__doc__.split("Run\n---")[0].strip())
    n_moe = LAYERS - FIRST_K_DENSE

    rule("STEP 1 - COUNT IT THE OBVIOUS WAY")
    emb, attn = embeddings(), N_FULL_ATTN * mla_attention() + N_KDA * kda_attention()
    dense = FIRST_K_DENSE * dense_mlp()
    router = n_moe * HIDDEN * N_EXPERTS
    naive_experts = n_moe * (N_EXPERTS + N_SHARED_EXPERTS) * expert(HIDDEN)
    naive = emb + attn + dense + router + naive_experts

    print(f"""
  Every expert an MLP over the {HIDDEN:,}-wide residual stream, which is what an
  expert normally is:

    {N_EXPERTS} routed + {N_SHARED_EXPERTS} shared, each 3 x {HIDDEN:,} x {MOE_INTERMEDIATE:,}
    x {n_moe} MoE layers

    embeddings          {emb/B:>10.2f} B
    attention           {attn/B:>10.2f} B
    dense MLP           {dense/B:>10.2f} B
    routers             {router/B:>10.2f} B
    experts             {naive_experts/B:>10.2f} B
    {'-'*20} {'-'*12}
    TOTAL               {naive/T:>10.2f} T     published: {TOTAL_PARAMS_PUB/T:.1f} T

  Off by {(naive - TOTAL_PARAMS_PUB)/TOTAL_PARAMS_PUB:+.1%}. Not a rounding error - nearly double.
""")
    assert naive > TOTAL_PARAMS_PUB * 1.8

    rule("STEP 2 - THE KEY I HAD NOT NOTICED")
    print(f"""
  routed_expert_hidden_size: {ROUTED_EXPERT_HIDDEN}

  The residual stream is {HIDDEN:,} wide. The routed experts are {ROUTED_EXPERT_HIDDEN:,}.

  The Kimi K3 technical report says the routed branch projects the full-width
  input into a latent, dispatches that latent to selected routed experts, then
  projects their aggregate back. The shared experts process the full-width input.

  That single key halves the largest term in the count.
""")

    rule("STEP 3 - COUNT IT AGAIN")
    routed = n_moe * N_EXPERTS * expert(ROUTED_EXPERT_HIDDEN)
    shared = n_moe * N_SHARED_EXPERTS * expert(HIDDEN)
    total = emb + attn + dense + router + routed + shared
    gap = (total - TOTAL_PARAMS_PUB) / TOTAL_PARAMS_PUB
    print(f"""
    embeddings                    {emb/B:>10.2f} B
    attention ({N_FULL_ATTN} MLA + {N_KDA} KDA)     {attn/B:>10.2f} B
    dense MLP                     {dense/B:>10.2f} B
    routers                       {router/B:>10.2f} B
    routed experts ({ROUTED_EXPERT_HIDDEN:,} wide)   {routed/B:>10.2f} B
    shared experts ({HIDDEN:,} wide)    {shared/B:>10.2f} B
    {'-'*29} {'-'*12}
    TOTAL                         {total/T:>10.3f} T

    published                     {TOTAL_PARAMS_PUB/T:>10.3f} T
    gap                           {gap*100:>+10.1f} %

  That lands within a couple of per cent of the published figure, from public
  values and arithmetic alone. This is still an estimate, not a checkpoint tensor
  inventory: the KDA projection budget is an explicit approximation and other
  unmodelled components can contribute to the remaining difference.
""")
    assert abs(gap) < 0.06, "the corrected count should land close to the published total"

    rule("RESULT")
    print(f"""
  First count: {naive/T:.2f} T. Published: {TOTAL_PARAMS_PUB/T:.1f} T. Second count: {total/T:.2f} T.

  Nothing changed except that I read one more line of the config. The obvious
  count was not sloppy - it applied a different architecture to this model.

  That is what a config is. Not an inventory, and not self-explanatory: a set of
  keys where the one you skim past is the one carrying half the parameters.

  Count the instantiated model, not the architecture you meant to build.
""")
