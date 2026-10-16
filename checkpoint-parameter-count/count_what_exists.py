"""
Kimi K3 parameter count, from the released checkpoint
=====================================================

Claim under test
----------------
Two honest ways to count Kimi K3's parameters both miss by about 2x, in
opposite directions. Counting from the config with every expert at the model's
full width gives ~5.47 trillion. Summing the element counts of the released
checkpoint's tensors gives ~1.50 trillion. Moonshot AI's technical report says
2.78 trillion (Table 1).

The config count misses because the routed experts work in a 3,584-wide latent,
not the 7,168-wide hidden state. The checkpoint count misses because those
experts are stored as packed 4-bit values, two per byte, next to scale tensors
that are not parameters at all. Counting what the files actually hold, with
both facts applied, should land on the published figure.

Evidence class : Released + Derived (Tier 2: the released checkpoint is inspected)
Provenance     : Ankit derivation. Every tensor name, shape and dtype is read
                 from the safetensors headers of all 96 weight shards of
                 huggingface.co/moonshotai/Kimi-K3 at a pinned revision.
                 Published figures: Kimi K3 Technical Report, abstract and
                 Table 1 (https://github.com/MoonshotAI/Kimi-K3).

Scope boundary
--------------
Headers only. A safetensors file begins with a JSON header listing every
tensor; this script reads those bytes with HTTP range requests and never
downloads a weight. It therefore establishes what tensors exist and how they
are stored, not what values they hold. "Activated per token" follows one stated
convention (output head counted, input-embedding lookup not); the report does
not state its convention, so a match is consistency, not proof of method.

Dependencies
------------
Python 3 standard library only. First run fetches ~72 MB of headers into
.headers/ next to this script; later runs reuse them.
"""

import hashlib, json, math, re, struct, sys, urllib.request
from collections import defaultdict
from pathlib import Path

REPO = "moonshotai/Kimi-K3"
REV = "f831ab66814297da540d832a5235f8e904f29d06"   # pinned: last modified 2026-09-02
N_SHARDS = 96
CACHE = Path(__file__).resolve().parent / ".headers"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}

# RELEASED - Kimi K3 Technical Report
PUB_TOTAL_ABSTRACT, PUB_TOTAL_TABLE = 2.8e12, 2.78e12
PUB_ACTIVE_ABSTRACT, PUB_ACTIVE_TABLE = 104e9, 104.2e9
B, T = 1e9, 1e12


def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"


def get(name, start=None, end=None):
    req = urllib.request.Request(url(name), headers={"User-Agent": "kimi-k3-systems-lab"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def cached(name, fetch):
    p = CACHE / name
    if not p.exists():
        CACHE.mkdir(exist_ok=True)
        p.write_bytes(fetch())
    return p.read_bytes()


def shard_header(i):
    name = f"model-{i:05d}-of-{N_SHARDS:06d}.safetensors"
    def fetch():
        n = struct.unpack("<Q", get(name, 0, 7))[0]
        return get(name, 8, 7 + n)
    return name, cached(f"{i:05d}.json", fetch)


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


if __name__ == "__main__":
    print(__doc__.split("Dependencies\n------------")[0].strip())
    print(f"\nRepository {REPO} at revision {REV}")

    # ------------------------------------------------------------------ config
    cfg = json.loads(cached("config.json", lambda: get("config.json")))
    tc = cfg["text_config"]
    q = tc["quantization_config"]
    d, ell = tc["hidden_size"], tc["routed_expert_hidden_size"]
    E, K, F = tc["num_experts"], tc["num_experts_per_token"], tc["moe_intermediate_size"]
    S, L, DENSE = tc["num_shared_experts"], tc["num_hidden_layers"], tc["first_k_dense_replace"]
    MOE = L - DENSE

    # -------------------------------------------------------------- the headers
    tensors, digest = {}, hashlib.sha256()
    for i in range(1, N_SHARDS + 1):
        name, raw = shard_header(i)
        digest.update(raw)
        h = json.loads(raw); h.pop("__metadata__", None)
        spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in h.items())
        pos = 0
        for s, e, k in spans:                     # contiguous, no gaps, no overlaps
            assert s == pos, f"{name}: gap or overlap before {k}"
            v = h[k]
            assert e - s == math.prod(v["shape"]) * DTYPE_BYTES[v["dtype"]], f"{name}: size mismatch {k}"
            pos = e
            assert k not in tensors, f"duplicate tensor {k}"
            tensors[k] = v

    rule("WHAT THE 96 HEADERS CONTAIN")
    stored = sum(math.prod(v["shape"]) * DTYPE_BYTES[v["dtype"]] for v in tensors.values())
    header_bytes = sum((CACHE / f"{i:05d}.json").stat().st_size for i in range(1, N_SHARDS + 1))
    dt = defaultdict(int)
    for v in tensors.values():
        dt[v["dtype"]] += 1
    print(f"""
  tensors                         {len(tensors):>12,}
  by dtype                        {', '.join(f'{k} {v:,}' for k, v in sorted(dt.items()))}
  weight bytes described          {stored/T:>12.4f} TB
  header bytes read to learn it   {header_bytes/1e6:>12.1f} MB
  sha256 of all headers, in order {digest.hexdigest()[:16]}...

  Every shard's tensors tile its data section exactly: no gaps, no overlaps,
  and each tensor's byte span equals shape x dtype size.""")

    # ------------------------------------------------- count 1: the config alone
    def mla():
        return (d * tc["q_lora_rank"] + tc["q_lora_rank"] * tc["num_attention_heads"] * (tc["qk_nope_head_dim"] + tc["qk_rope_head_dim"])
                + d * (tc["kv_lora_rank"] + tc["qk_rope_head_dim"])
                + tc["kv_lora_rank"] * tc["num_attention_heads"] * (tc["qk_nope_head_dim"] + tc["v_head_dim"])
                + tc["num_attention_heads"] * tc["v_head_dim"] * d)
    n_full = len(tc["linear_attn_config"]["full_attn_layers"])
    n_kda = len(tc["linear_attn_config"]["kda_layers"])
    est = {
        "embeddings": 2 * tc["vocab_size"] * d,
        "attention (KDA guessed = MLA size)": (n_full + n_kda) * mla(),
        "dense MLP": DENSE * 3 * d * tc["intermediate_size"],
        "routers": MOE * d * E,
        "shared experts": MOE * S * 3 * d * F,
    }
    naive = sum(est.values()) + MOE * E * 3 * d * F
    latent = sum(est.values()) + MOE * E * 3 * ell * F + MOE * 2 * d * ell

    rule("COUNT 1 - FROM THE CONFIG")
    print(f"""
  Every routed expert at the model width, {d:,}:
    {E} x 3 x {d:,} x {F:,} x {MOE} MoE layers = {MOE*E*3*d*F/B:,.1f} B
    + everything else                       = {sum(est.values())/B:,.1f} B
    TOTAL                                   = {naive/T:.2f} T   ({naive/PUB_TOTAL_TABLE-1:+.0%} vs the report's 2.78 T)

  The config also says routed_expert_hidden_size = {ell:,}. Count the experts
  at that width, add the down/up projections into and out of it:
    TOTAL                                   = {latent/T:.3f} T   ({latent/PUB_TOTAL_TABLE-1:+.2%})

  That second count still had to guess one thing: the size of the {n_kda}
  linear-attention (KDA) layers, which the config does not spell out. It
  assumed they match the {n_full} standard (MLA) layers.""")
    assert naive > 1.9 * PUB_TOTAL_TABLE

    # --------------------------------------------- count 2: the files, naively
    numel = sum(math.prod(v["shape"]) for v in tensors.values())
    packed = sum(math.prod(v["shape"]) for k, v in tensors.items() if k.endswith("weight_packed"))
    scales = sum(math.prod(v["shape"]) for k, v in tensors.items() if k.endswith("weight_scale"))

    rule("COUNT 2 - FROM THE CHECKPOINT, NAIVELY")
    print(f"""
  Add up the element count of every tensor in the files:
    TOTAL                                   = {numel/T:.2f} T   ({numel/PUB_TOTAL_TABLE-1:+.0%} vs 2.78 T)

  The config's quantization_config says why that is wrong:
    format {q['format']}, {q['config_groups']['group_0']['weights']['num_bits']}-bit {q['config_groups']['group_0']['weights']['type']} weights,
    one {q['config_groups']['group_0']['weights']['scale_dtype'].split('.')[-1]} scale per group of {q['config_groups']['group_0']['weights']['group_size']}

    *.weight_packed  {packed/B:>9,.1f} B elements, uint8, each holding TWO 4-bit weights
    *.weight_scale   {scales/B:>9,.1f} B elements, uint8 - storage metadata, not parameters
    everything else  {(numel-packed-scales)/B:>9,.1f} B elements, one parameter each""")
    assert numel < 0.6 * PUB_TOTAL_TABLE

    # check the packing against the config, tensor by tensor
    gs = q["config_groups"]["group_0"]["weights"]["group_size"]
    logical = {"w1": (F, ell), "w3": (F, ell), "w2": (ell, F)}
    n_experts = 0
    for k, v in tensors.items():
        m = re.search(r"\.experts\.\d+\.(w[123])\.weight_(packed|scale)$", k)
        if not m:
            continue
        rows, cols = logical[m.group(1)]
        want = [rows, cols // 2] if m.group(2) == "packed" else [rows, cols // gs]
        assert v["shape"] == want and v["dtype"] == "U8", f"unexpected expert tensor {k} {v}"
        n_experts += m.group(2) == "packed"
    assert n_experts == MOE * E * 3, n_experts
    print(f"\n  All {n_experts:,} expert matrices have exactly the packed and scale shapes")
    print(f"  that {ell:,}-wide experts with a {F:,}-wide MLP imply. None is {d:,} wide.")

    # ------------------------------------------------ count 3: what exists
    def component(k):
        if k.startswith(("vision_tower", "mm_projector")): return "vision tower + projector"
        if ".experts." in k: return "routed experts"
        if "routed_expert_down_proj" in k or "routed_expert_up_proj" in k: return "latent down/up projections"
        if "shared_experts" in k: return "shared experts"
        if "block_sparse_moe.gate." in k: return "routers"
        if ".self_attn." in k: return "attention"
        if ".mlp." in k: return "dense MLP"
        if "embed_tokens" in k: return "input embedding"
        if "lm_head" in k: return "output head"
        return "norms and small gates"
    P, Bt = defaultdict(int), defaultdict(int)
    for k, v in tensors.items():
        n = math.prod(v["shape"]); c = component(k)
        Bt[c] += n * DTYPE_BYTES[v["dtype"]]
        if k.endswith("weight_scale"):
            continue
        P[c] += 2 * n if k.endswith("weight_packed") else n
    total = sum(P.values())
    lm = total - P["vision tower + projector"]

    rule("COUNT 3 - WHAT THE CHECKPOINT HOLDS")
    print(f"\n  {'component':<28}{'parameters':>14}{'share':>9}{'stored as':>14}")
    for c in sorted(P, key=lambda c: -P[c]):
        bits = 8 * Bt[c] / P[c]
        print(f"  {c:<28}{P[c]/B:>12,.3f} B{P[c]/total:>9.2%}{bits:>9.2f} bits")
    print(f"  {'-'*28}{'-'*14}")
    print(f"  {'TOTAL':<28}{total/B:>12,.3f} B")
    print(f"""
  report, Table 1                 2.78 T    difference {total/PUB_TOTAL_TABLE-1:+.3%}
  report, abstract (rounded)      2.8  T    difference {total/PUB_TOTAL_ABSTRACT-1:+.2%}
  language model alone            {lm/B:,.1f} B  ({lm/PUB_TOTAL_TABLE-1:+.2%} vs Table 1)""")
    assert abs(total / PUB_TOTAL_TABLE - 1) < 0.001, "the inventory should land on the published total"

    # activated: every non-routed language-model parameter, plus K of E experts.
    # Convention: the output head is computed for every token and counts; the
    # input embedding is a one-row lookup and does not.
    act = lm - P["routed experts"] + P["routed experts"] * K / E - P["input embedding"]
    rule("ACTIVATED PER TOKEN")
    print(f"""
  non-routed language-model parameters    {(lm - P['routed experts'] - P['input embedding'])/B:>9.2f} B   (excluding the input-embedding lookup)
  + {K} of {E} routed experts per layer      {P['routed experts']*K/E/B:>9.2f} B
  = activated per token                   {act/B:>9.2f} B

  report, Table 1                         104.2 B   difference {act/PUB_ACTIVE_TABLE-1:+.3%}
  Counting the input embedding too gives {(act + P['input embedding'])/B:.2f} B; excluding the output
  head as well gives {(act - P['output head'])/B:.2f} B. Only the stated convention matches.""")
    assert abs(act / PUB_ACTIVE_TABLE - 1) < 0.002

    # ------------------------------------------- where the config estimate missed
    file_attn = P["attention"]
    est_attn = est["attention (KDA guessed = MLA size)"]
    rule("WHERE THE CONFIG ESTIMATE MISSED")
    print(f"""
  config estimate (count 1, latent-aware)  {latent/B:>10,.1f} B
  checkpoint                               {total/B:>10,.1f} B
  shortfall                                {(total-latent)/B:>10,.1f} B

  attention, estimated                     {est_attn/B:>10,.1f} B   (KDA layers guessed to be MLA-sized)
  attention, in the files                  {file_attn/B:>10,.1f} B
  attention alone explains                 {(file_attn-est_attn)/(total-latent):>10.0%} of the shortfall

  The part of the estimate read from the config landed; the one part it had to
  guess is where it missed.""")
    assert (file_attn - est_attn) / (total - latent) > 0.9

    rule("RESULT")
    print(f"""
  From the config, experts at full width:   {naive/T:.2f} T
  From the checkpoint, elements summed:      {numel/T:.2f} T
  From the checkpoint, read correctly:       {total/T:.4f} T   (report: 2.78 T)
  Activated per token, stated convention:    {act/B:.2f} B   (report: 104.2 B)

  Both wrong counts were exact arithmetic. One missed how wide the experts
  are; the other missed how the file stores them. The config is a blueprint
  and the quantization config is a storage format: the count needs both.

  Count the instantiated model, not the architecture you meant to build.
""")
