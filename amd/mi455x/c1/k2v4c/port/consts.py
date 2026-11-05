"""Fixed constants of the gfx1250 TP4/EP1 K2 port (owned by the lead; see INTERFACE.md).

Every port module imports grid / model / TP constants from here (``from port.consts import ...``) so the
helpers never disagree. Change only through the lead.
"""

# ---- device / grid (gfx1250, MI455X)
ARCH = "gfx1250"
CUS = 256
BLOCKS = 256  # one persistent CTA per CU, all co-resident
LANES = 32  # wave32 (P1 calls it ops.WAVE)
THREADS = 512  # as MI355X: thread-strided loops carry over
WAVES = THREADS // LANES  # 16 (MI355X: 8)
LDS_BYTES = 160 * 1024  # what the K2 may use per CTA (the CU has 320 KB)

# ---- tensor / expert parallel
TP = 4
EP = 1

# ---- model (Muse Spark Open 816B)
H = 8192  # hidden
MH = 4096  # pooled MoE hidden
E = 256  # routed experts (all on every rank at EP1)
TOPK = 8
I_FULL = 4096  # expert intermediate
RI = I_FULL // TP  # 1024: routed intermediate a rank (EP1 slice)
HEADS = 128
KV_HEADS = 16
HEAD_DIM = 64
CORE = HEADS * HEAD_DIM // TP  # 2048: o_proj K a rank
MAX_TOKENS = 8  # rows a step (C1: 1 request x 8 verify tokens)
ROWS = 16  # rows of a WMMA tile

# ---- TP-split dense pieces at TP4
EPR = E // TP  # 64 router rows a rank
POOL_N = MH // TP  # 1024 pooled columns a rank
UP_N = H // TP  # 2048 unpooler (branch) columns a rank

# ---- numerics constants (checkpoint / served code)
EPS_POST_ATTN = 1e-8
EPS_FFN = 1e-5
EPS_PRE_EXPERT = 1e-5
EPS_POST_EXPERT = 1e-8
ROUTE_EPS = 1e-15
ROUTE_SCALE = 1.0
ROUTER_MULT = 0.17677669529663687  # config.output_multiplier (applied to raw logits before sigmoid)
RESIDUAL_GATE_TEMP = 0.3

# ---- expert activation format (S2 decision: A8W4)
ACT_FMT = "mxfp8_e4m3"  # e8m0 per 32, OCP rule: exp = floor(log2(amax)) - 8, RNE, saturate 448


def row_owner(t: int) -> int:
    """Rank that owns token row t for the expert reduction / finalize (row-owner plan; 2 rows a rank at T=8)."""
    return t // (MAX_TOKENS // TP)
