"""Real Triton QKV partials (served _dense_partial, TP4 M=8 tile, imported read-only) behind a K2-shaped persistent
kernel, 62 layers with distinct weights (cold), in a torch CUDA graph. Arms per layer:
  linear : reset -> K2 -> QKV                         (today's order)
  pdl    : reset -> fork -> K2 | wait(trigger) -> QKV -> join   (early launch; QKV does NOT wait for K2's output:
           an upper bound on the saving, numerically unsafe)
Prints per-layer time and K2 residency-guard timeouts. Setting comes from the environment."""
import ctypes
import os
import sys

sys.path.insert(0, "/data/muse455/sglang/python")
import torch
import triton

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial

lib = ctypes.CDLL("/data/muse455/pdltest/bench/libk2shape.so")
for f in ("k2_launch", "k2_reset", "k2_wait_trigger"):
    getattr(lib, f).restype = ctypes.c_int
lib.k2_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.k2_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.k2_wait_trigger.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.k2_guard_read_and_clear.restype = ctypes.c_uint

M, N, K, SK, BN = 8, 4608, 8192, 8, 64
L = 62
dev = torch.device("cuda:0")
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(L)]
p = torch.empty(SK, M, N, device=dev, dtype=torch.float32)
ctr = torch.zeros(L, 4, device=dev, dtype=torch.int32)
setting = " ".join(f"{k[16:]}={v}" for k, v in os.environ.items() if k.startswith("DEBUG_HIP_GRAPH") or k == "DEBUG_HIP_FORCE_GRAPH_QUEUES") or "default"


def qkv(w):
    _dense_partial[(N // BN, 1, SK)](x, w, p, M, N, K, 16, BN, 128, SK, ".cg", num_warps=4, num_stages=2,
                                     matrix_instr_nonkdim=16)


def build(arm, vgpr, body, hand):
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    def body_fn():
        cur = torch.cuda.current_stream()
        for l in range(L):
            c = ctr[l].data_ptr()
            lib.k2_reset(ctypes.c_void_p(cur.cuda_stream), ctypes.c_void_p(c))
            if arm == "linear":
                lib.k2_launch(ctypes.c_void_p(cur.cuda_stream), ctypes.c_void_p(c), body, hand, vgpr)
                qkv(ws[l])
            else:
                s2.wait_stream(cur)
                lib.k2_launch(ctypes.c_void_p(cur.cuda_stream), ctypes.c_void_p(c), body, hand, vgpr)
                with torch.cuda.stream(s2):
                    lib.k2_wait_trigger(ctypes.c_void_p(s2.cuda_stream), ctypes.c_void_p(c))
                    qkv(ws[l])
                cur.wait_stream(s2)
    with torch.cuda.stream(s1):
        body_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s1):
        body_fn()
    return g


def time_graph(g, reps=10):
    g.replay()
    torch.cuda.synchronize()
    lib.k2_guard_read_and_clear()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / L)
    ts.sort()
    return ts[len(ts) // 2], lib.k2_guard_read_and_clear()


qkv(ws[0])
torch.cuda.synchronize()
HANDS = [float(h) for h in os.environ.get("HANDS", "4").split(",")]
for vgpr, hand in [(v, h) for v in (251, 224) for h in HANDS]:
    res = {}
    for arm in ("linear", "pdl"):
        g = build(arm, vgpr, 40.0, hand)
        res[arm] = time_graph(g)
        del g
    # QKV alone (cold) and K2 alone for reference
    print(f"RESULT real-QKV [{setting}] K2 vgpr={vgpr} hand-off window {hand:.0f} us: per layer linear {res['linear'][0]:.2f} us, "
          f"PDL(upper bound, no output wait) {res['pdl'][0]:.2f} us, saving {res['linear'][0] - res['pdl'][0]:+.2f} us | "
          f"K2 residency timeouts linear {res['linear'][1]} pdl {res['pdl'][1]}", flush=True)
