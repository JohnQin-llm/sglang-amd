"""K2 gfx1250: the persistent TP4 / EP1 MoE layer kernel for the served Muse Spark model.

One K2 launch a decoder layer and TP rank covers o_proj .. the MoE branch (attention boundary, router, pooler,
the routed experts over an EP1 copy, finalize, unpooler, every TP collective in the kernel); the native FFN
boundary closes the layer as on the served path. Off unless ``SGLANG_MUSE_MONO_GFX1250=1``.

The kernel, its peer runtime and the weight layouts are not part of this tree. They are imported from the port
package under ``SGLANG_MUSE_MONO_PORT`` (the directory holding ``port/``), so a faster port tree is served by
pointing the variable at it, without re-patching. The adapter below uses the same port entry points as the
port's own 4-GPU harness (``s3/run_k2.py``): ``k2.K2Build / build / k2_launch / peer_bytes / scratch_bytes``,
``sync.peer.PeerRuntime``, ``sync.epoch.advance_epoch_``, ``layout.gfx1250_weights``, ``gemv.gemv.shuffle_rows``.
A port tree that changes the operand layouts can ship ``port/serving.py`` with
``layer_operands(adapter_ops, layer_idx, rank, device) -> dict``; it then replaces ``layer_operands`` below.

Steps K2 takes: target-verify steps of exactly 8 rows (one request x 8 draft tokens) on the layers in
``SGLANG_MUSE_MONO_LAYERS`` (default 1-60; layers 0 and 61 have BF16 experts). One epoch advance a step
(``begin_step``), then one launch a layer; both are plain stream work, so they are captured in the verify CUDA
graph. Every other step and layer runs the served TP4 / EP4 path unchanged (its expert weights stay loaded).

Environment:
  SGLANG_MUSE_MONO_GFX1250=1          enable
  SGLANG_MUSE_MONO_PORT=<dir>         port tree root (default /data/muse455/k2/ep1)
  SGLANG_MUSE_MONO_LAYERS=1-60        layers K2 takes (``a-b`` ranges and commas)
  SGLANG_MUSE_MONO_GFX1250_CKPT=<dir> checkpoint of the EP1 expert copy (default: the served model path)
  SGLANG_MUSE_MONO_CHECK=1            fail-stop check (host sync) before every eager K2 step; debug only
  SGLANG_MUSE_MONO_REPORT_S=60        rank 0 logs the K2 step count (the epoch, graph replays included) and the
                                      ranks' in-kernel failure flags every N s, from a side thread (0: off)
"""

import importlib
import json
import logging
import os
import sys
import threading
import time

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_MUSE_MONO_GFX1250", "0") == "1"
PORT_ROOT = os.path.realpath(os.environ.get("SGLANG_MUSE_MONO_PORT", "/data/muse455/k2/ep1"))
LAYERS = os.environ.get("SGLANG_MUSE_MONO_LAYERS", "1-60")
CHECK = os.environ.get("SGLANG_MUSE_MONO_CHECK", "0") == "1"
REPORT_S = float(os.environ.get("SGLANG_MUSE_MONO_REPORT_S", "60"))
DEFAULT_CKPT = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
BF16_EXPERT_LAYERS = (0, 61)
EXPERT_PREFIX = "model.language_model.layers.{}.mlp.experts.{}."
EXPERT_CHUNK = 32  # experts a host -> device chunk of the EP1 copy

RUNNER = None  # this process's K2Runner (post_load_weights)


def parse_layers(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


class PortOps:
    """The port modules serving uses, imported from ``root`` (first on ``sys.path``)."""

    def __init__(self, root: str = PORT_ROOT):
        if not os.path.isfile(os.path.join(root, "port", "k2.py")):
            raise FileNotFoundError(f"SGLANG_MUSE_MONO_PORT={root}: no port/k2.py there")
        loaded = sys.modules.get("port")
        if loaded is not None:
            where = os.path.dirname(os.path.dirname(os.path.realpath(loaded.__file__ or "")))
            if where != root:
                raise ImportError(f"another 'port' package is already imported (from {where}), not {root}")
        if root not in sys.path:
            sys.path.insert(0, root)
        self.root = root
        self.K = importlib.import_module("port.consts")
        self.k2 = importlib.import_module("port.k2")
        self.layout = importlib.import_module("port.layout")
        self.gemv = importlib.import_module("port.gemv.gemv")
        self.epoch = importlib.import_module("port.sync.epoch")
        self.peer = importlib.import_module("port.sync.peer")
        self.sync = importlib.import_module("port.sync.sync")
        try:
            self.serving = importlib.import_module("port.serving")
        except ModuleNotFoundError:
            self.serving = None

    def describe(self) -> str:
        S = self.sync
        return (
            f"port {self.root} (sources {self.k2._SOURCES[:12]}), poll sleep {S._POLL_SLEEP}, "
            f"inv poll {int(S._INV_POLL)}, shuffled {self.gemv.SHUFFLED}"
            + (", port/serving.py operands" if self.serving is not None else "")
        )


# ---------------------------------------------------------------- EP1 expert copy (read from the checkpoint)
class Checkpoint:
    def __init__(self, root: str):
        from safetensors import safe_open

        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._open = safe_open
        self._files = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def slice(self, name: str, dim: int, start: int, length: int) -> torch.Tensor:
        fn = self.map[name]
        f = self._files.get(fn)
        if f is None:
            f = self._files[fn] = self._open(os.path.join(self.root, fn), "pt")
        sl = f.get_slice(name)
        idx = [slice(None)] * len(sl.get_shape())
        idx[dim] = slice(start, start + length)
        return sl[tuple(idx)]


def load_ep1_experts(ops: PortOps, ck: Checkpoint, layer: int, rank: int, device, experts=None):
    """(w13, s13, w2, s2) of ``layer`` for ``rank`` on ``device``: all 256 experts (or ``experts``; the others
    zero), the rank's quarter of every intermediate (gate rows, up rows, down K), in the gfx1250 layout.
    Read with safetensors slices (only this rank's bytes) and converted on the device a chunk at a time, so the
    only device memory beyond the copy itself is one chunk."""
    K, L = ops.K, ops.layout
    ri, E, MH = K.RI, K.E, K.MH
    shapes = ((E, 2 * ri, MH // 2), (E, 2 * ri, MH // 32), (E, MH, ri // 2), (E, MH, ri // 32))
    alloc = torch.empty if experts is None else torch.zeros
    out = [alloc(s, dtype=torch.uint8, device=device) for s in shapes]
    todo = list(range(E)) if experts is None else sorted(set(int(e) for e in experts))
    for c0 in range(0, len(todo), EXPERT_CHUNK):
        ids = todo[c0:c0 + EXPERT_CHUNK]
        host = [torch.empty((len(ids),) + s[1:], dtype=torch.uint8) for s in shapes]
        for j, e in enumerate(ids):
            p = EXPERT_PREFIX.format(layer, e)
            host[0][j, :ri] = ck.slice(p + "gate_proj.weight", 0, rank * ri, ri)
            host[0][j, ri:] = ck.slice(p + "up_proj.weight", 0, rank * ri, ri)
            host[1][j, :ri] = ck.slice(p + "gate_proj.weight_scale", 0, rank * ri, ri)
            host[1][j, ri:] = ck.slice(p + "up_proj.weight_scale", 0, rank * ri, ri)
            host[2][j] = ck.slice(p + "down_proj.weight", 1, rank * ri // 2, ri // 2)
            host[3][j] = ck.slice(p + "down_proj.weight_scale", 1, rank * ri // 32, ri // 32)
        parts = L.gfx1250_weights(*(h.to(device) for h in host))
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        for dst, src in zip(out, parts):
            dst.index_copy_(0, idx, src)
        del parts
    return tuple(out)


# ---------------------------------------------------------------- one layer's dense operands (served modules)
def _expect(cond: bool, what: str, layer: int):
    if not cond:
        raise ValueError(f"K2 gfx1250, layer {layer}: served module does not match the K2 build ({what})")


def residual_coefficients(gate: torch.Tensor, temperature: float, eps: float):
    """``MuseSpark1xNormPreservingResidualConnection._coefficients`` (channel-wise, stream gate on)."""
    w = gate.reshape(-1).float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return alpha.contiguous(), beta.contiguous()


def _norm_ok(norm, eps: float, weighted: bool = True) -> bool:
    if abs(norm.eps - eps) > 1e-12 * max(1.0, eps):
        return False
    if not weighted:
        return norm.weight is None
    return norm.weight is not None and norm.zero_centered_gamma and float(norm.gain_center) == 1.0


def check_layer(ops: PortOps, layer_idx: int, mod) -> None:
    """The served layer's shapes and constants are the ones K2 is built with."""
    K = ops.K
    key = ops.k2.K2Build()
    attn, moe, gate = mod.self_attn, mod.mlp, mod.post_attention_residual_gate
    _expect(tuple(attn.o_proj.weight.shape) == (K.H, K.CORE), f"o_proj {tuple(attn.o_proj.weight.shape)}", layer_idx)
    _expect(_norm_ok(mod.post_attn_norm, key.eps_post_attn, weighted=False), "post_attn_norm", layer_idx)
    _expect(_norm_ok(mod.pre_feedforward_layernorm, key.eps_ffn), "pre_feedforward_layernorm", layer_idx)
    _expect(_norm_ok(moe.pre_expert_norm, key.eps_pre_expert), "pre_expert_norm", layer_idx)
    _expect(_norm_ok(moe.post_expert_norm, key.eps_post_expert), "post_expert_norm", layer_idx)
    _expect(
        gate.use_stream_gate and gate.gate.numel() == K.H and abs(gate.temperature - K.RESIDUAL_GATE_TEMP) < 1e-12,
        "post_attention_residual_gate", layer_idx,
    )
    _expect(moe.num_experts == K.E and moe.top_k == K.TOPK, "experts / top-k", layer_idx)
    _expect(moe.router_in_fp32 and tuple(moe.router.weight.shape) == (K.H, K.E), "router", layer_idx)
    _expect(
        abs(moe.router_multiplier - key.router_output_multiplier) < 1e-12
        and abs(moe.route_scale - K.ROUTE_SCALE) < 1e-12 and abs(moe.route_eps - K.ROUTE_EPS) < 1e-30,
        "routing constants", layer_idx,
    )
    _expect(tuple(moe.pre_expert_proj.weight.shape) == (K.MH, K.H), "pre_expert_proj", layer_idx)
    _expect(tuple(moe.post_expert_proj.weight.shape) == (K.H, K.MH), "post_expert_proj", layer_idx)


def dense_operands(ops: PortOps, mod) -> dict:
    """K2's dense operands from the served (loaded) decoder layer. Views where the served tensor already has
    K2's dtype and layout; copies for the three shuffled GEMV weights and the [E, H] router."""
    gemv = ops.gemv
    shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t.contiguous())
    attn, moe, gate = mod.self_attn, mod.mlp, mod.post_attention_residual_gate
    bf = torch.bfloat16
    w = {}
    w["w_o"] = shuf(attn.o_proj.weight.data.to(bf).contiguous())  # [H, CORE]: this rank's heads
    w["alpha_a"], w["beta_a"] = residual_coefficients(gate.gate.data, gate.temperature, gate.eps)
    w["w_fn"] = mod.pre_feedforward_layernorm.weight.data.to(bf).contiguous()
    w["w_router"] = moe.router.weight.data.t().float().contiguous()  # served [H, E] -> [E, H]
    w["expert_bias"] = moe.expert_bias.data.float().contiguous()
    w["w_pool"] = shuf(moe.pre_expert_proj.weight.data.to(bf).contiguous())
    w["w_pre"] = moe.pre_expert_norm.weight.data.to(bf).contiguous()
    w["w_post"] = moe.post_expert_norm.weight.data.to(bf).contiguous()
    w["w_unpool"] = shuf(moe.post_expert_proj.weight.data.to(bf).contiguous())
    return w


def layer_operands(ops: PortOps, mod, layer_idx: int, rank: int, device, ck: Checkpoint, experts=None) -> dict:
    if ops.serving is not None:
        return ops.serving.layer_operands(ops, mod, layer_idx, rank, device, ck, experts)
    w = dense_operands(ops, mod)
    w["w13"], w["w13s"], w["w2"], w["w2s"] = load_ep1_experts(ops, ck, layer_idx, rank, device, experts)
    return w


def _new_bytes(w: dict, mod) -> int:
    """Bytes of ``w`` not shared with the served module's parameters / buffers."""
    served = set()
    if callable(getattr(mod, "parameters", None)):
        served = {t.untyped_storage().data_ptr() for t in list(mod.parameters()) + list(mod.buffers())}
    seen, n = set(), 0
    for t in w.values():
        s = t.untyped_storage()
        if s.data_ptr() not in served and s.data_ptr() not in seen:
            seen.add(s.data_ptr())
            n += s.nbytes()
    return n


# ---------------------------------------------------------------- the runner
class K2Runner:
    def __init__(self, ops: PortOps, group_cpu, rank: int, device, ckpt: str):
        K, k2 = ops.K, ops.k2
        self.ops, self.rank, self.device = ops, rank, device
        self.key = k2.K2Build()
        t0 = time.time()
        k2.build(self.key)
        self.build_s = time.time() - t0
        self.rt = ops.peer.PeerRuntime(k2.peer_bytes(), group_cpu, rank, K.TP, device, blocks=K.BLOCKS)
        self.scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=device)
        self.stream_out = torch.empty(K.MAX_TOKENS, K.H, dtype=torch.float32, device=device)
        self.branch = torch.empty(K.MAX_TOKENS, K.H, dtype=torch.bfloat16, device=device)
        self.ckpt = Checkpoint(ckpt)
        self.layers = {}
        self.extra_bytes = (
            k2.peer_bytes() + self.scratch.numel() * 4 + self.rt.health.numel() * 4
            + self.stream_out.numel() * 4 + self.branch.numel() * 2
        )
        self.rows = K.MAX_TOKENS
        self.last_capture = 0.0

    def add_layer(self, layer_idx: int, mod, experts=None) -> None:
        check_layer(self.ops, layer_idx, mod)
        w = layer_operands(self.ops, mod, layer_idx, self.rank, self.device, self.ckpt, experts)
        missing = [n for n in self.ops.k2.ABI.names[:16] if n not in ("core", "stream_a") and n not in w]
        if missing:
            raise ValueError(f"K2 gfx1250, layer {layer_idx}: operands missing {missing}")
        self.extra_bytes += _new_bytes(w, mod)
        self.layers[layer_idx] = w

    def takes_step(self, rows: int, forward_batch) -> bool:
        return bool(self.layers) and rows == self.rows and forward_batch.forward_mode.is_target_verify()

    def takes(self, layer_idx: int, rows: int, forward_batch) -> bool:
        return layer_idx in self.layers and self.takes_step(rows, forward_batch)

    def begin_step(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            self.last_capture = time.time()
        elif CHECK:
            torch.cuda.current_stream().synchronize()
            self.rt.check()
        self.ops.epoch.advance_epoch_(self.rt.epoch)

    def run(self, layer_idx: int, core: torch.Tensor, stream: torch.Tensor):
        """core [8, 2048] bf16 (o_proj input, this rank's heads), stream [8, 8192] fp32 -> (stream, branch):
        the residual stream after the attention boundary (fp32) and the MoE branch after the unpooler (bf16)."""
        assert core.shape == (self.rows, self.ops.K.CORE) and core.dtype == torch.bfloat16, core.shape
        assert stream.shape == (self.rows, self.ops.K.H) and stream.dtype == torch.float32, stream.shape
        t = dict(
            self.layers[layer_idx], core=core.contiguous(), stream_a=stream.contiguous(),
            stream_out=self.stream_out, branch_out=self.branch,
        )
        self.ops.k2.k2_launch(
            self.key, t, self.rank, layer_idx, self.rt.epoch, self.rt.peers, self.scratch,
            self.rt.health, self.rt.alarm, torch.cuda.current_stream().cuda_stream,
        )
        return self.stream_out, self.branch

    def check(self) -> None:
        self.rt.check()


def _report(runner: K2Runner, period: float) -> None:
    """Side thread (rank 0): the epoch and the ranks' failure words, copied on a non-blocking side stream into
    pinned memory, so serving streams are never synchronized. No CUDA call from this thread while graphs are
    being captured: the first one comes after startup (>= 300 s) and none within 120 s of a captured step."""
    try:
        time.sleep(max(period, float(os.environ.get("K2_REPORT_DELAY", "300"))))
        while time.time() - runner.last_capture < 120.0:
            time.sleep(30.0)
        torch.cuda.set_device(runner.device)
        stream = torch.cuda.Stream(device=runner.device)
        err = runner.rt.error_words()
        host = torch.zeros(1 + err.numel(), dtype=torch.int32, pin_memory=True)
        last = None
        while True:
            if time.time() - runner.last_capture < 120.0:
                time.sleep(30.0)
                continue
            with torch.cuda.stream(stream):
                host[:1].copy_(runner.rt.epoch, non_blocking=True)
                host[1:].copy_(err, non_blocking=True)
            stream.synchronize()
            ep, bad = int(host[0]), host[1:].tolist()
            if any(bad):
                logger.error("K2 gfx1250: %d K2 steps; a rank gave up an in-kernel wait (flags %s)", ep, bad)
                try:
                    dbg = importlib.import_module("port.sync.debug")
                    hh = torch.empty(runner.rt.health.numel(), dtype=torch.int32, pin_memory=True)
                    with torch.cuda.stream(stream):
                        hh.copy_(runner.rt.health, non_blocking=True)
                    stream.synchronize()
                    waits = dbg.given_up_waits(hh, runner.k2.REGIONS)
                    logger.error("K2 gfx1250: rank 0 give-up records (%d CTAs), first 16:\n  %s", len(waits),
                                 "\n  ".join(waits[:16]))
                except Exception as ex:  # reporting only
                    logger.warning("K2 gfx1250: could not decode the give-up records: %r", ex)
                return
            elif ep != last:
                logger.info("K2 gfx1250: %d K2 steps so far, no rank failed", ep)
            last = ep
            time.sleep(period)
    except Exception as ex:  # reporting only; never take the server down
        logger.warning("K2 gfx1250 reporter stopped: %r", ex)


def _report_rank(runner: K2Runner, rank: int, period: float) -> None:
    """Debug side thread (every rank, K2_REPORT_ALL=1): this rank's K2 step count and its own sticky FAILED word;
    on the first failure, its own give-up records. Same capture-safety rules as _report."""
    try:
        time.sleep(max(period, float(os.environ.get("K2_REPORT_DELAY", "300"))))
        while time.time() - runner.last_capture < 120.0:
            time.sleep(30.0)
        torch.cuda.set_device(runner.device)
        stream = torch.cuda.Stream(device=runner.device)
        host = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        last = None
        while True:
            with torch.cuda.stream(stream):
                host[:1].copy_(runner.rt.epoch, non_blocking=True)
                host[1:].copy_(runner.rt.health[:1], non_blocking=True)
            stream.synchronize()
            ep, failed = int(host[0]), int(host[1])
            if failed:
                dbg = importlib.import_module("port.sync.debug")
                hh = torch.empty(runner.rt.health.numel(), dtype=torch.int32, pin_memory=True)
                with torch.cuda.stream(stream):
                    hh.copy_(runner.rt.health, non_blocking=True)
                stream.synchronize()
                waits = dbg.given_up_waits(hh, runner.k2.REGIONS)
                logger.error("K2 gfx1250 rank %d: FAILED at epoch %d; %d CTAs gave up; first 12:\n  %s", rank, ep,
                             len(waits), "\n  ".join(waits[:12]))
                return
            if ep != last:
                logger.info("K2 gfx1250 rank %d: epoch %d, ok", rank, ep)
            last = ep
            time.sleep(period)
    except Exception as ex:  # reporting only
        logger.warning("K2 gfx1250 rank %d reporter stopped: %r", rank, ex)


def _checkpoint_path() -> str:
    path = os.environ.get("SGLANG_MUSE_MONO_GFX1250_CKPT")
    if path:
        return path
    try:
        from sglang.srt.runtime_context import get_server_args

        return get_server_args().model_path
    except Exception:
        return DEFAULT_CKPT


@torch.no_grad()
def build_runner(lm) -> None:
    """Build this process's runner and every taken layer's operands (collective over the TP group).

    Called at the end of ``post_load_weights``, i.e. inside weight loading, so the KV pool is sized after the
    EP1 copy is resident (the copy comes out of the KV budget, not out of the non-static slack)."""
    global RUNNER
    if RUNNER is not None:
        logger.info("K2 gfx1250: runner already built; keeping it (operands are not reloaded)")
        return
    from sglang.srt.runtime_context import get_parallel

    tp = get_parallel().tp_group
    dev = torch.device("cuda", torch.cuda.current_device())
    arch = torch.cuda.get_device_properties(dev).gcnArchName
    if tp.world_size != 4 or not arch.startswith("gfx1250"):
        raise ValueError(f"SGLANG_MUSE_MONO_GFX1250 needs gfx1250 at TP4 (got {arch}, TP{tp.world_size})")
    model = lm.model
    if getattr(model, "_ffn_boundaries", None) is None:
        raise ValueError("SGLANG_MUSE_MONO_GFX1250 needs the fused FFN boundary (MUSE_FUSE_FFN_BOUNDARY=1)")
    ops = PortOps()
    ckpt = _checkpoint_path()
    want = parse_layers(LAYERS)
    taken = [
        i for i, layer in enumerate(model.layers)
        if i in want and i not in BF16_EXPERT_LAYERS and layer.is_moe_layer
        and getattr(layer.mlp, "mxfp4_checkpoint", False)
    ]
    free0 = torch.cuda.mem_get_info(dev)[0]
    t0 = time.time()
    runner = K2Runner(ops, tp.cpu_group, tp.rank_in_group, dev, ckpt)
    if tp.rank_in_group == 0:
        logger.info("K2 gfx1250: %s; build %.1f s; layers %s; checkpoint %s",
                    ops.describe(), runner.build_s, f"{taken[0]}..{taken[-1]} ({len(taken)})" if taken else "none", ckpt)
    for n, i in enumerate(taken):
        runner.add_layer(i, model.layers[i])
        if tp.rank_in_group == 0 and (n + 1) % 10 == 0:
            logger.info("K2 gfx1250: %d/%d layers loaded (%.0f s)", n + 1, len(taken), time.time() - t0)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    used = free0 - torch.cuda.mem_get_info(dev)[0]
    logger.info(
        "K2 gfx1250 rank %d: %d layers in %.0f s; K2 extra %.2f GiB (%.0f bytes counted, %.2f GiB by mem_get_info)",
        tp.rank_in_group, len(runner.layers), time.time() - t0, runner.extra_bytes / 2**30, runner.extra_bytes,
        used / 2**30,
    )
    RUNNER = runner
    if tp.rank_in_group == 0 and REPORT_S > 0:
        threading.Thread(target=_report, args=(runner, REPORT_S), name="k2g-report", daemon=True).start()
    if os.environ.get("K2_REPORT_ALL") == "1" and REPORT_S > 0:
        threading.Thread(target=_report_rank, args=(runner, tp.rank_in_group, REPORT_S), name="k2g-report-rank",
                         daemon=True).start()
