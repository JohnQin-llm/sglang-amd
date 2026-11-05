# K2 gfx1250 integration patch (S4)

`k2_gfx1250.patch` (built by `make_patch.py` from copies; the served tree is never edited) adds:

- `python/sglang/srt/rocm_muse_spark/mono_gfx1250/`: the merged port. It contains `k2.py` (the persistent kernel:
  one launch a layer a rank, 256 CTAs x 512 threads), `runner.py` (peer runtime, per-layer operands, EP1 expert
  copy), `layout.py`, `microbench.py`, and the helper packages `common/ sync/ moe/ gemv/` with their imports
  rewritten.
- hooks in `rocm_muse_spark/model.py`, all behind `SGLANG_MUSE_MONO_GFX1250=1` (`apply_hooks.py` lists them).
  Without the flag the model behaves exactly as served.

## Apply

```
cd /path/to/sglang            # a checkout matching /data/muse455/sglang
git apply --check k2_gfx1250.patch && git apply k2_gfx1250.patch
```

## Run

Requirements: gfx1250, `--tp 4`, `MUSE_FUSE_FFN_BOUNDARY=1` (K2 hands its stream / branch to the native FFN
boundary op).

```
SGLANG_MUSE_MONO_GFX1250=1 MUSE_FUSE_FFN_BOUNDARY=1 \
  [SGLANG_MUSE_MONO_GFX1250_CKPT=/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4] \
  python3 -m sglang.launch_server ... --tp 4 --mem-fraction-static <served value - 0.24>
```

- Memory: the EP1 copy of the routed experts adds **102.7 GB a GPU** (all 256 experts, the rank's 1024-column
  slice of every intermediate, gfx1250 WMMA layout). It is read from the checkpoint with safetensors slices
  (`SGLANG_MUSE_MONO_GFX1250_CKPT`, default the served path). Lower `--mem-fraction-static` by that much (0.24 of
  the 432 GB MI455X; peer and scratch buffers add ~25 MB); the served EP4 expert weights stay loaded for the steps
  K2 does not take.
- Which steps take K2: verify steps of exactly 8 rows (the q8 build C1 served). Layers 1..60 only (layers 0 and 61
  have BF16 experts). Every other step and layer runs the served path unchanged.
- One epoch advance a step (`K2Runner.begin_step`, graph-captured). The mailbox tag holds 22 bits of epoch, and
  the runner fails loudly before reuse.

## Microbench

```
PYTHONSAFEPATH=1 python3 -m sglang.srt.rocm_muse_spark.mono_gfx1250.microbench --pool 64 --out k2_mb.json
```

This uses 4 processes on the TP group's GPUs, synthetic weights, and routes steered to about 43 distinct experts
(the C1 verify mean). It prints the K2 time a layer on each rank (graph replay, median). Compare that with the
served span K2 replaces (o_proj through unpooler, FFN boundary excluded). That span was 170 µs of kernel time /
180 µs wall a layer a rank in the C1 128K profile (`/data/muse455/k2/ep1/s3/served_span_layers.json`), and it
shrinks after `m-moe-triton`. Numerics are checked in the dev tree (`/data/muse455/k2/ep1/s3/run_k2.py`: real
weights, stream cosine and routes against the bars).
