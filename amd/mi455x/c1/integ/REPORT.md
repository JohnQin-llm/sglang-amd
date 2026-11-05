# Bundle1 integration report — Muse Spark Open 816B (MXFP4, DFlash2), 4x MI455X gfx1250, TP4, AgentX C1

Branch `mi455x/c1-bundle1` in worktree `/data/muse455/integ/sglang` (3 commits on 3e62d3ec1f):
`38efae2ec3` bundle1/5 DFlash draft fusions, `fd612fda0e` bundle1/2 QKV, `bf96ca7618` bundle1/3+4 verify attention.
Patch `/data/muse455/integ/bundle1.patch` (git diff 3e62d3ec1f..mi455x/c1-bundle1; 11 files, +1264 / -48).
Env set `/data/muse455/integ/bundle1.env`. **No AITER patch**: changes 3 and 4 wrap the AITER wrapper module's
`get_unified_attention_config` / `_unified_attention_3d_triton` / `_reduce_segments_triton` globals from SGLang
(`python/sglang/srt/rocm_muse_spark/verify_attention.py`, installed from `aiter_backend.py` at import, no-op unless a
flag is on; every non-matching call falls through to the served helper). `/data/muse455/sglang`,
`/sgl-workspace/aiter` and muse_mi455x were not modified.

## Env flags (all default off = served)

| change | flag(s) |
|---|---|
| 1 K2 v4 | `SGLANG_MUSE_MONO_PORT=/data/muse455/k2/cand/k2v4_land1` (served: k2v2_land1) |
| 2 QKV projection | `SGLANG_MUSE_QKV_TDM=1` (gluon TDM split-K partials; Triton num_stages 4 if the gluon kernel is unavailable), `SGLANG_MUSE_QKV_SPLIT_EPI=1` (`_qkv_epilogue_split`, grid (m, hq + hk)) |
| 3 full-attention verify v2 | `SGLANG_MUSE_ATTN_V2=1` |
| 4 sliding-window verify config | `SGLANG_MUSE_SWA_CFG=1` (kv_split["D_LEQ_64.SW.DT_any_fp8"] of DEFAULT.proposed.json only: 4 segments/CU, 32-token tiles -> 64 x 32 at C1 verify) |
| 5 DFlash2 draft | `SGLANG_MUSE_DFLASH_KV_FP8_FUSE=1 SGLANG_MUSE_DFLASH_CONV_NORM_FUSE=1 SGLANG_MUSE_DFLASH_PACKED_CANDIDATE_GATHER=1`; `SGLANG_MUSE_DFLASH_TRITON_TOPK` off |

## Per-change verification inside the integrated tree (sglang imported from the worktree)

All GPU jobs through `k2safe.sh` (K2 with the mpol_exec wrapper), all "GPUs drained, no new MES errors".
Logs `/data/muse455/integ/verify/logs/`, JSON `/data/muse455/integ/verify/results/`.

MICROBENCH (unit us, before = served / flag off, after = flag on):

| component | before | after | numerics |
|---|---|---|---|
| K2 per layer, slowest rank, warm L1 / L31 / L60 | 65.1 / 64.8 / 64.4 | 61.4 / 61.3 / 61.8 | stream digests equal on every rank (363d3c1a.., 002c3013.., cbf9461d..), replay == golden all ranks |
| K2 per layer, cold L1 / L31 / L60 | 67.6 / 68.1 / 67.8 | 64.6 / 64.4 / 65.3 | same |
| QKV + gate op (8 rows, N 4608, K 8192), cold / warm, TDM + split epilogue | 13.24 / 8.65 | 9.37 / 7.85 | 0 bytes differ in Q, K, V, gate, K cache, V cache |
| QKV op cold, Triton-only fallback (st4 + split epilogue) | 13.24 | 9.64 | 0 bytes differ |
| QKV op cold, split epilogue alone | 13.24 | 12.42 | 0 bytes differ |
| full-attention verify (split + reduce), cold, 32K / 128K / 340K | 16.75 / 27.89 / 55.59 | 14.39 / 21.53 / 40.05 | max err = served on all 12 fixtures; 5 / 9 / 14 of 262,144 outputs differ from served; 0 differ from attn2's own v2_T128 |
| sliding-window verify, cold, 32K / 128K / 340K | 9.76 / 10.25 / 9.90 | 8.73 / 9.24 / 8.73 | max err unchanged; 375-991 of 65,536 outputs differ from served per fixture; 0 differ from vagm's s64t32 override |
| DFlash draft step QK-norm/RoPE + FP8 K/V store, 5 layers | 59.83 | 7.54 | bitwise |
| DFlash context KV commit, 5 layers | 67.78 | 1.71 | bitwise (commit_len 1/4/8) |
| DFlash conv finish + add-RMSNorm, 10 per step | 41.09 | 23.50 | bitwise (0 of 655,360) |
| DFlash 5-layer stub draft forward | 292.7 | 278.8 | bitwise |
| DFlash candidate top-k gather, slowest rank | 90.21 | 68.26 | ids + value bits equal to served on all ranks |

- QKV byte check: 16 fixtures (1, 3, 8, 16 rows; 2 seeds; with / without SWA slot mapping, incl. -1 slots) x 5 flag
  combinations, all 0 bytes in all six outputs.
- Flags off: QKV flags off = 0 bytes vs the served module; attention flags off (wrappers installed) = 0 bf16 outputs
  differ from AITER before install, on all 24 fixtures.
- v2 routing predicate on the served C1-verify params: taken; rejected for num_seqs 2, 9 rows, sinks, softcap,
  alibi, output scale, shuffled cache, head 128, FP8 Q, sliding window, tile 64.

### qkv_epilogue_split ISA difference (aot_compare.log said "DIFFERENT")

Debug metadata only. The AOT dump of the integrated `_qkv_epilogue_split` differs from vagm's copy in `.file` / `.loc`
lines (source path and line numbers: the integrated kernel lives in `qk_norm_rope_cache.py` and its helpers are in the
same file, vagm's copy in a temp file calling the served module), which my first filter missed. With directives and
comments stripped the instruction streams are identical: 590 / 601 lines (HAS_MAP 0 / 1), 0 differing; 48 VGPRs,
occupancy 16. `attn_v2_split` (T128 S128 QSPLIT0 FOLD1 w4 st2): identical, 3725 lines, 364 VGPRs, no scratch.
Re-measured on GPU after the reboot (qkv_verify.json): bytes identical everywhere (above), op cold 13.24 -> 9.37 us
(gluon TDM + split epilogue), 9.64 us (Triton st4 + split epilogue); warm 8.65 -> 7.85 / 7.87 us; split epilogue
alone 12.42 cold.

## Server smoke test

Container `muse_integ` (same image, devices, mounts as muse_mi455x; worktree at /sgl-workspace/sglang). The C1 AITER
state could not be `git apply`-ed onto the image's AITER (the image's tree already carries 3 local edits), so the 12
modified files and the 4 JIT .so were copied from muse_mi455x; sha1 of all 16 identical to muse_mi455x
(`/data/muse455/integ/ctr/`). pip freeze identical except the sglang commit. Served argv/env from iter_12 (port 30100),
single-threaded loader, page cache warm, my own flock on the 4 GPU locks while servers ran. Startup ~2.5 min
(weights 30 s), same as served. 20 sequential /generate requests, temperature 0: 14 short, 12K, 30K, and 95K / 110K /
121K / 125K tokens.

Live lines (flags-on server, all 4 ranks): `Muse verify attention overrides installed: attn_v2 True, sliding-window
config True`; `K2 gfx1250: port /data/muse455/k2/cand/k2v4_land1 (sources c77868e3886c)`; `sliding-window kv split (4
segments per CU, 32-token tiles) live`; `attn_v2_split for full-attention layers (tile 128, 128 segments) live`;
`Muse QKV projection: split epilogue (grid m x (hq + hk)) live`; `gluon TDM split-K partials (BN 32, BK 128, split-K
8, 3 buffers, 2 warps) live`; `DFlash draft: fused FP8 K/V write in QK-norm/RoPE`; `DFlash draft: one-launch FP8
context KV commit`. CONV_NORM_FUSE / PACKED_CANDIDATE_GATHER have no log line in the patch; both set in every
scheduler process's environment. K2 health: `K2 gfx1250: 770 K2 steps so far, no rank failed`; no give-up in any log.
Tracebacks in every log (incl. flags-off) are uvicorn/asyncio lifespan cancellations at SIGTERM shutdown.

Token-level agreement (output_ids; "prefix" = first differing token for mid0 mid1 long0 long1 long2 long3, 128 = identical):

| A vs B | condition | requests identical | positions equal | prefix mid/long |
|---|---|---|---|---|
| off vs off2 | served env (simulated acceptance 4.05), same server, 2nd pass hits radix cache | 1/20 | 25.2% | 0 0 5 28 2 15 |
| off vs on | served env (simulated), fresh servers | 1/20 | 20.8% | 4 20 5 28 8 8 |
| off_real vs off_real2 | real acceptance, same server, 2nd pass hits radix cache | 10/20 | 59.0% | 0 0 3 3 31 2 |
| off_real vs off_real3 | real, same server, cache flushed | 19/20 | 94.7% | identical |
| on_real vs on_real2 | real, same server, cache flushed | 18/20 | 93.8% | identical |
| off_real vs noattn_real | real; K2 v4 + QKV + DFlash on, ATTN_V2 / SWA_CFG off | 19/20 | 94.7% | identical |
| off_real vs v2only_real | real; + ATTN_V2 (SWA_CFG off) | 3/20 | 41.3% | 71 84 2 3 2 45 |
| v2only_real vs on_real | real; the only difference is SWA_CFG=1 | 3/20 | 44.6% | 22 65 9 10 2 2 |
| off_real vs on_real | real; all flags | 3/20 | 38.5% | 22 65 2 3 128 2 |

Real acceptance = `SGLANG_SIMULATE_ACC_LEN=-1`; everything else as served. Under the served simulated acceptance the
outputs are not reproducible run to run, so only the real-acceptance rows measure numerics. The one non-identical
request in the noise-floor rows (short8, 7-token prompt) differs at token 0 even within flags-off.

## Why attention v2 (and the sliding-window config) change outputs: CPU investigation

Conclusion: no routing or masking bug found. The divergence is the greedy-decoding sensitivity of these prompts to any
non-bit-identical attention numerics, and **SWA_CFG diverges exactly as much as ATTN_V2**:
- `v2only_real` vs `on_real` differ only in SWA_CFG=1 and agree on 3/20 (44.6%), long prompts diverging after 2-10
  tokens, the same as ATTN_V2 alone (3/20, 41.3%, long after 2-45 tokens). So the gate's 4-change bundle (SWA_CFG=1,
  ATTN_V2 off) is not output-identical to served either. By fixture numerics SWA_CFG perturbs far more outputs per
  layer (375-991 of 65,536, ~1%) than v2 (5-14 of 262,144, ~0.004%).
- The served code path diverges the same way when only its prefill path changes (radix-cache hit: `off_real` vs
  `off_real2`, mid/long first differing token 0 0 3 3 31 2), with no flag involved.
- Everything bit-exact (K2 v4, QKV, DFlash) matches the noise floor in serving (19/20).
These are raw-text continuations (no chat template) where the top logits are close; a bf16-ulp change in a few
attention outputs flips an argmax within a few tokens and the sequences then drift apart. Token identity cannot
judge either attention change; an accuracy gate (e.g. GSM8K with real acceptance, accept length) has to.

Routing conditions checked (code: `verify_attention.py`, AITER `unified_attention.py` / `kernel_unified_attention_3d`
from muse_mi455x, `aiter_backend.py`):
- Call sites: v2 can only be reached through `_unified_attention_3d_triton`, i.e. AITER's 3D path. (a) target_verify
  (`aiter_backend.py` ~4225): q BF16, q_descale None, seqused_k = seq_lens + max_q_len (int64), cu_seqlens = qo_indptr,
  page table [bs, pages]: taken on every verify step of the 16 full layers at bs 1 (graph max_seqlen_k ->
  128 segments, tile 128). (b) extend/prefill (`_forward_extend_unified`, ~1237): only with <= 8 new tokens, one
  sequence and > 16 keys (NUM_SEGMENTS > 1), i.e. a short extend on a cached prefix; seqused_k = seq_lens (total),
  ctx = seq_len - qlen as in the served kernel. Not hit in the fresh-cache smoke passes (short prompts are 7-36
  tokens; the 7-token one gives NUM_SEGMENTS 1). (c) decode (~4834): q is quantized to FP8 before the call -> rejected
  (q dtype). (d) draft: the DFlash draft runs on the triton backend (head 128) -> never reaches AITER's wrapper; the
  aiter draft_extend path (~4343) is not used by DFlash. 
- Multi-sequence / padded batches: route only when len(seqused_k) == 1, num_tokens <= 8 and max_seqlen_q <= 8
  (host-side, decided at graph capture); bs > 1 graphs fall back. cu_q[0] / cu_q[1] and seq_lens[0] are what the
  served kernel reads for seq_idx 0.
- Page table / strides: v2 indexes `block_table` row 0 with `offs_t // PAGE` (served: `seq_idx * block_table_stride +
  seq_offset // BLOCK_SIZE`, seq_idx 0); K/V strides (page, in-page, head) passed from the same views; last-dim
  stride 1 checked; Q strides from the tensor.
- Scales: qk_scale = softmax_scale * log2(e) * k_descale, output * v_descale, exactly as served (q_descale None);
  FOLD only changes the contraction (exp2(fma(S, c, -m)) vs exp2(c*S - m)), the source of the 5-14 differing outputs.
- Full vs sliding: route requires SLIDING_WINDOW == 0 (window (-1,-1)); sliding layers keep the served kernel (with
  or without SWA_CFG).
- Segments / reduce: v2's `_segment_range(seq_len, TILE, NSEG)` = AITER `segment_range` with first_tile 0 (no window
  split on full layers); the reduce wrapper uses the same NUM_SEGMENTS / TILE_SIZE / seq_lens and the served reduce
  config with SEGMENT_BLOCK = max(64, NUM_SEGMENTS) (single-tile path; act_num_segments from the same segment_range).
  Partial layout [token, head, seg, 64] identical.
- Causal masking of the last tiles: unmasked tiles are those entirely below ctx + 1 (keys visible to every row of the
  block); the rest take masked loads (key < seq_len) and the causal mask key <= ctx + row, the served condition
  `seq_offset < context_len + query_pos + 1`; padded rows (row >= qlen) are not stored. Served num_tiles uses
  min(ctx + 8, seq_len) = seq_len for 8 rows, as does v2.
- Real (odd) lengths: attn2 `logs/odd.log` (same kernel, ISA-identical): 1000 / 4093 / 100003 / 262147 / 340001 keys
  give max err identical to served, 3 / 7 / 8 / 11 / (<=14) outputs differ, no NaN.
Residual untested: real (non-unit) k/v descale values in fixtures (code path identical to served), bs > 1 (falls back).

## Did not integrate cleanly / open items

- Changes 3 and 4 are not output-identical in the model (above); both need an accuracy gate, not only ATTN_V2.
- AITER C1 state is not reproducible by `git apply` on the image (copy files from muse_mi455x instead).
- Simulated acceptance in the served env makes greedy outputs irreproducible; use SGLANG_SIMULATE_ACC_LEN=-1 and a
  flushed radix cache for any output-identity check.
- The 06:46 UTC GPU fault (0003:01:00.0 SMU / failed mode-2 reset) preceded my first GPU job; all reported
  measurements are from after the 06:57 reboot.

State: muse_integ stopped and removed (07:57 UTC); my lock holder killed; no GPU process of mine; GPU locks free.
