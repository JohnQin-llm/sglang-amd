# K2 gfx1250 TP4/EP1 port: module interface contract

Lead: integrator (k2.py, loader, validation, timing, S4). Helpers: P1 `common/`, P2 `sync/`, P3 `moe/`,
P4 `gemv/`. Source: `/root/yanyuqin/handoff-mi455x/knowledge/mono-kernel/k2bundle/` (the C1 K2 path).
Status page: `/data/muse455/k2/ep1/README.md` (**lead only: helpers write their own `port/<dir>/README.md`**).
This file is the contract; ask the lead to change it. Do not delete or rewrite files outside your directory
(`port/INTERFACE.md`, `port/consts.py`, `port/validation.py`, `port/k2.py` and `README.md` are the lead's).

## Lead requests (newest first; helpers: check this list before each delivery)

- **04:55 → P3, P4: the lead is making the 04:40 hand-off edits in `moe/moe.py` and `gemv/moe_dense.py:stage_up`
  itself** (no delivery since 04:22; `k2.py` is blocked on them). The edits are additive and gated on `"poll" in c`
  (trace-time), so your standalone launchers and tests keep their launch-boundary hand-offs unchanged. Activation
  loads of announced scratch data (`xqb`, `xscb`, `interb`, `iscb`) take `c.get("xcm", 0)` as cache policy (K2:
  `CM_DEV`). If you edit these files, re-read them first and keep the `"poll" in c` branches.

- **04:35 → all: the tree was rolled back at ~04:22** (a snapshot from ~04:17 was restored over
  `/data/muse455/k2/ep1`): files created after the snapshot were deleted (e.g. `moe/test_moe.py`,
  `moe/test_moe_asm.*`, this file) and newer edits reverted. Check your directory against your notes and re-create
  what is missing. Keep a copy of your work outside the tree (your agent store) until merged.

- **04:40 → P3 (hand-offs, so `k2.py` can run the MoE stages in one launch).** Keep the stage arithmetic; change
  only where data comes from / goes to, using P4's formats (already implemented in `gemv/moe_dense.py`):
  1. `stage_xq(c, t)`: POOL is the peer region `c["peer"]["pool_own"]`, tagged pairs, written by P4's
     `pool_compute` with `put_bf(c["peer"]["pool"][p], t * MH + col, 4 bf16)`. Read row t with `c["poll"]`
     (same indexing as `put_bf`), not `args["pool"]`.
  2. `load_route(c)`: ROUTE is the scratch pair region `c["route"]`: pair `t * 16 + k` = expert id (int),
     pair `t * 16 + 8 + k` = weight (f32 bits), k in slot order. Poll it, not `args["route"]`; the finalize
     reads the weights from the same region.
  3. UG → down: after `stage_ug(c, slot, g)` stores INTER/ISC, `publish(c["put"], c["ugrdy"], slot * 32 + g, 1,
     tid == 0)` (scratch pairs, `MAX_U * 32`); `stage_down` polls the 32 flags of its slot before loading INTER.
  4. down → finalize (cross rank): write YR as tagged bf16 pairs with `c["put_bf"](c["yr_owner"][p], ...)` (MI355X
     style; the finalize polls `c["yr"]` pairs, no extra flags). If you measure plain stores + per-rank flags as
     faster, document it; correctness first.
  5. finalize: FMSQ is a scratch pair region (`c["fmsq"]`, put/poll). The FIN rows go **plain** bf16 into every
     rank's peer region `c["fin_peer"][p]` (P4's `stage_up` reads `c["fin"]` = this rank's copy with
     `dev_rows_to_lds`), then `publish(c["put"], c["frdy_peer"][p], t * 32 + j, 1, ...)` at system scope on every
     rank (256 FRDY flags: row t, slice j).
  6. Export as module constants: `UG_TASK_COUNT(nu)` rule, `DOWN_TILES`, `FIN_CTAS = 2 * 32`, and the LDS each
     stage needs (`rt`, `kred`, `amx` sizes) so `k2.py` sizes its LDS union.
- **04:40 → P4.** `stage_up` must poll **256** FRDY flags (`t * 32 + j`; row-owner finalize), not 32.
  `c["fin"]` lives in the peer buffer (remote owners write it); keep reading it with device scope after FRDY.

## 0. Ground rules

- **Scope: what C1 served.** Width 8 ran `K2Build(tokens=8, experimental_two_cta=True)` (the q8 build:
  `SGLANG_MUSE_MONO_EXPERT_STACK=1` → `k2_dispatch.default_key(8, experimental_two_cta=True)`); width 1 ran the
  default build. Port the stage functions both builds call (P4's README lists the q8 order: `stage_oproj_deep`,
  router/pool early tasks, `stage_ar_owner_sum`, `stage_seam_token`, `stage_route`, experts, `up_loads_masked`,
  `stage_up`, `stage_gather`). **CTA placement is the lead's** (k2.py): the q8 build's 512 co-resident CTAs may
  not fit at 512 threads on gfx1250, so stage functions must take a task index, not assume a CTA id.
  Drop k1, ep8_*, wide_*, k2_experiment, wave_splitk, and timeline debug points.
- **Keep the MI355X function names, argument lists and return conventions.** Where wave32 forces a different
  shape (e.g. an operand becomes 8 dwords instead of 4), keep the name and document the new shape in the module
  README and here. Do not rename regions.
- **Imports:** package root is `/data/muse455/k2/ep1` on `sys.path`; the merged tree imports as
  `from port.common.ops import ...`, `from port.sync.sync import ...`, `from port.consts import ...`
  (S4 rewrites `port.` to `sglang.srt.rocm_muse_spark.mono_gfx1250.`). Helpers' local test-time imports
  (`sys.path.insert` + `import mxq`) are fine; the lead rewrites them at merge.
- **Constants:** grid, model, TP and eps values come from `port/consts.py` (lead-owned). Do not redefine
  `BLOCKS/THREADS/WAVES/TP/H/MH/E/TOPK/RI/...` with different values; stage-local derived constants are fine.
- **Toolchain:** container `muse_mi455x`, FlyDSL 0.3.4.1, AITER `buffer_ops`. Run as
  `flock /data/muse455/locks/gpu<g>.lock docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=<g> python3 ...'`.
  P1/P3/P4: GPU 0 or 1 with its lock. P2: takes gpu0..gpu3 locks **in order** for peer tests, briefly.
  Before any GPU run: `docker exec muse_mi455x pgrep -f sglang.launch_server` and
  `tail -3 /data/muse455/campaign/c1/logs/driver.log`. If a server or AgentX bench is running, pause.
- **Deliverable per helper:** the ported modules, tests (each prints PASS/FAIL per check and writes a JSON),
  and `port/<dir>/README.md` (what changed vs MI355X, what is unverified). The lead reviews and merges.

## 1. Fixed gfx1250 / TP4 conventions

### Grid and waves
- `BLOCKS=256` (one CTA per CU, co-resident), **`THREADS=512`, `WAVE=32`, `WAVES=16`**, `LDS_BYTES=160 KB`
  (agreed by P1/P2/P4: thread-strided loops carry over; per-wave task maps double their wave count).
  At 512 threads LLVM caps a lane at 256 VGPRs; a build that spills is refused (`residency.problems`).
- `lane = tid % 32`, `wave = tid // 32` everywhere (no `% 64`).
- **64-lane numerical contracts:** carry a logical wave64 value as a `(lo, hi)` pair (logical lane L = physical
  lane L % 32, element L // 32); `xshfl/xred/butterfly/wave_sum/wave_max/wave_incl_scan/ballot` on a pair give
  the bit-identical gfx950 result. On a scalar they reduce 32 lanes; offset 32 on a scalar raises.

### Scaled WMMA (replaces `mfma_scale_f32_16x16x128_f8f6f4`), confirmed on hardware
`rocdl.wmma_scale_f32_16x16x128_f8f6f4(T.vec(8, T.f32), a, b, c, sa, sb, fmtA=, fmtB=, scaleAType=opsel_a, scaleBType=opsel_b)`
(fmt: 0 = FP8 e4m3, 4 = FP4 e2m1). Probes: `/data/muse455/k2/port/wmma_probe.py`, `/data/muse455/k2/ep1/s2/wmma_probe_fp8.py`.
- **FP4 operand, 8 dwords a lane:** lane l = row (A) / column (B) `l % 16`; dwords 0..3 = K chunk `l // 16`
  (32 fp4: k 32g..32g+31), dwords 4..7 = K chunk `l // 16 + 2`; low nibble = even k.
- **FP8 operand, 16 dwords a lane:** lane l = row/column `l % 16`; 16-byte chunk c (dwords 4c..4c+3) =
  K `32c + 16 (l // 16) + 0..15`.
- General rule (P1, bf16/fp8/fp4): lane l holds row/column `l % 16`; its 16-byte chunk j holds K run
  `2j + l // 16` (8 bf16, 16 fp8 or 32 fp4 per run).
- **Scales:** one dword a lane; byte j = E8M0 of K block j (32 wide) of row/column `l % 16` for one 128-k step.
  **opsel 0 reads lanes 0..15, opsel 1 reads lanes 16..31** (the FlyDSL docstring "lo/hi 16-bit half" is wrong).
  One dword a lane covers a pair of steps: lanes 0..15 hold step 2p, lanes 16..31 step 2p+1, opsel = step parity.
- **D (8 f32 a lane):** lane l = column `l % 16`, rows `8 (l // 16) + v`, v = 0..7.
  Store accumulators with `acc_store(red, acc, wave, lane)` (`red + (w·32 + lane)·8`); `row_sum` reads that.
- `mfma_bf16` = `v_wmma_f32_16x16x32_bf16` (16 bf16 a lane, 8 f32); `mfma_fp8` = `v_wmma_f32_16x16x64_fp8_fp8`.
- FP4×FP4 is exact (fp32 accumulate); bf16 ~3e-8. **FP8 products are not exact** (FP8×FP8 ~4-7e-5 relative,
  FP8×FP4 1e-8..1e-5 depending on data). A8 paths are tested with a tolerance (rel-L2 ≤ 3e-3 on bf16 outputs),
  A4 paths bit-exactly.

### Expert weights (lead's loader, S1 PASS)
- **EP1 slice per rank:** all 256 experts; w13 = [gate rows r·1024..+1024 ; up rows r·1024..+1024]
  → `[256, 2048, 4096/2]` e2m1x2, s13 `[256, 2048, 4096/32]`; w2 = down packed-K bytes r·512..+512
  → `[256, 4096, 1024/2]`, s2 `[256, 4096, 1024/32]`. Code: `/data/muse455/k2/ep1/ep1_layout.py`.
- **Weight bytes:** the MI355X single layout (AITER (16,16) preshuffle) **unchanged**. A lane reads two b128 per
  16-row group per 128-k step: dword `rg*(2K) + st*256 + half*128 + (l//16)*64 + (l%16)*4`, half 0/1
  (= `layout_gfx1250.w_dword`). `fp4_tile_load(r_w, base, rg, k, st, at)` returns these 8 dwords
  (two b128 at `at` and `at + 128`, `at = lane * 4`, TH_NT).
- **Scales:** gfx1250 order `[E, N/16, K/256, parity, 16 rows, 4 blocks]` (`layout_gfx1250.shuffle_scale_gfx1250`,
  one permute at load; `mx.scale_index_gfx1250`). A lane's dword for step st:
  `((e*(N/16)+rg)*(K/256) + st//2)*32 + l` (= `layout_gfx1250.s_dword`), opsel `st % 2`. Byte extraction
  (`(sw >> ...) & 0xFF`) in the MI355X stages goes away.
- FlyDSL reference for both expert stages (EP1 shapes, A4W4 exact / A8W4):
  `/data/muse455/k2/ep1/fly_expert_gemm_ep1.py`; harness `/data/muse455/k2/ep1/s2/ep1_moe.py`.

### Activations
- **Expert inputs (xe and inter) are MXFP8 e4m3** (S2: A4 fails the 0.997 floor on layer 60; A8 gives stream
  cosine ≥ 0.99981 on layers 1/31/60). Rule (OCP): per 32 values, `e = floor(log2(amax)) - 8`, code `e + 127`
  (amax 0 → code 0 and zeros), elements `RNE(x / 2^e)` to e4m3, saturating at ±448. Torch reference:
  `port/validation.py:quant_a8`. `mxq.py` keeps its names and takes an `act` argument (a4 / a8).
- Pre-expert norm output is rounded to bf16 before quantization (as served).
- gfx1250 has no `v_cvt_scalef32_pk_fp4_f32`: use `cvt_scalef32_pk8_fp4_f32` (P3: = torch E2M1 RNE);
  `cvt_pk_fp8_f32` = torch E4M3 RNE.

### Wave32 primitives (P1, delivered: `port/common/README.md`, 137/137 GPU checks)
- `xshfl(v, off)`: 1/2/4/8 one DPP, 16 `v_permlanex16`; 32 only on pairs. `ballot` returns Int64 with the
  upper 32 bits zero (a pair gives the 64-lane mask). `permlane_swap(16)` = `v_permlane16_swap_b32`.
- `block_max`, `block_excl_scan` need `WAVES` = 16 words; last lane 31.
- `fp8x8_bf16_pk` = `v_cvt_scale_pk8_bf16_fp8`; `memrealtime` = `s_sendmsg_rtn_b64 GET_REALTIME` (100 MHz,
  **not synchronized across CTAs**: compare stamps only within a CTA); `vm_wait()` replaces `s_waitcnt vmcnt(0)`.
- New names: `WAVE, ACC_PER_LANE, ACC_WORDS, acc_slot, acc_store, vm_wait, mx.scale_index_gfx1250`.
- P1's README §3 has the table of stage call sites whose meaning changes (seam butterfly with 32, `wave_sum`
  64-lane sums, router ballots over `lane + 64 i`, `tid % 64`, accumulator stores, `mfma_scaled` byte scales).
- Summation order is a numerical contract: where the MI355X code matched a served kernel's reduction order,
  match the **gfx1250 served** kernel or document the ulp difference.

### Memory model and mailbox (P2; S0 measurements in `/data/muse455/k2/ep1/s0/`)
- Cache policy (ISA-verified by P1 and P2): **`CM_DEV = 16` (SCOPE_DEV), `CM_SYS = 24` (SCOPE_SYS),
  `CM_NT = 1` (TH_NT)**. MI355X literals are wrong here (17 = DEV|NT, 2 = TH_HT): use the names only.
  `rocdl.s_waitcnt` raises on gfx1250 → `vm_wait()` / `memmodel.wait_stores()`.
- `publish` adds a release fence at the pair's scope (`global_wb scope:SCOPE_DEV/SYS`): the per-XCD L2 is not
  device-coherent for plain stores.
- Peer memory: one uncached buffer per rank (`hipExtMallocWithFlags(hipDeviceMallocUncached)` + IPC),
  `sync/peer.py:PeerRuntime`.
- **Measured costs (4×MI455X, uncached):** one-way push hop 2.33-2.40 µs with sys-scope release/acquire,
  **2.02 µs relaxed** (value+tag, no fence); pull (poll the peer) 3.6 µs; same-GPU hop 1.0 µs; 4-rank push
  barrier 3.6 µs; **tagged all-to-all exchange 3.4-3.9 µs up to 128 KB per peer** with 64-256 CTAs (512 KB/peer
  18.5 µs); **a 256-CTA device-scope grid fan-in costs ~2.6 µs** (16 CTAs: ~1 µs).
- Protocol rules: (a) tagged pairs `{value, tag}` in one 64-bit store, consumer polls until all tags match
  (no torn pairs in 3×10⁶ epochs); (b) push into the consumer's memory, poll locally; (c) reuse a peer parity
  buffer only after a fan-in or launch boundary; (d) no grid-wide barriers on the critical path, targeted flags;
  (e) every wait bounded (`WAIT_SECONDS`), fail-stop as MI355X.
- Tag: `mailbox_tag(epoch, slot)` unchanged (`(epoch % 2^22) * 256 + slot + 1`; K2 slot = 128 + layer).
- Pitfalls: `from __future__ import annotations` breaks FlyDSL kernel pointer arguments unless `Int64`/`Int32`
  are module-level names (truncated addresses, GPU fault). Kernel names must not contain `-`.

### TP4/EP1 dataflow (what k2.py assembles; torch model in `port/validation.py:k2ref`)
1. o_proj: `[T, 2048]` core × this rank's W_o columns → bf16 partials → AR#1 to the column owners.
2. router (fp32, this rank's 64 experts) and pooler (bf16, this rank's 1024 pooled columns) → `rlog`, `pool`.
3. seam B: AR#1 rank-ordered sum → post_attn_norm → gated residual → `stream_out` → pre_ffn_norm → x rows.
4. route (every rank, all T rows, identical): sigmoid(logit·mult) + bias, top-8 (ties → lower id), weights s/(Σ+1e-15).
5. xq: pooled → pre_expert_norm → bf16 → MXFP8.
6. UG: union U (≤ 64) × this rank's 1024 intermediate columns; silu(g)·u → bf16 → MXFP8 inter per pick.
7. down: every expert of U × its picks' inter, K = 1024 → bf16 partial per route (unweighted) → **row owner**
   `row_owner(t) = t // 2` (`yr` `[src][t][k][MH]` bf16; 128 KB/peer at T=8).
8. finalize (owner): YR summed over ranks **in rank order** (fp32) → bf16 y; v = bf16(y·bf16(w_post+1));
   sc_k = w_k / sqrt(mean v² + 1e-8); fin = bf16(Σ_k sc_k·v_k).
9. FIN all-gather → unpool (this rank's 2048 branch columns) → AG#4 → gather → `branch_out`.
10. finish consensus. The FFN boundary stays a native kernel.

## 2. Module contracts

### P1 `port/common/` — `ops.py`, `mx.py` (delivered)
All MI355X names kept (see P1 README §1 for the 29 symbols the stages import). `mfma_scaled(a, b, c, sa, sb,
a_fmt, b_fmt, sa_byte, sb_byte)`: operands 8 (FP4) / 16 (FP8) dwords, `sa/sb` scale dwords, `sa_byte/sb_byte` =
opsel (0/1), returns 8 f32.

### P2 `port/sync/` — `sync, residency, epoch, plan, ranks, peer, abi, failstop, tp4, memmodel`
- `sync.py`: `Mailbox(layer, health, alarm, region_ids)` with `put, put_bf, put_words, poll(specs, batch),
  poll_terminal`; `Addr, sreg, preg, shift, publish(put, a, i, v, who), import_alarms, ld_health, stall`.
- `plan.py`: `BLOCKS=256, THREADS=512, WAVE=32, WAVES=16` (= `port.consts`), the rest unchanged.
- `peer.py`: `PeerBuffer`, `tp_agree`, `bind_agreed`, new `PeerRuntime(nbytes, group, rank, tp, device, *, blocks)`.
- `tp4.py`: TP4 constants and owner maps (`ar_owner`, `row_owner`, `fin_slice`, ...), `PLACEMENT_CONFLICTS`.
- Pending: the 4-rank test result (`test_sync_4rank.py`, all four GPU locks).

### P3 `port/moe/` — `moe.py`, `mxq.py`
- `stage_xq(c, t)`, `load_route(c)`, `stage_ug(c, slot, g)`, `stage_down(c, slot, rgd)`, `stage_fin(c, tl, j, upto)`
  over the context dict; TP4/EP1 constants; row-owner YR. Hand-offs are plain stores to `c` regions today
  (a launch boundary per hand-off). **Request:** now that `port/sync/sync.py` exists, add the in-kernel hand-offs
  (`c["put"]` / `c["poll"]` / `publish`, MI355X region names) so the stages can run in one launch; expose task
  counts (UG tasks, down tasks, fin CTAs) as module constants for the lead's CTA map.

### P4 `port/gemv/` — `gemv, oproj, seam, seam_b, quarter_seam, moe_dense`
- `stage_oproj_deep(c, bid)`, `stage_oproj`, `stage_ar_owner_sum(c, bid)`, `stage_seam_token(c, t)`,
  `router_loads/router_compute(_pair)`, `pool_loads/pool_compute`, `stage_route(c, t)`, `up_loads(_masked)`,
  `stage_up(c, j, ops)`, `stage_gather(c, j)`, `peer_pairs(s)`, `scratch_pairs(s)`.
- At merge the lead replaces `gemv/common.py` and `gemv/mailbox.py` by `port.common.ops` / `port.sync.sync`.

### Lead `port/k2.py`, `port/validation.py`, loader, runner
- `k2.py`: `K2Build(tokens, eps_*, router_output_multiplier=consts.ROUTER_MULT, ...)`, `make_key`, `ABI`
  (MI355X argument names: `core, stream_a, w_o, alpha_a, beta_a, w_fn, w_router, expert_bias, w_pool, w_pre,
  w13, w13s, w2, w2s, w_post, w_unpool, stream_out, branch_out, scratch, peers, rank, epoch, layer, health,
  alarm, tl`), CTA role map at 256 CTAs, `build`, `k2_launch`, `scratch_layout`, `peer_layout`.
- `validation.py`: `LayerData(layer)`, `make_case`, `calibrate`, `k2ref`, `stock`, `boundary`, `check(outs, case,
  lay, ref, st)`; bars: stream2 cosine ≥ 0.997 floor / 0.999 target, routes equal except near-ties.

## 3. Open items

| item | owner |
|---|---|
| 4-rank sync test result | P2 |
| in-kernel hand-offs in the MoE stages | P3 |
| CTA role map at TP4: MI355X placement breaks (pooler 80+256 > 256 CTAs; unpooler U0=128 overlaps gather 128..191, `sync/tp4.py:PLACEMENT_CONFLICTS`); q8 512-CTA residency on gfx1250 | lead |
| re-run of S0 no-fan-in exchange and custom-AR numbers on idle GPUs | P2 (when all 4 GPUs are free) |
