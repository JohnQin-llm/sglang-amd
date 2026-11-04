# Lane C: K2 ready to serve (patch, stronger test, end-to-end plan)

Lane C copy: `rsync -a --exclude __pycache__ /data/muse455/k2/ep1/ /data/muse455/k2/lanes/c/` (07:00 UTC).
The hard-coded `/data/muse455/k2/ep1` was rewritten to `/data/muse455/k2/lanes/c` in every `.py`, `.sh` and
`.json` file. Docs and logs were left as they were, as in lanes a/b. The kernel tree is unchanged (sync
defaults `K2_POLL_SLEEP=32`, `K2_INV_POLL=0`). Nothing in ep1, lanes a/b or `/data/muse455/sglang` was edited.

## Status

| item | state |
|---|---|
| S4 patch, current served tree | **ready**: `s4/k2_gfx1250.patch` (2 files, 481 lines). `git apply --check` OK on a scratch copy of served HEAD d558a20d48 + the working-tree diff (11 files) + 2 untracked files |
| port tree selectable at run time | **ready**: `SGLANG_MUSE_MONO_PORT` (default `/data/muse455/k2/ep1`). The patch vendors no port code. The adapter imports ep1 and lanes a, b, c without error, all with the same 26-name ABI |
| patched tree to serve | **ready**: `sglang_k2/` (rebuilt by `s4/make_patch.py --deploy`). The full model module imports from it with a GPU visible |
| stronger K2 correctness test | **ready, passes**: `s3/test_b2b_varied.py`. 189 launches, all pass, all bitwise equal to their golden launch |
| locked server start/stop | **ready**: `bench/start_probe_server_locked.sh` (dry-run tested) |
| end-to-end plan | **ready, not run**: `bench/k2_e2e_plan.sh <port tree>` |
| K2 in a live server | **not run**: needs ~45-60 min of exclusive GPUs |

**The one command**, once a lane's K2 beats ~130-140 us/layer:

```
bash /data/muse455/k2/lanes/c/bench/k2_e2e_plan.sh /data/muse455/k2/lanes/a     # or lanes/b, ep1
```

## 1. The patch (`s4/`)

The draft patch carried 31 files: a vendored copy of the port with its imports rewritten. It also read the router
as `moe.gate.weight` / `moe.gate.e_score_correction_bias`. The served module has `moe.router.weight`, stored
transposed as `[8192, 256]` fp32 by the loader, and `moe.expert_bias`, so the draft would have failed in
`post_load_weights`. Its INSTRUCTIONS also told users to lower `--mem-fraction-static` by 0.24; that is not
needed (see §3). The draft is kept as `s4/k2_gfx1250.patch.s4draft`.

The new patch has two parts:

- **`rocm_muse_spark/mono_gfx1250.py` (new, the adapter).**
  - Port import: `PortOps` puts `SGLANG_MUSE_MONO_PORT` first on `sys.path`. It refuses if a different `port`
    package is already imported. It uses only the entry points the port's own harness `s3/run_k2.py` uses:
    `k2.K2Build/build/k2_launch/peer_bytes/scratch_bytes`, `sync.peer.PeerRuntime`,
    `sync.epoch.advance_epoch_`, `layout.gfx1250_weights`, `gemv.gemv.shuffle_rows`. A tree that changes
    operand layouts can override with `port/serving.py`.
  - Operands: `check_layer` checks the served shapes and constants against the K2 build: every norm's eps,
    zero-centered gamma, gain center 1, residual-gate temperature, router multiplier, route scale and eps, top-k.
    It raises a named error on a mismatch.
  - EP1 copy: `load_ep1_experts` reads the checkpoint with safetensors slices (only this rank's quarter) and
    converts to the gfx1250 layout on the GPU, 32 experts at a time.
  - `K2Runner`: `takes_step` / `takes` (target verify, 8 rows, a taken layer), `begin_step`, `run`.
  - Monitoring: rank 0 logs the K2 step count (the device epoch, so graph replays count) and the ranks' failure
    flags every 60 s. It copies them on a side stream into pinned memory, and never during graph capture.
- **`model.py` hooks.** Each anchor is matched against the current code, exactly once:
  - `MuseSpark1xAttention.forward(core_only)` returns the `_fused_o_norm_gate` output, i.e. the o_proj input,
    `[8, 2048]` bf16.
  - `MuseSpark1xDecoderLayer.forward`: when `sp is None`, the FFN boundary is fused and the runner takes the
    layer, it runs attention core, then K2, then `ffn_boundary.forward(stream, branch)`. This is the same call
    and dtypes the served TP4 path uses (its unpool-boundary fusion is TP8-only).
  - `MuseSpark1xModel.forward`: `begin_step()` before layer 0.
  - `MuseSparkTextForCausalLM.post_load_weights`: `build_runner(self)`.

  `MuseSparkForConditionalGeneration` (the served class) reaches it through `super()`. A second call keeps the
  first runner, so memory is never doubled.
- **Hook-site checks against the current model code:**
  - All 62 layers are `sparse`. Layers 0 and 61 have BF16 experts (Quark `exclude`), so K2 takes 1..60.
  - The constants match the checkpoint config: `rms_norm_eps` 1e-5, `post_norm_eps` 1e-8, temperature 0.3,
    `output_multiplier` 0.1768, `routed_scaling_factor` 1.
  - The FFN boundary returns new tensors, so K2's shared `stream_out` buffer is never read and written by the
    same launch.
  - The served EP4 experts stay loaded and run every step K2 does not take.
- **Graph capture.** `begin_step` (a `_assert_async` plus an in-place add) and the K2 launch are plain work on
  the current stream, so the verify graph captures them. Only bs=1 (8 rows) qualifies; bs>=2 verify graphs
  (16+ rows) keep the served path. The test captures exactly this sequence: `K2Runner.begin_step` + `run`.
- **Parse and import.** Both patched files byte-compile. The patched `model.py` imports from `sglang_k2` inside
  the container (`IMPORT OK ... core_only True`, under the GPU locks).

## 2. Correctness test with different inputs every launch (`s3/test_b2b_varied.py`)

**Cases.** 9 cases: layers 1, 31, 60 times seeds 0, 1, 2. Each case has its own `core` / `stream_a` rows,
calibrated to 43 distinct experts.

**Rotation.** Cases rotate so that a launch never has the same inputs as the launch before it or the one two
before (the same epoch parity). The same layer never sees the same seed in two steps less than three apart.
Distinct cases differ by at least 5.8 (max abs) in `stream_out`, so a stale read would show.

**Recording.** Outputs are pre-filled with NaN. Every launch writes its own buffers, or is copied out after its
graph replay. Route ids are copied out of the scratch after each launch.

**Phases.** Six, all back to back with no host sync:

| phase | what |
|---|---|
| `eager_step` | serving pattern: one epoch advance a step, then layers 1, 31, 60 at the same epoch (slots 129, 159, 188) |
| `eager_launch` | harness pattern: an epoch advance before every launch |
| `graph_step`, `graph_launch` | the same as HIP graphs; 4 replays back to back, static inputs refilled on the stream between replays, as the CUDA-graph runner does |
| `runner_eager`, `runner_graph` | the adapter's own `K2Runner` (operands from served-like modules via `add_layer`, all 256 experts), `begin_step` + `run` in serving order, eager and graph-captured |

Before the serving pattern, K2 had only been tested one layer per epoch.

**Grading.** Every launch is graded with `validation.check` against stock: the stream2 cosine bar is 0.997 floor
/ 0.999 target, and routes must be equal except near-ties. It is also graded against `k2ref`, and compared
bitwise with an isolated, barrier-separated golden launch of its case.

**Result (07:15 UTC, all four locks, 23 s wall; `s3/test_b2b_varied.json`, `.log`):**

| phase | launches | pass floor / target | bitwise = golden | stream2 cos min | branch cos min vs k2ref | in-kernel give-ups |
|---|---:|---:|---:|---:|---:|---|
| eager_step | 27 | 27 / 27 | 27 | 0.999795 | 0.999873 | 0 on every rank |
| eager_launch | 27 | 27 / 27 | 27 | 0.999795 | 0.999873 | 0 |
| graph_step | 36 | 36 / 36 | 36 | 0.999795 | 0.999873 | 0 |
| graph_launch | 36 | 36 / 36 | 36 | 0.999795 | 0.999873 | 0 |
| runner_eager | 27 | 27 / 27 | 27 | 0.999795 | 0.999873 | 0 |
| runner_graph | 36 | 36 / 36 | 36 | 0.999795 | 0.999873 | 0 |

Every launch is bitwise equal to its golden launch, so each launch's cosines are its case's golden cosines:

| case (layer, seed) | stream2 cos min | branch cos vs stock | branch cos vs k2ref | routes |
|---|---:|---:|---:|---|
| 1, 0 / 1 / 2 | 0.999857 / 0.999910 / 0.999928 | 0.99867 / 0.99897 / 0.99950 | 0.99996 / 0.99995 / 0.99999 | equal |
| 31, 0 / 1 / 2 | 0.999923 / 0.999912 / 0.999917 | 0.99921 / 0.99923 / 0.99921 | 0.99994 / 0.99996 / 0.99998 | equal (seed 0: rows 0, 2 near-ties, allowed) |
| 60, 0 / 1 / 2 | 0.999795 / 0.999800 / 0.999822 | 0.99899 / 0.99896 / 0.99910 | 0.99987 / 0.99998 / 0.99997 | equal |

Other results from the same run:

- **Adapter operands.** The served-adapter operands (router given as the served `[H, E]` layout, all 256 experts
  from the checkpoint) are bitwise equal to `run_k2.py`'s operands, for all 14 tensors on all 4 ranks and 3
  layers.
- **Layer selection.** `takes` is true for verify at 8 rows only: false for 16 rows, extend at 8 rows, and
  layers 0 and 61.
- **Timing.** Graph replay in the serving pattern is 205-208 us per launch (harness pattern 208-211).
- **Load time.** The EP1 load is 1.3-2.0 s per layer per rank with a warm page cache and 8 s cold.

Re-run (about 1 min). `K2C_PORT_ROOT=/data/muse455/k2/lanes/a` (plus `PYTHONDONTWRITEBYTECODE=1`) tests
another lane's tree read-only:

```
cd /tmp && flock /data/muse455/locks/gpu0.lock flock /data/muse455/locks/gpu1.lock flock /data/muse455/locks/gpu2.lock \
  flock /data/muse455/locks/gpu3.lock docker exec muse_mi455x bash -c 'pgrep -f "[s]glang.launch_server" && exit 9; \
  cd /tmp && PYTHONSAFEPATH=1 PYTHONPATH=/data/muse455/k2/lanes/c timeout 900 python3 /data/muse455/k2/lanes/c/s3/test_b2b_varied.py'
```

## 3. Memory (per GPU, TP4)

The extra memory is the EP1 expert copy (all 256 experts, this rank's quarter of every intermediate) plus
three shuffled dense copies. The runner's measured extra bytes for 3 layers, 5,675,677,568, match this count
exactly.

| item | bytes a layer | 60 layers |
|---|---:|---:|
| w13 `[256, 2048, 2048]` u8 | 1,073,741,824 | |
| s13 `[256, 2048, 128]` u8 | 67,108,864 | |
| w2 `[256, 4096, 512]` u8 | 536,870,912 | |
| s2 `[256, 4096, 32]` u8 | 33,554,432 | |
| **EP1 expert copy** | 1,711,276,032 | **102.68 GB (95.63 GiB)** |
| w_o / w_pool / w_unpool shuffled copies (bf16) | 33,554,432 + 67,108,864 + 67,108,864 | |
| router `[E, H]` fp32 (served is `[H, E]`), alpha/beta | 8,388,608 + 65,536 | |
| **dense copies** | 176,226,304 | **10.57 GB (9.85 GiB)** |
| peer buffer 12,103,680 + scratch 564,064 + health + out buffers (once) | | 13.2 MB |
| **total extra** | | **113,263,310,720 B = 113.26 GB = 105.49 GiB** |

- Transient during load: one 32-expert chunk, native plus shuffled, about 0.43 GB, freed before KV sizing.
- The other norm and bias operands alias the served parameters.
- The served EP4 experts stay loaded, so the expert weights are held twice (EP4 slice plus EP1 slice, each about
  1.71 GB a layer). The rest of the weights are not duplicated.

**How it fits** (served log of 05:10, TP0, GiB as sglang prints them):

| | served today | with K2 |
|---|---:|---:|
| free before load | 422.37 | 422.37 |
| target weights | 154.73 | 154.73 + 105.49 = 260.22 |
| free after target + draft (3.63) | 264.00 | 158.51 |
| KV pools (target + draft) | 121.99 + 24.20 = 146.19 (10.15 M full-attention / 2.03 M SWA tokens) | about 40.7 (about 2.8 M full / 0.57 M SWA tokens) |
| free once idle (after pools + graphs) | 99.48 | about 99.5 (unchanged) |

- SGLang sizes the KV pool after `load_weights` (`kv_cache_configurator._profile_available_bytes`: free memory
  minus `pre_load x (1 - 0.85)` and the multimodal reserve). `build_runner` runs inside `load_weights`, so the
  K2 extra is charged to the KV pool. The non-static slack and the idle headroom do not change.
- **It fits at `--mem-fraction-static 0.85` unchanged.** The KV pool shrinks about 72%, which still holds a
  1M-token request and is far above what the batch-1 300K sweep needs. Raising the fraction to 0.88-0.90 would
  give back 13-21 GiB of pool if more is wanted.

## 4. End-to-end plan (`bench/`)

- **`start_probe_server_locked.sh` (start / stop / status).** The old script was run under `flock` and returned
  once the server was healthy, so the locks were released while the server still owned the GPUs.
  - A detached host holder takes `gpu0..3.lock` in order, starts the server, and keeps the locks until the
    server's pid is definitely gone. A failed `docker exec` does not release them.
  - Same `/tmp/arbor-server-30000.pid` semantics as the old script. `stop` signals only the process group of
    the pid its own holder started (recorded in `bench/run/server.pid`). It never pkills.
  - Refuses if port 30000 answers, a holder runs, or a `launch_server` process exists.
  - `gen_launch.py` writes the launch script from `arbor/config.json` (`server_env` + `server_flags`) plus
    `SGLANG_MUSE_MONO_GFX1250`, `SGLANG_MUSE_MONO_PORT`, `MUSE_FUSE_FFN_BOUNDARY=1`, and
    `PYTHONPATH=lanes/c/sglang_k2/python`. `REAL_ACCEPT=1` drops `SGLANG_SIMULATE_ACC_*`.
  - Tested with `DRY_RUN_CMD="sleep 300"`: all four locks held after `start` returned, released only after
    `stop` ended the process.
- **`k2_e2e_plan.sh <port tree>` (not run; about 45-60 min).**
  - **Gate:** the tree's own `s3/run_k2_a8.json` must be at or below 140 us/layer (`FORCE=1` skips it).
  - **Rebuild:** `make_patch.py --deploy`, i.e. re-check the patch against the served tree as it is then.
  - **K2 test:** `test_b2b_varied.py` on that tree; everything must pass.
  - **K2 on, real acceptance:** startup lines (4 runners, memory, KV tokens); `correctness.sh` must score 15/16;
    accept length at 32K; "K2 steps so far" must be above 0 and no failure flag.
  - **K2 off, real acceptance:** the accept-length baseline (`REAL_OFF=0` skips it).
  - **K2 on, served config:** `correctness.sh` with `GSM8K_MIN=0.25`, then `ctx_sweep.py 1000 32000 128000
    300000`.
  - **K2 off, served config:** the same tree with `SGLANG_MUSE_MONO_GFX1250=0` (the patch is inert, identical
    to the served tree): correctness and the sweep.
  - **Compare:** `compare.json` / `compare.md` with decode tok/s and ms/token per context, the on/off ratio,
    accept lengths and correctness scores.
  - Results go to `bench/e2e/<UTC>_<tree>/`. A failed step stops the server and the plan.

## Incident (07:10 UTC): an SGLang server was started by mistake

While I was testing a "generate only" option of the start script, the script file had been overwritten by an
older version that lacked the option. That happened because one out-of-band edit and later in-editor edits
conflicted. So `start` really launched the K2-on server, with `SGLANG_MUSE_MONO_PORT=/data/muse455/k2/ep1`.

- It ran from 07:10:13 to 07:11:14 UTC and was stopped during weight loading, before the K2 build.
- It was stopped through its own pid file (process-group TERM). No sglang processes or GPU processes remain.
- It held all four GPU locks for its whole life, so no K2 test overlapped it.
- I removed its stale `/tmp/arbor-server-30000.pid` (it named the dead pid).
- Since then the launch script is generated by a separate `gen_launch.py`, and every script run checks the
  file's content first.

Logs: `bench/run/holder_k2on_gen.log`, `bench/run/server_k2on_gen.log`.

## Open risks for the first live run

- Not yet exercised in a server:
  - K2's 256 co-resident CTAs next to the served streams (the graph's metadata-prep fork, NCCL / custom-AR);
  - real attention outputs as `core`.

  A K2 wait that gives up returns garbage after 10 s. The rank-0 reporter logs it, and the plan fails on it.
- The mailbox tag has 22 bits of epoch, one advance a verify step. That is about 4.2 M steps, roughly 19 h at
  60 steps/s; after that `advance_epoch_` asserts and the server must be restarted.
- Startup cost: about 2 min of extra load per server with a warm page cache, up to about 8 min cold, plus the K2
  compile (cached by source digest).
