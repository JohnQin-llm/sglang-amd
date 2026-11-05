# PDL-style early launch on MI455X (gfx1250), ROCm 10.2.0a20261013 / HIP 7.17.26405

**Correction (2026-11-05, `bench/RESULT.md`):** the graph rows below did not reset the flag inside the graph, so after
the first replay the wait passed on the previous value; they cannot separate early start from a stale flag. With the
flag reset inside the graph: release latency 1.1-1.4 us (eager, classic path, segment + 2 queues), and default
scheduling depends on the graph's shape (a 1-layer graph serialized; a 62-layer graph started the consumer early in
62 of 62 layers). The eager results and the API findings stand. For QKV behind K2 the mechanism is a no-go: K2's
251 VGPRs x 512 threads fill the VGPR file, no consumer CTA is resident beside it (co-residency needs <= 224), the
optimistic arm is 4.7-5.6 us per layer slower than linear, and early release can block K2's own CTAs (deadlock).

Test: `pdl_test.hip` (`run.py` through `k2safe.sh`, GPU 0). Producer A: 8 blocks, spins, block 0 writes a flag
at its start ("trigger"), every block increments a done counter at its end. Consumer B: 8 blocks; each records
the done counter when it starts (clock-free; `wall_clock64` is not comparable across XCDs).

| mode | result |
|---|---|
| same stream (control) | B starts after all 8 A blocks finished |
| `hipLaunchKernelExC` with launch attribute id 6 (CUDA's `ProgrammaticStreamSerialization`) | `invalid argument`: not implemented |
| B on a second stream behind `hipStreamWaitValue32(flag >= 1)`, eager | **B starts while A runs** (0 of 8 A blocks done), every iteration |
| same, captured into one HIP graph, default settings | B waits for A (graph branches serialized) |
| graph + `DEBUG_HIP_FORCE_GRAPH_QUEUES=2` or `4`, or `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` | B waits for A |
| graph + `DEBUG_HIP_GRAPH_CLASSIC_PATH=1` | **B starts while A runs** |
| graph + `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=2` | **B starts while A runs** |

So the PDL effect (a dependent kernel released by the producer's mid-kernel signal) works on gfx1250 today:
producer writes a flag at its trigger point = `cudaTriggerProgrammaticLaunchCompletion`; the consumer's stream
waits on it with `hipStreamWaitValue32`; inside the consumer, data written after the trigger still needs its own
device-scope flag (the counterpart of `cudaGridDependencySynchronize`). There is no CUDA PDL API
(attributes 6/7 rejected; graph edge data unsupported), no `griddepcontrol`-like ISA instruction in this LLVM,
and AITER / Triton (AMD backend) / FlyDSL / SGLang do not use PDL.

Not yet measured: the release latency (flag write -> consumer start), the cost of the graph settings above on a
whole serving graph, and co-residency: the consumer needs free CU resources while the producer runs (K2 holds
one 512-thread CTA per CU with 145 KB LDS; a consumer CTA fits only if its LDS + VGPRs fit beside it; K1 at
311 KB LDS does not). Use epoch values (wait for flag >= epoch) instead of resetting flags between replays.
