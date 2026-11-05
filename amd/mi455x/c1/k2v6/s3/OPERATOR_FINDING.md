# Operator finding, 05:08 UTC: K2 fails only when consecutive launches overlap across ranks

`s3/probe_repeat.py` (copy of run_k2.py; layer 31, a8, same operands):

- 5 launches with `dist.barrier()` + `torch.cuda.synchronize()` between them: `rt.check()` passes.
- 5 launches back to back, no host barrier: `MonoKernel step failed: TP ranks [0, 1, 2, 3] gave up an in-kernel wait`.
- The graph-replay timing in run_k2.py fails the same way (05:05 run).

So the single-launch numerics are fine; the bug is the launch-to-launch hand-off: a rank that finishes
launch N and starts N+1 (epoch advanced) can reach a mailbox region or the finish consensus before a slower
peer is done with the same region for launch N, or the terminal done pairs / rank-0 header of launch N are
still being read. The served path launches K2 back to back on every layer (no host barrier), so this
blocks S4 as well as the S3 timing. The sync port's 4-rank test never ran exchanges without a launch
boundary between epochs (its open item).

Priority: fix this, then the per-layer timing (graph of R launches) is the number the operator waits for.

## Where to look (operator, 05:14)

The MI355X design already tolerates barrier-free launches (epoch tags on every word, peer regions
double-buffered by parity, finish consensus keeping ranks within one launch), and port/k2.py shifts every
_PEER_PAIRS region by parity. So look at what is new on gfx1250 / EP1:
1. Visibility: gfx1250 plain stores can stay dirty in one XCD's L2 until a release (sync port finding). With a host
   barrier the kernel boundary flushes everything, which would hide a missing release. Check every cross-rank
   store in the new code paths (moe.py hand-offs: YR tagged pairs to the row owner, FIN stored plain into every
   rank's peer copy, FRDY at system scope; gemv stage_up polling 256 frdy flags) goes through publish / a
   release fence and is polled at system scope.
2. Regions outside the parity halves: anything a peer writes that is not in _PEER_PAIRS (e.g. done in rank 0's
   header, xrow/xbuf, scratch regions a peer can reach), and any flag polled with a tag that does not include
   the epoch (frdy_flags count 256 vs 32).
The health / alarm records name the first missing pair: read them after the no-barrier probe to get the region.
