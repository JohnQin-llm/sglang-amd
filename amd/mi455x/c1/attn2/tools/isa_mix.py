"""Classify the instructions of an amdgcn loop body (lines a..b) into issue classes."""
import re, sys, collections
path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(path).read().split("\n")[a - 1:b]
c = collections.Counter(); ops = collections.Counter()
for l in lines:
    l = l.strip()
    if not l or l.startswith((";", ".", "//")) or l.endswith(":"):
        continue
    op = l.split()[0]
    ops[op] += 1
    if op.startswith("v_wmma"): k = "wmma"
    elif op.startswith(("global_load_async", "global_load_lds", "cluster_load_async")): k = "async->LDS"
    elif op.startswith("tensor_load"): k = "tdm"
    elif op.startswith(("global_load", "buffer_load", "flat_load")): k = "vmem load"
    elif op.startswith(("global_store", "buffer_store")): k = "vmem store"
    elif op.startswith("ds_load") or op.startswith("ds_read"): k = "ds read"
    elif op.startswith("ds_store") or op.startswith("ds_write"): k = "ds write"
    elif op.startswith("ds_"): k = "ds other"
    elif op.startswith("v_cvt") or "cvt" in op: k = "v_cvt"
    elif op.startswith(("v_exp", "v_log", "v_rcp")): k = "transcendental"
    elif op.startswith(("v_max", "v_min")): k = "v_max/min"
    elif op.startswith(("v_permlane", "v_mov_b32_dpp")) or "dpp" in l or op.startswith(("v_swap", "ds_swizzle", "ds_bpermute")): k = "cross-lane"
    elif op.startswith(("v_cndmask", "v_cmp")): k = "v_cmp/cndmask (mask)"
    elif op.startswith(("v_lshl", "v_add_co", "v_add_nc_u64", "v_mad_u64", "v_mad_co", "v_lshrrev", "v_and_b32", "v_or_b32", "v_mul_u32", "v_mul_hi", "v_add_nc_u32", "v_sub_nc_u32", "v_bfe", "v_alignbit", "v_lshl_add", "v_add3", "v_perm", "v_mul_lo", "v_ashr", "v_lshlrev", "v_xor", "v_bfi", "v_lshl_or", "v_and_or")): k = "v int/addr/pack"
    elif op.startswith(("v_fma", "v_mul_f32", "v_add_f32", "v_sub_f32", "v_pk_mul_f32", "v_pk_fma_f32", "v_pk_add_f32", "v_mul_f", "v_add_f", "v_fmac", "v_pk_")): k = "v fp32 math"
    elif op.startswith("v_mov") or op.startswith("v_readfirstlane") or op.startswith("v_writelane") or op.startswith("v_readlane"): k = "v_mov/readlane"
    elif op.startswith("s_wait") or op.startswith("s_barrier") or op == "s_nop" or op.startswith("s_delay"): k = "wait/barrier/nop"
    elif op.startswith("s_"): k = "salu/smem/branch"
    elif op.startswith("v_"): k = "valu other"
    else: k = "other"
    c[k] += 1
tot = sum(c.values())
print(f"total {tot}")
for k, v in c.most_common():
    print(f"  {k:24s} {v:5d}  {100*v/tot:5.1f}%")
print("top ops:", ", ".join(f"{o} {n}" for o, n in ops.most_common(45)))
