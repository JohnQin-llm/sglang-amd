# K2 MoE stage, gfx1250 (MI455X), TP4 / EP1, 8 rows: work in progress

Status log (newest last):
- probe_ops.py: gfx1250 has no `v_cvt_scalef32_pk_fp4_f32` (LLVM asserts); `cvt_scalef32_pk8_fp4_f32` with
  scale 2^(code-127) = torch E2M1 RNE rule exactly; `cvt_pk_fp8_f32` = torch E4M3 exactly.
- moe.py / ref.py / test_moe.py written; layer 31, 1 set, A4: every stage bit-exact (xq, inter, isc, yr, fin,
  replicas, end-to-end). Next: full layers 1/31/60, stock cosine, timing, A8 / asm rules.
