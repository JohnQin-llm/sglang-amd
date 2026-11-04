	.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.section	.text._Z1kPKDv16_iS1_PKiS3_PDv8_f,"ax",@progbits
	.protected	_Z1kPKDv16_iS1_PKiS3_PDv8_f ; -- Begin function _Z1kPKDv16_iS1_PKiS3_PDv8_f
	.globl	_Z1kPKDv16_iS1_PKiS3_PDv8_f
	.p2align	8
	.type	_Z1kPKDv16_iS1_PKiS3_PDv8_f,@function
_Z1kPKDv16_iS1_PKiS3_PDv8_f:            ; @_Z1kPKDv16_iS1_PKiS3_PDv8_f
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[4:11], s[0:1], 0x0 nv
	v_lshlrev_b32_e32 v26, 6, v0
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x20 nv
	s_wait_kmcnt 0x0
	s_clause 0x5
	global_load_b128 v[14:17], v26, s[6:7] offset:48
	global_load_b128 v[10:13], v26, s[6:7] offset:32
	global_load_b128 v[22:25], v26, s[4:5] offset:16
	global_load_b128 v[6:9], v26, s[6:7] offset:16
	global_load_b128 v[18:21], v26, s[4:5]
	global_load_b128 v[2:5], v26, s[6:7]
	s_clause 0x1
	global_load_b32 v1, v0, s[8:9] scale_offset
	global_load_b32 v34, v0, s[10:11] scale_offset
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v0, 5, v0
	s_wait_loadcnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[26:33], v[18:25], v[2:17], 0, v1, v34 matrix_a_fmt:MATRIX_FMT_FP4
	s_clause 0x1
	global_store_b128 v0, v[30:33], s[0:1] offset:16
	global_store_b128 v0, v[26:29], s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	_Z1kPKDv16_iS1_PKiS3_PDv8_f, .Lfunc_end0-_Z1kPKDv16_iS1_PKiS3_PDv8_f
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z1kPKDv16_iS1_PKiS3_PDv8_f
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 35
		.amdhsa_next_free_sgpr 66
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z1kPKDv16_iS1_PKiS3_PDv8_f)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z1kPKDv16_iS1_PKiS3_PDv8_f,"ax",@progbits
                                        ; -- End function
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.num_vgpr, 35
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.num_agpr, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.numbered_sgpr, 66
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.num_named_barrier, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.private_seg_size, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.uses_vcc, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.uses_flat_scratch, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.has_dyn_sized_stack, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.has_recursion, 0
	.set .L_Z1kPKDv16_iS1_PKiS3_PDv8_f.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 220
; TotalNumSgprs: 66
; NumVgprs: 35
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 35
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_130363cb5bd5ab3b,@object ; @__hip_cuid_130363cb5bd5ab3b
	.section	.bss.__hip_cuid_130363cb5bd5ab3b,"aw",@nobits
	.globl	__hip_cuid_130363cb5bd5ab3b
__hip_cuid_130363cb5bd5ab3b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_130363cb5bd5ab3b, 1

	.ident	"AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git c7ceccf9a5789b92fda4056b8501450d0b7a75de)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_130363cb5bd5ab3b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z1kPKDv16_iS1_PKiS3_PDv8_f
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         _Z1kPKDv16_iS1_PKiS3_PDv8_f.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     35
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgpu12.50-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
