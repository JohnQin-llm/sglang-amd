	.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.protected	relaxed_sys_pair        ; -- Begin function relaxed_sys_pair
	.globl	relaxed_sys_pair
	.p2align	8
	.type	relaxed_sys_pair,@function
relaxed_sys_pair:                       ; @relaxed_sys_pair
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0 nv
	v_mov_b32_e32 v2, 0
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x10 nv
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v0, s6 :: v_dual_mov_b32 v1, s7
	s_clause 0x1
	global_store_b64 v2, v[0:1], s[4:5] scope:SCOPE_SYS
	global_load_b64 v[0:1], v2, s[4:5] offset:8 scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	global_store_b64 v2, v[0:1], s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	relaxed_sys_pair, .Lfunc_end0-relaxed_sys_pair
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel relaxed_sys_pair
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_next_free_vgpr 3
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-relaxed_sys_pair)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lrelaxed_sys_pair.num_vgpr, 3
	.set .Lrelaxed_sys_pair.num_agpr, 0
	.set .Lrelaxed_sys_pair.numbered_sgpr, 66
	.set .Lrelaxed_sys_pair.num_named_barrier, 0
	.set .Lrelaxed_sys_pair.private_seg_size, 0
	.set .Lrelaxed_sys_pair.uses_vcc, 0
	.set .Lrelaxed_sys_pair.uses_flat_scratch, 0
	.set .Lrelaxed_sys_pair.has_dyn_sized_stack, 0
	.set .Lrelaxed_sys_pair.has_recursion, 0
	.set .Lrelaxed_sys_pair.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 112
; TotalNumSgprs: 66
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 3
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
	.text
	.protected	relaxed_dev_pair        ; -- Begin function relaxed_dev_pair
	.globl	relaxed_dev_pair
	.p2align	8
	.type	relaxed_dev_pair,@function
relaxed_dev_pair:                       ; @relaxed_dev_pair
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x0 nv
	v_mov_b32_e32 v2, 0
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x10 nv
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v0, s6 :: v_dual_mov_b32 v1, s7
	s_clause 0x1
	global_store_b64 v2, v[0:1], s[4:5] scope:SCOPE_DEV
	global_load_b64 v[0:1], v2, s[4:5] offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	global_store_b64 v2, v[0:1], s[0:1]
	s_endpgm
.Lfunc_end1:
	.size	relaxed_dev_pair, .Lfunc_end1-relaxed_dev_pair
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel relaxed_dev_pair
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_next_free_vgpr 3
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-relaxed_dev_pair)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lrelaxed_dev_pair.num_vgpr, 3
	.set .Lrelaxed_dev_pair.num_agpr, 0
	.set .Lrelaxed_dev_pair.numbered_sgpr, 66
	.set .Lrelaxed_dev_pair.num_named_barrier, 0
	.set .Lrelaxed_dev_pair.private_seg_size, 0
	.set .Lrelaxed_dev_pair.uses_vcc, 0
	.set .Lrelaxed_dev_pair.uses_flat_scratch, 0
	.set .Lrelaxed_dev_pair.has_dyn_sized_stack, 0
	.set .Lrelaxed_dev_pair.has_recursion, 0
	.set .Lrelaxed_dev_pair.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 112
; TotalNumSgprs: 66
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 3
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
	.text
	.protected	relacq_sys              ; -- Begin function relacq_sys
	.globl	relacq_sys
	.p2align	8
	.type	relacq_sys,@function
relacq_sys:                             ; @relacq_sys
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 1
	global_wb scope:SCOPE_SYS
	s_wait_storecnt 0x0
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_store_b32 v0, v1, s[0:1] scope:SCOPE_SYS
	global_load_b32 v1, v0, s[0:1] offset:4 scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
.Lfunc_end2:
	.size	relacq_sys, .Lfunc_end2-relacq_sys
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel relacq_sys
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 2
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-relacq_sys)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lrelacq_sys.num_vgpr, 2
	.set .Lrelacq_sys.num_agpr, 0
	.set .Lrelacq_sys.numbered_sgpr, 66
	.set .Lrelacq_sys.num_named_barrier, 0
	.set .Lrelacq_sys.private_seg_size, 0
	.set .Lrelacq_sys.uses_vcc, 0
	.set .Lrelacq_sys.uses_flat_scratch, 0
	.set .Lrelacq_sys.has_dyn_sized_stack, 0
	.set .Lrelacq_sys.has_recursion, 0
	.set .Lrelacq_sys.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 128
; TotalNumSgprs: 66
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 2
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	fence_rel_agent         ; -- Begin function fence_rel_agent
	.globl	fence_rel_agent
	.p2align	8
	.type	fence_rel_agent,@function
fence_rel_agent:                        ; @fence_rel_agent
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	v_dual_mov_b32 v1, 7 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 1
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 0x0
	global_wb scope:SCOPE_DEV
	s_wait_storecnt 0x0
	global_store_b32 v2, v3, s[2:3]
	s_endpgm
.Lfunc_end3:
	.size	fence_rel_agent, .Lfunc_end3-fence_rel_agent
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fence_rel_agent
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 4
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-fence_rel_agent)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lfence_rel_agent.num_vgpr, 4
	.set .Lfence_rel_agent.num_agpr, 0
	.set .Lfence_rel_agent.numbered_sgpr, 66
	.set .Lfence_rel_agent.num_named_barrier, 0
	.set .Lfence_rel_agent.private_seg_size, 0
	.set .Lfence_rel_agent.uses_vcc, 0
	.set .Lfence_rel_agent.uses_flat_scratch, 0
	.set .Lfence_rel_agent.has_dyn_sized_stack, 0
	.set .Lfence_rel_agent.has_recursion, 0
	.set .Lfence_rel_agent.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 100
; TotalNumSgprs: 66
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 4
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	fence_rel_sys           ; -- Begin function fence_rel_sys
	.globl	fence_rel_sys
	.p2align	8
	.type	fence_rel_sys,@function
fence_rel_sys:                          ; @fence_rel_sys
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	v_dual_mov_b32 v1, 7 :: v_dual_mov_b32 v2, 0
	v_mov_b32_e32 v3, 1
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_wait_storecnt 0x0
	global_wb scope:SCOPE_SYS
	s_wait_storecnt 0x0
	global_store_b32 v2, v3, s[2:3]
	s_endpgm
.Lfunc_end4:
	.size	fence_rel_sys, .Lfunc_end4-fence_rel_sys
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fence_rel_sys
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 4
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-fence_rel_sys)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lfence_rel_sys.num_vgpr, 4
	.set .Lfence_rel_sys.num_agpr, 0
	.set .Lfence_rel_sys.numbered_sgpr, 66
	.set .Lfence_rel_sys.num_named_barrier, 0
	.set .Lfence_rel_sys.private_seg_size, 0
	.set .Lfence_rel_sys.uses_vcc, 0
	.set .Lfence_rel_sys.uses_flat_scratch, 0
	.set .Lfence_rel_sys.has_dyn_sized_stack, 0
	.set .Lfence_rel_sys.has_recursion, 0
	.set .Lfence_rel_sys.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 100
; TotalNumSgprs: 66
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 4
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	fence_acq_agent         ; -- Begin function fence_acq_agent
	.globl	fence_acq_agent
	.p2align	8
	.type	fence_acq_agent,@function
fence_acq_agent:                        ; @fence_acq_agent
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[0:3], s[0:1], 0x0 nv
	s_wait_kmcnt 0x0
	s_load_b32 s4, s[0:1], 0x0
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	s_load_b32 s0, s[0:1], 0x4
	s_wait_kmcnt 0x0
	s_add_co_i32 s0, s0, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
.Lfunc_end5:
	.size	fence_acq_agent, .Lfunc_end5-fence_acq_agent
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fence_acq_agent
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 2
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-fence_acq_agent)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lfence_acq_agent.num_vgpr, 2
	.set .Lfence_acq_agent.num_agpr, 0
	.set .Lfence_acq_agent.numbered_sgpr, 66
	.set .Lfence_acq_agent.num_named_barrier, 0
	.set .Lfence_acq_agent.private_seg_size, 0
	.set .Lfence_acq_agent.uses_vcc, 0
	.set .Lfence_acq_agent.uses_flat_scratch, 0
	.set .Lfence_acq_agent.has_dyn_sized_stack, 0
	.set .Lfence_acq_agent.has_recursion, 0
	.set .Lfence_acq_agent.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 112
; TotalNumSgprs: 66
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 2
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	clocks                  ; -- Begin function clocks
	.globl	clocks
	.p2align	8
	.type	clocks,@function
clocks:                                 ; @clocks
	.cfi_startproc
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b64 s[0:1], s[0:1], 0x0 nv
	s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)
	v_mov_b32_e32 v2, 0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[2:3]
	global_store_b64 v2, v[0:1], s[0:1]
	s_sleep 1
	s_get_shader_cycles_u64 s[2:3]
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[0:1], s[2:3]
	global_store_b64 v2, v[0:1], s[0:1] offset:8
	s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[2:3]
	global_store_b64 v2, v[0:1], s[0:1] offset:16
	s_endpgm
.Lfunc_end6:
	.size	clocks, .Lfunc_end6-clocks
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel clocks
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
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
		.amdhsa_next_free_vgpr 3
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-clocks)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lclocks.num_vgpr, 3
	.set .Lclocks.num_agpr, 0
	.set .Lclocks.numbered_sgpr, 66
	.set .Lclocks.num_named_barrier, 0
	.set .Lclocks.private_seg_size, 0
	.set .Lclocks.uses_vcc, 0
	.set .Lclocks.uses_flat_scratch, 0
	.set .Lclocks.has_dyn_sized_stack, 0
	.set .Lclocks.has_recursion, 0
	.set .Lclocks.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 120
; TotalNumSgprs: 66
; NumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 3
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_a098e4ecdd8bd4de,@object ; @__hip_cuid_a098e4ecdd8bd4de
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a098e4ecdd8bd4de
__hip_cuid_a098e4ecdd8bd4de:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a098e4ecdd8bd4de, 1

	.ident	"AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git c7ceccf9a5789b92fda4056b8501450d0b7a75de)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a098e4ecdd8bd4de
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           relaxed_sys_pair
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         relaxed_sys_pair.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           relaxed_dev_pair
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         relaxed_dev_pair.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           relacq_sys
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         relacq_sys.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           fence_rel_agent
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         fence_rel_agent.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           fence_rel_sys
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         fence_rel_sys.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           fence_acq_agent
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         fence_acq_agent.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           clocks
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         clocks.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgpu12.50-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
