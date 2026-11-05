"""gfx1250 verify-attention overrides of AITER's Triton ``unified_attention`` (Muse Spark C1 verify), from SGLang.

AITER's wrapper looks up its config / 3D split / reduce helpers as module globals on every call, so they are
wrapped here instead of patching AITER; each override falls through to the served helper when it does not apply.

* ``SGLANG_MUSE_ATTN_V2=1``: full-attention layers' Triton 3D split runs ``attn_v2_split``
  (``sglang.kernels.ops.attention.muse_attn_v2``): same fp32 partials and device-side segment split as
  ``kernel_unified_attention_3d``, combined by the served ``reduce_segments`` on its single-tile path. Only for an
  unshuffled FP8 cache, BF16 Q, head_dim 64, GQA 8, no sinks / alibi / qq_bias / softcap / output scale, one
  sequence of <= 8 query rows, tile 128 (the C1 verify).
* ``SGLANG_MUSE_ATTN3=1``: full-attention layers' 3D split runs ``attn3_split``
  (``sglang.kernels.ops.attention.muse_attn3``, Gluon: TDM page loads into an LDS ring, WMMA, fp32 softmax) on
  64-token tiles (same segment count), combined by the served ``reduce_segments`` on its single-tile path. Same
  conditions as attn_v2 plus a contiguous ``[pages, 16, kv heads, 64]`` cache; it takes precedence over attn_v2.
* ``SGLANG_MUSE_SWA_CFG=1``: sliding-window layers' Triton 3D kv split at 4 segments per CU over 32-token tiles
  (64 segments x 32 at C1 verify) instead of 1 per CU over 64-token tiles.
"""

import logging
import os

import torch
import triton

logger = logging.getLogger(__name__)

ATTN_V2 = os.environ.get("SGLANG_MUSE_ATTN_V2", "0") == "1"
ATTN3 = os.environ.get("SGLANG_MUSE_ATTN3", "0") == "1"
SWA_CFG = os.environ.get("SGLANG_MUSE_SWA_CFG", "0") == "1"

# kv_split["D_LEQ_64.SW.DT_any_fp8"] of vagm's DEFAULT.proposed.json (served: SEGMENTS_PER_CU 1, tiles 64).
SWA_KV_SPLIT = {
    "MIN_SEGMENTS": 1,
    "SEGMENTS_PER_CU": 4,
    "SEGMENT_TILE_MIN": 32,
    "TILE_SIZE_MIN": 32,
    "TILE_SIZE_MAX": 32,
}
V2_TILE = 128
V2_ROWS = 8  # query rows per block: BLOCK_M 64 = 8 rows x 8 GQA heads
ATTN3_TILE = 64
ATTN3_NBUF = 4  # 64-token units in the LDS ring

_installed = False
_live = set()


def _note_live(name: str) -> None:
    if name not in _live:
        _live.add(name)
        logger.info("Muse verify attention: %s live", name)


def _swa_split_applies(op, params, backend) -> bool:
    return (
        SWA_CFG
        and op == "kv_split"
        and backend == "triton"
        and params.sliding_window > 0
        and max(params.head_size, params.head_size_v) <= 64
        and not params.shuffled_kv_cache
        and params.kv_cache_dtype == torch.float8_e4m3fn
    )


def _v2_applies(params, num_segments, tile_size, window_split) -> bool:
    q, k, v = params.q, params.k, params.v
    return (
        ATTN_V2
        and num_segments > 1
        and tile_size == V2_TILE
        and not window_split
        and not params.skip_reduce
        and params.sliding_window == 0
        and not params.shuffled_kv_cache
        and params.kv_cache_dtype == torch.float8_e4m3fn
        and params.q_dtype == torch.bfloat16
        and params.head_size == 64
        and params.head_size_v == 64
        and params.num_queries_per_kv == 8
        and params.sinks is None
        and not params.use_alibi_slopes
        and not params.use_qq_bias
        and not params.softcap
        and params.output_scale is None
        and params.num_seqs == 1
        and params.num_tokens <= V2_ROWS
        and int(params.max_seqlen_q) <= V2_ROWS
        and isinstance(params.k_descale, torch.Tensor)
        and isinstance(params.v_descale, torch.Tensor)
        and q.stride(2) == 1
        and k.stride(3) == 1
        and v.stride(3) == 1
    )


def _attn3_shape_applies(params) -> bool:
    q, k, v = params.q, params.k, params.v
    page = params.block_size
    contiguous = (page * params.num_kv_heads * 64, params.num_kv_heads * 64, 64, 1)
    return (
        ATTN3
        and params.sliding_window == 0
        and not params.skip_reduce
        and not params.shuffled_kv_cache
        and params.kv_cache_dtype == torch.float8_e4m3fn
        and params.q_dtype == torch.bfloat16
        and params.head_size == 64
        and params.head_size_v == 64
        and params.num_queries_per_kv == 8
        and page == 16
        and params.sinks is None
        and not params.use_alibi_slopes
        and not params.use_qq_bias
        and not params.softcap
        and params.output_scale is None
        and params.num_seqs == 1
        and params.num_tokens <= V2_ROWS
        and int(params.max_seqlen_q) <= V2_ROWS
        and isinstance(params.k_descale, torch.Tensor)
        and isinstance(params.v_descale, torch.Tensor)
        and q.stride(2) == 1
        and tuple(k.stride()) == contiguous
        and tuple(v.stride()) == contiguous
    )


def _attn3_applies(params, num_segments, tile_size, window_split) -> bool:
    return num_segments > 1 and tile_size == ATTN3_TILE and not window_split and _attn3_shape_applies(params)


def install() -> None:
    """Wrap the AITER wrapper's helpers once (no-op unless a flag is on)."""
    global _installed
    if _installed or not (ATTN_V2 or SWA_CFG or ATTN3):
        return
    import aiter.ops.triton.attention.unified_attention as uaw
    from aiter.ops.triton.utils.unified_attention_utils import (
        compute_segment_params,
        compute_tile_params,
    )

    from sglang.kernels.ops.attention.muse_attn_v2 import attn_v2_split

    attn3_split = None
    if ATTN3:
        from sglang.kernels.ops.attention.muse_attn3 import attn3_split

    get_config = uaw.get_unified_attention_config
    split_3d = uaw._unified_attention_3d_triton
    reduce_triton = uaw._reduce_segments_triton

    def get_unified_attention_config(op, params, backend="triton", arch=None):
        if arch is None and _swa_split_applies(op, params, backend):
            _note_live("sliding-window kv split (4 segments per CU, 32-token tiles)")
            config = compute_tile_params(dict(SWA_KV_SPLIT), params.block_size)
            return compute_segment_params(config, params)
        config = get_config(op, params, backend=backend, arch=arch)
        if arch is None and op == "kv_split" and backend == "triton" and _attn3_shape_applies(params):
            # attn3 walks 64-token units: same segments, 64-token tiles for the split and the reduce
            config = dict(config)
            config["TILE_SIZE"] = ATTN3_TILE
        return config

    def _unified_attention_3d_triton(
        params, segm_output, segm_max, segm_expsum, NUM_SEGMENTS, TILE_SIZE, window_split=False,
        segment_grid_first=False,
    ):
        if attn3_split is not None and _attn3_applies(params, NUM_SEGMENTS, TILE_SIZE, window_split):
            _note_live(f"attn3_split for full-attention layers (tile {TILE_SIZE}, {NUM_SEGMENTS} segments)")
            q, k = params.q, params.k
            attn3_split[(NUM_SEGMENTS, params.num_kv_heads, 1)](
                segm_output, segm_max, segm_expsum,
                q, k, params.v, params.block_table, params.seqused_k, params.cu_seqlens_q,
                params.k_descale, params.v_descale, params.softmax_scale,
                q.stride(0), q.stride(1), k.shape[0],
                NQ=params.num_query_heads, NKV=params.num_kv_heads, G=params.num_queries_per_kv, D=64,
                PAGE=params.block_size, TILE=TILE_SIZE, NSEG=NUM_SEGMENTS, NBUF=ATTN3_NBUF, BLOCK_M=64, SW=0,
                num_warps=4,
            )
            return
        if not _v2_applies(params, NUM_SEGMENTS, TILE_SIZE, window_split):
            return split_3d(
                params, segm_output, segm_max, segm_expsum, NUM_SEGMENTS, TILE_SIZE,
                window_split=window_split, segment_grid_first=segment_grid_first,
            )
        _note_live(f"attn_v2_split for full-attention layers (tile {TILE_SIZE}, {NUM_SEGMENTS} segments)")
        q, k, v = params.q, params.k, params.v
        attn_v2_split[(NUM_SEGMENTS, params.num_kv_heads, triton.cdiv(params.num_tokens, V2_ROWS))](
            segm_output, segm_max, segm_expsum,
            q, k, v, params.block_table, params.seqused_k, params.cu_seqlens_q,
            params.k_descale, params.v_descale, params.softmax_scale,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
            NQ=params.num_query_heads, G=params.num_queries_per_kv, D=64, PAGE=params.block_size,
            TILE=TILE_SIZE, NSEG=NUM_SEGMENTS, BLOCK_M=64, QSPLIT=0, FOLD=True, DUAL=False, PFD=0,
            PF_SCOPE=0, num_warps=4, num_stages=2, waves_per_eu=0,
        )

    def _reduce_segments_triton(
        params, segm_output, segm_max, segm_expsum, NUM_SEGMENTS, TILE_SIZE, window_split=False
    ):
        if not (
            _v2_applies(params, NUM_SEGMENTS, TILE_SIZE, window_split)
            or (attn3_split is not None and _attn3_applies(params, NUM_SEGMENTS, TILE_SIZE, window_split))
        ):
            return reduce_triton(
                params, segm_output, segm_max, segm_expsum, NUM_SEGMENTS, TILE_SIZE, window_split=window_split
            )
        # the served reduce kernel and config, on its single-tile path (all segments in one step)
        config = get_config("reduce", params, backend="triton")
        config["SEGMENT_BLOCK"] = max(config.get("SEGMENT_BLOCK") or 0, NUM_SEGMENTS)
        d = triton.next_power_of_2(params.head_size_v)
        uaw.reduce_segments[(params.num_tokens, params.num_query_heads, triton.cdiv(d, config.get("BLOCK_D") or d))](
            output_ptr=params.out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=params.seqused_k,
            num_seqs=params.num_seqs,
            num_query_heads=params.num_query_heads,
            out_scale_ptr=params.output_scale,
            output_stride_0=params.out.stride(0),
            output_stride_1=params.out.stride(1),
            block_table_stride=params.block_table.stride(0),
            HEAD_SIZE=params.head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size),
            V_HEAD_SIZE=params.head_size_v,
            V_HEAD_SIZE_PADDED=d,
            query_start_len_ptr=params.cu_seqlens_q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            TILE_SIZE=TILE_SIZE,
            BLOCK_Q=None,
            SLIDING_WINDOW=params.sliding_window,
            WINDOW_SPLIT=window_split,
            **config,
            enable_fp_fusion=True,
        )

    uaw.get_unified_attention_config = get_unified_attention_config
    uaw._unified_attention_3d_triton = _unified_attention_3d_triton
    uaw._reduce_segments_triton = _reduce_segments_triton
    _installed = True
    logger.info(
        "Muse verify attention overrides installed: attn_v2 %s, attn3 %s, sliding-window config %s",
        ATTN_V2, ATTN3, SWA_CFG
    )
