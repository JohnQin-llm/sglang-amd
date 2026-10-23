# SPDX-License-Identifier: Apache-2.0
"""Width dispatch for the unchanged narrow K2 and isolated native wide K2."""

from __future__ import annotations

NARROW_WIDTHS = tuple(range(1, 9))
WIDE_WIDTHS = (16, 24)
SUPPORTED_WIDTHS = NARROW_WIDTHS + WIDE_WIDTHS


def family_for_width(tokens: int) -> str:
    if tokens in NARROW_WIDTHS:
        return "narrow"
    if tokens in WIDE_WIDTHS:
        return "wide"
    raise ValueError(f"unsupported FullK2 width {tokens}; expected {SUPPORTED_WIDTHS}")


def module_for_width(tokens: int):
    if family_for_width(tokens) == "narrow":
        from sglang.srt.rocm_muse_spark.mono.ops import k2

        return k2
    from sglang.srt.rocm_muse_spark.mono.ops import k2_wide

    return k2_wide


def module_for_key(key):
    from sglang.srt.rocm_muse_spark.mono.ops import k2

    if isinstance(key, k2.K2Build):
        return k2
    from sglang.srt.rocm_muse_spark.mono.ops import k2_wide

    if isinstance(key, k2_wide.K2Build):
        return k2_wide
    raise TypeError(f"not a K2 build key: {type(key).__name__}")


def default_key(tokens: int, *, experimental_two_cta: bool = False):
    mod = module_for_width(tokens)
    if family_for_width(tokens) == "narrow":
        return mod.K2Build(
            tokens=tokens,
            experimental_two_cta=experimental_two_cta and tokens == 8,
        )
    if experimental_two_cta:
        raise ValueError("experimental_two_cta is an M8-only build option")
    return mod.K2Build(tokens=tokens)


def make_key(*, tokens: int, experimental_two_cta: bool = False, **kwargs):
    mod = module_for_width(tokens)
    if family_for_width(tokens) == "narrow":
        return mod.make_key(
            tokens=tokens,
            experimental_two_cta=experimental_two_cta,
            **kwargs,
        )
    if experimental_two_cta:
        raise ValueError("experimental_two_cta is an M8-only build option")
    return mod.make_key(tokens=tokens, **kwargs)
