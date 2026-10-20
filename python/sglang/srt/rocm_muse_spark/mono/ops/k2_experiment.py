# SPDX-License-Identifier: Apache-2.0
"""Pure geometry helpers for K2's opt-in rows=8 two-CTA experiment."""

DEFAULT_BLOCKS = 256
EXPERIMENT_BLOCKS = 512
FINISH_FANIN = 8
HEALTH_RECORD_WORDS = 8


def grid_blocks(key) -> int:
    """Return the persistent grid width selected by ``key``."""
    if key.experimental_two_cta:
        if key.tokens != 8:
            raise ValueError("experimental_two_cta is restricted to tokens == 8")
        return EXPERIMENT_BLOCKS
    return DEFAULT_BLOCKS


def health_words_for_key(key) -> int:
    """Required health-buffer words, including its header record."""
    return (grid_blocks(key) + 1) * HEALTH_RECORD_WORDS


def finish_counts(blocks: int) -> tuple[int, int]:
    """Return the two fan-in level counts for a persistent grid."""
    groups = (blocks + FINISH_FANIN - 1) // FINISH_FANIN
    roots = (groups + FINISH_FANIN - 1) // FINISH_FANIN
    return groups, roots
