# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout every MonoKernel's error words follow (gfx1250 / TP4 port of
``k2bundle.runtime.abi``).

A rank's health words: a header whose word FAILED is set once any of its
in-kernel waits gave up (sticky: a failed rank stays failed, and its later waits
exit at once), then one record a CTA of the first wait it gave up on (flag,
region id, pair index, tag, seen tag, CTA, clock, pad).

The output rank's flags: one int32 word a TP rank in the first PEER_HEAD bytes
of the output rank's peer buffer. A rank whose wait gives up writes its word
there, drains the store, then sets FAILED, so nothing it stores later can
overtake the flag. The runner reads these words after every step.

TP4: the flag words are 0 .. 15 (TP * 4 bytes); the K2 completion pairs keep
their MI355X offset (64) and shrink to TP pairs, so the header layout stays
compatible with an 8-rank reader.
"""

MONO_ABI_VERSION = 1

TP = 4  # MI355X: 8

RECORD_WORDS = 8
FAILED = 0
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
# K2 uses one tagged pair a rank in the otherwise-reserved peer header to prove
# that every rank reached the launch tail before any rank can return.  Once all
# completion tags are visible, each rank imports the shared alarm words into
# its local sticky health state.  Keep this disjoint from the flag words and
# within PEER_HEAD.
K2_DONE_OFFSET = 64
K2_DONE_RANKS = TP  # MI355X: 8
K2_DONE_BYTES = K2_DONE_RANKS * 8
assert 4 * TP <= K2_DONE_OFFSET
assert K2_DONE_OFFSET + K2_DONE_BYTES <= PEER_HEAD
WAIT_SECONDS = float(__import__("os").environ.get("K2_WAIT_SECONDS", "10"))  # an in-kernel wait gives up after this long


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of ``ctas``: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS
