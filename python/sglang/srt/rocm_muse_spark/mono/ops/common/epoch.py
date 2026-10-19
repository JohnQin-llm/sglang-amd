# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe epoch lifecycle shared by the persistent layer kernels."""

import torch

EPOCH_TAG_BITS = 22
EPOCH_TAG_PERIOD = 1 << EPOCH_TAG_BITS
LAST_VALID_EPOCH = EPOCH_TAG_PERIOD - 1
TAG_SLOTS = 256


def mailbox_tag(epoch: int, slot: int) -> int:
    """Return the integer form of the tag generated inside K1 and K2."""
    return (epoch % EPOCH_TAG_PERIOD) * TAG_SLOTS + slot + 1


def advance_epoch_(epoch: torch.Tensor) -> None:
    """Advance once, failing before the 22-bit device tag can wrap.

    ``torch._assert_async`` is captured into HIP graphs.  It therefore guards
    every replay as well as eager execution without a host read or sync.  The
    final representable nonzero epoch is valid; the next step is terminal and
    leaves ``epoch`` unchanged, so kernels cannot observe a reused tag.
    """
    torch._assert_async(
        epoch < LAST_VALID_EPOCH,
        "MonoKernel epoch exhausted before 22-bit mailbox tag reuse",
    )
    epoch.add_(1)
