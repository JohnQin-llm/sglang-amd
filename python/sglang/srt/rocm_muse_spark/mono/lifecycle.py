# SPDX-License-Identifier: Apache-2.0
"""Host-only resource cleanup; safe when optional Mono device imports are absent."""


def close_model_runners(model):
    """Close each decoder-owned runner once, including wrapped multimodal models.

    Call collectively on the graceful shutdown path, while the TP process group
    remains valid. Failure paths must not synchronize a potentially wedged GPU.
    """
    closed = set()
    for module in model.modules():
        runner = getattr(module, "_ep8_full_k2", None)
        if runner is not None and id(runner) not in closed:
            runner.close()
            closed.add(id(runner))
