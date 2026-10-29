"""Build directories of the Muse Spark ROCm JIT extensions (``torch.utils.cpp_extension.load``).

A build, and torch's hipify output, never land next to the source: they go to ``$<env var>/<digest>`` when the
extension's environment variable is set, else to ``$SGLANG_CACHE_DIR/muse_jit/<name>/<digest>`` (a local cache,
``~/.cache/sglang`` by default). The source is copied into the build directory first, because hipify writes its
``.hip`` file beside the file it is given.
"""

import os
from pathlib import Path

from sglang.srt.environ import envs


def stage_source(source: Path, data: bytes, digest: str, env_var: str, name: str):
    """(build directory, staged copy of ``source``) for an extension whose source bytes are ``data``."""
    root = os.environ.get(env_var)
    base = (
        Path(root)
        if root
        else Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "muse_jit" / name
    )
    build = base / digest[:16]
    build.mkdir(parents=True, exist_ok=True)
    staged = build / source.name
    if not staged.exists() or staged.read_bytes() != data:
        # Every TP rank stages the same bytes; the rename keeps a reader from seeing a partial file.
        tmp = build / f".{source.name}.{os.getpid()}"
        tmp.write_bytes(data)
        os.replace(tmp, staged)
    return build, staged
