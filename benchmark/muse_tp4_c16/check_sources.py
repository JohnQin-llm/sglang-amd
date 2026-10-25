"""Check the packaged source bytes without importing torch or using a GPU."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-root", type=Path)
    parser.add_argument("--apply-aiter-patch", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    binding = json.loads((here / "source-binding.json").read_text())
    errors = []
    for name, entry in binding["sglang_modules"].items():
        path = root / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            errors.append(name)
    if errors:
        raise SystemExit("SGLang source mismatch: " + ", ".join(errors))
    print("30 SGLang modules match the measured overlay byte for byte.")
    if args.apply_aiter_patch and args.aiter_root is None:
        parser.error("--apply-aiter-patch requires --aiter-root")
    if args.aiter_root is None:
        return
    for name, expected in binding["aiter_runtime_pins"].items():
        path = args.aiter_root / name
        if not path.is_file() or digest(path) != expected:
            errors.append(name)
    wrapper = args.aiter_root / "aiter/ops/triton/attention/unified_attention.py"
    device = args.aiter_root / "aiter/ops/triton/_triton_kernels/attention/unified_attention.py"
    pins = binding["aiter_wrapper"]
    if not device.is_file() or digest(device) != pins["device_sha256"]:
        errors.append(str(device.relative_to(args.aiter_root)))
    if errors:
        raise SystemExit("AITER dependency mismatch: " + ", ".join(errors))
    current = digest(wrapper)
    expected = pins["candidate_wrapper_sha256"]
    if current == expected:
        print("AITER dependencies and SWA stage-2 wrapper match.")
        return
    if current != pins["baseline_wrapper_sha256"]:
        raise SystemExit("Unsupported AITER attention wrapper; refusing to patch.")
    patch = str(here / "aiter-swa-stage2.patch")
    command = ["git", "-C", str(args.aiter_root), "apply"]
    subprocess.run(command + ["--check", patch], check=True)
    if not args.apply_aiter_patch:
        raise SystemExit("Pinned AITER baseline found; rerun with --apply-aiter-patch to reproduce the measured wrapper.")
    subprocess.run(command + [patch], check=True)
    if digest(wrapper) != expected:
        raise SystemExit("Patched AITER wrapper does not match the measured bytes.")
    print("Applied and verified the measured SWA stage-2 wrapper patch.")


if __name__ == "__main__":
    main()
