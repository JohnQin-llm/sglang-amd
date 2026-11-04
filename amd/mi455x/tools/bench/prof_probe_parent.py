"""The parent initializes HIP first (as `import sglang` does), then a spawned child profiles."""
import ctypes, sys
import torch
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile


def native(name):
    v = ctypes.CDLL(None).getenv
    v.restype = ctypes.c_char_p
    return v(name.encode())


def child(rank):
    import os
    if "--clear" in sys.argv:
        os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)
    x = torch.randn(256, 256, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        x @ x
        torch.cuda.synchronize()
    print(f"child: inherited={native('ROCPROFILER_REGISTER_LIBRARY') is not None} "
          f"device events {sum(1 for e in p.events() if e.device_type.name != 'CPU')}", flush=True)


if __name__ == "__main__":
    if "--parent-hip" in sys.argv:
        torch.cuda.is_available()
    if "--parent-clear" in sys.argv:
        import os
        os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)
        os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")
    print(f"parent: native var set={native('ROCPROFILER_REGISTER_LIBRARY') is not None}", flush=True)
    mp.spawn(child, nprocs=1)
