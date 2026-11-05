"""exec a command under the MPOL_LOCAL task memory policy (inherited by every child process and thread).

A non-default policy without MPOL_F_NUMA_BALANCING has no MPOL_F_MOF, so NUMA balancing's task_numa_work skips the
process's VMAs: no change_prot_numa scans, hence no SVM MMU-notifier invalidations and no KFD queue quiesce.
  python3 mpol_exec.py python3 -m sglang.launch_server ...
"""

import ctypes
import os
import sys

libc = ctypes.CDLL(None, use_errno=True)
if libc.syscall(238, 4, None, 0) != 0:  # __NR_set_mempolicy (x86_64), MPOL_LOCAL
    raise OSError(ctypes.get_errno(), "set_mempolicy(MPOL_LOCAL)")
os.execvp(sys.argv[1], sys.argv[1:])
