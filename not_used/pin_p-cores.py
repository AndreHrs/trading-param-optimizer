"""
Standalone diagnostic script — run directly to verify P-core detection and affinity pinning.
Logic lives in utilities/pin_p_cores.py.
"""
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utilities.pin_p_cores import _detect_vendor, _get_p_core_ids, get_worker_count, worker_init

if __name__ == "__main__":
    print(f"Platform : {platform.system()}")
    print(f"Vendor   : {_detect_vendor()}")
    print(f"Workers  : {get_worker_count()}")

    if platform.system() == "Linux" and _detect_vendor() == "intel":
        print(f"P-cores  : {_get_p_core_ids()}")

    worker_init()

    if platform.system() == "Linux":
        print(f"Affinity after worker_init: {sorted(os.sched_getaffinity(0))}")
    else:
        print("Affinity pinning is Linux-only. No-op on this platform.")
