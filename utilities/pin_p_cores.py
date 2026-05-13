"""
CPU affinity utility for multiprocessing experiment workers.

NOTE: P-core pinning via os.sched_setaffinity is Linux-only.
On Windows/macOS, worker_init() is a no-op and all cores are used.

Behaviour by vendor:
  Intel (Linux): pins each worker process to P-cores only (identified by highest
                 max frequency) so E-cores never skew runtime measurements.
  AMD   (Linux): no pinning. All cores used at full power (AMD has no P/E split).
  Other / non-Linux: no pinning, os.cpu_count() workers.
"""
import os
import platform


def _detect_vendor():
    """Read /proc/cpuinfo to return 'intel', 'amd', or 'unknown'."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    vendor = line.split(":", 1)[1].strip().lower()
                    if "genuineintel" in vendor:
                        return "intel"
                    if "authenticamd" in vendor:
                        return "amd"
    except FileNotFoundError:
        pass
    return "unknown"


def _get_p_core_ids():
    """
    Return logical CPU IDs that are P-cores on Intel hybrid CPUs.
    P-cores are identified by having the highest cpuinfo_max_freq (within 10% tolerance
    to account for minor BIOS frequency differences between HT siblings).
    """
    cores = []
    for cpu_id in range(os.cpu_count() or 0):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq") as f:
                cores.append((cpu_id, int(f.read())))
        except FileNotFoundError:
            pass

    if not cores:
        return list(range(os.cpu_count() or 1))

    max_freq = max(freq for _, freq in cores)
    threshold = max_freq * 0.9
    return [cpu for cpu, freq in cores if freq >= threshold]


def get_worker_count():
    """
    Return the number of parallel workers to spawn.
    Intel/Linux: number of logical P-cores.
    All others:  os.cpu_count() (full machine).
    """
    if platform.system() != "Linux":
        return os.cpu_count() or 1
    if _detect_vendor() == "intel":
        return len(_get_p_core_ids())
    return os.cpu_count() or 1


def worker_init():
    """
    Called once per worker process at startup (pass as initializer= to ProcessPoolExecutor).

    Intel/Linux: pins the process to P-cores via sched_setaffinity so the OS scheduler
                 never migrates it to an E-core, keeping runtime measurements fair.
    AMD/Linux:   no affinity change — all cores available at full power.
    Non-Linux:   no-op (sched_setaffinity does not exist on Windows/macOS).

    Also forces single-threaded BLAS/OpenMP so workers don't fight each other over cores.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if platform.system() != "Linux":
        return

    if _detect_vendor() == "intel":
        p_cores = _get_p_core_ids()
        os.sched_setaffinity(0, p_cores)
