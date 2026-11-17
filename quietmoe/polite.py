"""The "polite" part: run llama.cpp inside limits so the PC stays usable.

How it works:
  * The model is memory-mapped (llama.cpp default). Expert weights that stay on the
    CPU side live in the Linux page cache instead of being copied into RAM.
  * We start llama.cpp in its own systemd scope with MemoryHigh=<cap>. Page cache
    counts toward that cap, so when the cap is reached Linux drops the coldest
    experts and re-reads them from the NVMe drive later. That is expert streaming
    for free, and the model can never eat all your RAM.
  * A small governor thread watches free memory. When you open something heavy it
    lowers the cap (QuietMoE gets slower, your apps stay smooth). When the PC is idle
    it raises the cap again (more experts cached, faster).
  * CPU and disk priority are lowered with nice / ionice / cgroup weights.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

from .profiles import Profile


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def mem_available_gb() -> float:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def meminfo() -> dict[str, float]:
    """/proc/meminfo in GB."""
    out: dict[str, float] = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, val = line.partition(":")
        parts = val.split()
        if parts:
            out[key] = int(parts[0]) / 1024**2
    return out


def swap_out_pages() -> int:
    """Total pages Linux has moved to swap since boot (/proc/vmstat pswpout)."""
    for line in (_read("/proc/vmstat") or "").splitlines():
        if line.startswith("pswpout "):
            return int(line.split()[1])
    return 0


# Memory always kept free for the system, the desktop and your apps' file cache.
RESERVE_GB = 3.0


def ceiling_gb(info: dict[str, float], our_anon_gb: float, reserve: float = RESERVE_GB) -> float:
    """Biggest RAM cap QuietMoE may have right now: total RAM minus the memory your other
    apps really use (anonymous + shared memory, which can't simply be dropped) minus a reserve.
    MemAvailable alone is not enough: it goes UP when Linux swaps your apps out."""
    others = info.get("AnonPages", 0) + info.get("Shmem", 0) - our_anon_gb
    return info.get("MemTotal", 0) - max(0.0, others) - reserve


def pressure(resource: str) -> dict[str, float]:
    """Read /proc/pressure/<resource>. Returns {'some': avg10, 'full': avg10}."""
    out = {"some": 0.0, "full": 0.0}
    for line in (_read(f"/proc/pressure/{resource}") or "").splitlines():
        kind, *fields = line.split()
        for f in fields:
            if f.startswith("avg10="):
                out[kind] = float(f.split("=")[1])
    return out


def cgroup_memory_ok() -> bool:
    """True if systemd lets this user put memory limits on their own processes."""
    if not shutil.which("systemd-run") or not shutil.which("systemctl"):
        return False
    uid = os.getuid()
    path = f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers"
    controllers = _read(path) or ""
    return "memory" in controllers.split()


def scope_cgroup(unit: str) -> str | None:
    """Return the /sys/fs/cgroup path of a running systemd user scope."""
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", f"{unit}.scope"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.partition("=")[2]
    return "/sys/fs/cgroup" + path if path else None


def scope_memory(unit: str) -> dict[str, float]:
    """RAM used by the scope, split into 'file' (model cache, can be dropped and re-read)
    and 'anon' (fixed memory that cannot be dropped)."""
    cg = scope_cgroup(unit)
    out: dict[str, float] = {}
    if not cg:
        return out
    for line in (_read(f"{cg}/memory.stat") or "").splitlines():
        key, _, val = line.partition(" ")
        if key in ("anon", "file"):
            out[key] = int(val) / 1024**3
    swap = _read(f"{cg}/memory.swap.current")
    if swap and swap.isdigit():
        out["swap"] = int(swap) / 1024**3
    return out


@dataclass
class Launch:
    cmd: list[str]
    unit: str | None          # systemd scope name, if limits are active
    limited: bool
    warnings: list[str]


def wrap(cmd: list[str], profile: Profile) -> Launch:
    """Wrap a command with memory limits and low priority according to the profile."""
    warnings: list[str] = []
    if not profile.polite:
        return Launch(cmd, None, False, warnings)

    prefix: list[str] = []
    unit = None
    limited = False
    if cgroup_memory_ok():
        unit = f"quietmoe-{uuid.uuid4().hex[:8]}"
        high = f"{int(profile.ram_gb * 1024)}M"
        # Hard ceiling well above the soft cap, only there to stop a runaway.
        hard = f"{int((profile.ram_max_gb + 2) * 1024)}M"
        prefix = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
                  "-p", f"MemoryHigh={high}", "-p", f"MemoryMax={hard}",
                  # QuietMoE's own memory never goes to swap: when the cap is reached, Linux
                  # can only drop model cache (re-read from the SSD later). Without this,
                  # Linux swaps QuietMoE itself, and that looks like "apps are being swapped".
                  "-p", "MemorySwapMax=0",
                  "-p", f"CPUWeight={profile.cpu_weight}",
                  "-p", f"IOWeight={max(1, profile.cpu_weight)}"]
        limited = True
    else:
        warnings.append("systemd user memory limits unavailable: running with low priority only "
                        "(RAM is not capped). See README > Troubleshooting.")

    if profile.nice and shutil.which("nice"):
        prefix += ["nice", "-n", str(profile.nice)]
    if shutil.which("ionice"):
        prefix += ["ionice", "-c", str(profile.io_class)]
        if profile.io_class == 2:
            prefix += ["-n", str(profile.io_level)]
    return Launch(prefix + cmd, unit, limited, warnings)


class Governor(threading.Thread):
    """Adjusts the RAM cap of the running scope based on how much memory the rest of the PC needs.

    Signals (checked every `interval` seconds):
      * swapping: Linux is moving apps' memory to disk -> apps need RAM -> shrink 20 %
      * free RAM below `low_water` GB                   -> shrink 20 %
      * cap above the ceiling (total RAM - what your apps use - reserve) -> shrink to ceiling
      * free RAM above `high_water`, no swapping, for `calm_secs` -> grow by `grow_step`,
        never above the ceiling

    Linux memory-pressure numbers are NOT used: QuietMoE's own cap creates pressure and the
    governor would react to itself. Free RAM alone is not trusted either: it goes UP when
    Linux swaps apps out, which looks "calm" but is the opposite.
    The cap always stays inside [profile.ram_min_gb, profile.ram_max_gb].
    """

    def __init__(self, unit: str, profile: Profile, interval: float = 2.0,
                 low_water: float = 2.0, high_water: float = 4.0, calm_secs: float = 10.0,
                 grow_step: float = 1.0, swap_threshold_mb: float = 64.0,
                 swap_ticks: int = 2, on_change=None):
        super().__init__(daemon=True)
        self.unit = unit
        self.profile = profile
        self.cap = profile.ram_gb
        self.interval = interval
        self.low_water = low_water
        self.high_water = high_water
        self.calm_secs = calm_secs
        self.grow_step = grow_step
        self.swap_threshold_pages = int(swap_threshold_mb * 1024 * 1024 / 4096)
        self.swap_ticks = swap_ticks
        self._swap_streak = 0
        self.floor = profile.ram_min_gb  # raised at runtime to QuietMoE's fixed memory + 1 GB
        self.on_change = on_change
        self._halt = threading.Event()
        self._calm_since: float | None = None
        self._last_swap = swap_out_pages()
        self.history: list[tuple[float, float]] = []

    def stop(self) -> None:
        self._halt.set()

    def swap_signal(self, other_swap_pages: int) -> bool:
        """True only for sustained heavy swapping of OTHER apps (not small background swaps)."""
        if other_swap_pages > self.swap_threshold_pages:
            self._swap_streak += 1
        else:
            self._swap_streak = 0
        return self._swap_streak >= self.swap_ticks

    def _apply(self, new_cap: float, reason: str) -> None:
        low = max(self.profile.ram_min_gb, self.floor)
        new_cap = round(min(self.profile.ram_max_gb, max(low, new_cap)), 2)
        if abs(new_cap - self.cap) < 0.05:
            return
        subprocess.run(["systemctl", "--user", "set-property", "--runtime", f"{self.unit}.scope",
                        f"MemoryHigh={int(new_cap * 1024)}M"],
                       capture_output=True, timeout=10)
        old, self.cap = self.cap, new_cap
        self.history.append((time.time(), new_cap))
        if self.on_change:
            self.on_change(old, new_cap, reason)

    def step(self, free_gb: float, now: float, swapping: bool = False,
             ceiling: float | None = None) -> None:
        if swapping:
            self._calm_since = None
            self._apply(self.cap * 0.8, "Linux is moving your apps' memory to disk")
        elif free_gb < self.low_water:
            self._calm_since = None
            self._apply(self.cap * 0.8, f"PC needs memory (free {free_gb:.1f} GB)")
        elif ceiling is not None and self.cap > ceiling + 0.25:
            self._calm_since = None
            self._apply(ceiling, "your apps are using more RAM")
        elif free_gb > self.high_water:
            if self._calm_since is None:
                self._calm_since = now
            elif now - self._calm_since >= self.calm_secs:
                self._calm_since = now
                target = self.cap + self.grow_step
                if ceiling is not None:
                    target = min(target, ceiling)
                if target > self.cap:
                    self._apply(target, f"PC is calm (free {free_gb:.1f} GB)")
        else:
            self._calm_since = None

    def run(self) -> None:
        last_ours = 0.0
        while not self._halt.wait(self.interval):
            mem = scope_memory(self.unit)
            our_anon = mem.get("anon", 0.0)
            # Never squeeze QuietMoE below its own fixed memory + 1 GB of model cache.
            self.floor = min(self.profile.ram_max_gb, our_anon + 1.0)
            # Swap activity of everything else = system total - what QuietMoE itself swapped.
            swap = swap_out_pages()
            ours = mem.get("swap", 0.0)
            ours_pages = max(0, int((ours - last_ours) * 1024**3 / 4096))
            last_ours = ours
            other_pages = max(0, swap - self._last_swap - ours_pages)
            self._last_swap = swap
            self.step(mem_available_gb(), time.time(), self.swap_signal(other_pages),
                      ceiling_gb(meminfo(), our_anon))
