"""Detect GPU, VRAM, RAM, CPU and disk type on Linux. Everything is best-effort."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class Hardware:
    gpu_name: str | None
    vram_gb: float
    vram_used_gb: float
    vulkan: bool
    ram_gb: float
    ram_available_gb: float
    cpu_name: str
    cores: int
    threads: int
    disk: str | None


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_gpu() -> tuple[str | None, float, float]:
    """Return (name, vram_total_gb, vram_used_gb). Uses amdgpu sysfs, falls back to lspci."""
    best = (None, 0.0, 0.0)
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        total = _read(os.path.join(dev, "mem_info_vram_total"))
        if not total:
            continue
        used = _read(os.path.join(dev, "mem_info_vram_used")) or "0"
        total_gb = int(total) / 1024**3
        name = _gpu_name_from_lspci(dev) or "AMD GPU"
        if total_gb > best[1]:
            best = (name, total_gb, int(used) / 1024**3)
    if best[0] is None:
        out = _run(["lspci"])
        for line in out.splitlines():
            if re.search(r"VGA|3D|Display", line):
                best = (line.split(":", 2)[-1].strip(), 0.0, 0.0)
                break
    return best


def _gpu_name_from_lspci(dev_path: str) -> str | None:
    slot = os.path.basename(os.path.realpath(dev_path))
    out = _run(["lspci", "-s", slot])
    if not out:
        return None
    name = out.split(":", 2)[-1].strip()
    # "Advanced Micro Devices, Inc. [AMD/ATI] Navi 23 [Radeon RX 6600/6600 XT/6600M] (rev c1)"
    m = re.findall(r"\[([^\]]+)\]", name)
    if m:
        return "AMD " + m[-1]
    return name


def detect_vulkan() -> bool:
    if shutil.which("vulkaninfo"):
        out = _run(["vulkaninfo", "--summary"], timeout=10)
        return "deviceName" in out
    return bool(glob.glob("/usr/share/vulkan/icd.d/*.json") or glob.glob("/etc/vulkan/icd.d/*.json"))


def detect_ram() -> tuple[float, float]:
    info = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, val = line.partition(":")
        info[key] = int(val.split()[0]) * 1024 if val.split() else 0
    return info.get("MemTotal", 0) / 1024**3, info.get("MemAvailable", 0) / 1024**3


def detect_cpu() -> tuple[str, int, int]:
    text = _read("/proc/cpuinfo") or ""
    name = "unknown CPU"
    m = re.search(r"model name\s*:\s*(.+)", text)
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip()
    threads = os.cpu_count() or 1
    cores = len({
        (p.group(1), c.group(1))
        for block in text.split("\n\n")
        if (p := re.search(r"physical id\s*:\s*(\d+)", block))
        and (c := re.search(r"core id\s*:\s*(\d+)", block))
    }) or threads
    return name, cores, threads


def detect_disk(path: str) -> str | None:
    """Describe the disk that holds `path` (NVMe / SATA SSD / HDD)."""
    try:
        dev_id = os.stat(path).st_dev
    except OSError:
        return None
    major, minor = os.major(dev_id), os.minor(dev_id)
    sys_dev = os.path.realpath(f"/sys/dev/block/{major}:{minor}")
    # Walk up from partition to the whole disk.
    disk = sys_dev
    while disk != "/" and not os.path.exists(os.path.join(disk, "queue", "rotational")):
        disk = os.path.dirname(disk)
    name = os.path.basename(disk)
    rotational = _read(os.path.join(disk, "queue", "rotational"))
    if name.startswith("nvme"):
        kind = "NVMe SSD"
    elif rotational == "1":
        kind = "HDD (slow for streaming!)"
    elif rotational == "0":
        kind = "SATA SSD"
    else:
        return None
    return f"{kind} ({name})"


def detect(model_dir: str | None = None) -> Hardware:
    gpu, vram, vram_used = detect_gpu()
    ram, ram_avail = detect_ram()
    cpu, cores, threads = detect_cpu()
    disk = detect_disk(model_dir or os.path.expanduser("~"))
    return Hardware(gpu, vram, vram_used, detect_vulkan(), ram, ram_avail, cpu, cores, threads, disk)


def vram_used_gb() -> float:
    """Fast VRAM-used reading (sysfs only, no subprocess). Max over all AMD cards."""
    best = 0.0
    for path in glob.glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_used"):
        val = _read(path)
        if val:
            best = max(best, int(val) / 1024**3)
    return best
