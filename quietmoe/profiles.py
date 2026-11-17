"""Resource profiles and the memory planner (how many expert layers go on the GPU)."""

from __future__ import annotations

from dataclasses import dataclass

from .gguf_info import GB, ModelInfo
from .hardware import Hardware

# VRAM always left free for the desktop, browser, video, etc.
DESKTOP_VRAM_RESERVE_GB = 1.0
# Scratch space llama.cpp needs on the GPU for computing (rough, grows with context).
COMPUTE_RESERVE_GB = 0.7


@dataclass
class Profile:
    name: str
    ram_gb: float          # starting RAM cap (model cache + buffers)
    ram_min_gb: float      # governor never shrinks below this
    ram_max_gb: float      # governor may grow up to this while the PC is idle
    vram_gb: float         # VRAM budget for QuietMoE
    threads: int
    nice: int              # 0 = normal, 19 = lowest CPU priority
    io_class: int          # ionice: 2 = best-effort, 3 = idle
    io_level: int
    cpu_weight: int        # cgroup CPUWeight (100 = normal)
    ctx: int               # context length
    polite: bool = True

    @property
    def priority(self) -> str:
        if not self.polite:
            return "normal (no limits)"
        return "low" if self.nice >= 10 else "normal"


def make_profile(name: str, hw: Hardware, ctx: int | None = None) -> Profile:
    prof = _make_profile(name, hw, ctx)
    prof.ram_max_gb = round(max(prof.ram_min_gb, prof.ram_max_gb), 1)
    prof.ram_gb = round(min(prof.ram_max_gb, max(prof.ram_min_gb, prof.ram_gb)), 1)
    return prof


def _make_profile(name: str, hw: Hardware, ctx: int | None = None) -> Profile:
    # If VRAM could not be detected, assume a small card so we never overfill it.
    vram_total = hw.vram_gb or 4.0
    ram_total = hw.ram_gb or 16.0
    cores = hw.cores or 4

    # Start with a high RAM cap: use free memory for speed while the PC is idle. The governor
    # shrinks it as soon as your apps need memory. Leave ~3 GB for the system and your apps.
    ram_free = hw.ram_available_gb or ram_total * 0.7
    ram_ceiling = ram_total * 0.75
    start_cap = min(ram_ceiling, max(2.0, ram_free - 3.0))

    if name == "light":
        # Lowest CPU/disk priority: QuietMoE uses all cores while the PC is idle but steps aside
        # the moment you need the CPU. ~2 GB of VRAM stays free for the desktop and browser.
        return Profile("light", ram_gb=start_cap, ram_min_gb=2.0, ram_max_gb=ram_ceiling,
                       vram_gb=vram_total - DESKTOP_VRAM_RESERVE_GB - 1.0,
                       threads=cores, nice=19, io_class=2, io_level=7,
                       cpu_weight=20, ctx=ctx or 8192)
    if name == "balanced":
        return Profile("balanced", ram_gb=start_cap, ram_min_gb=3.0, ram_max_gb=ram_total * 0.8,
                       vram_gb=vram_total - DESKTOP_VRAM_RESERVE_GB,
                       threads=cores, nice=10, io_class=2, io_level=4,
                       cpu_weight=50, ctx=ctx or 8192)
    if name == "max":
        return Profile("max", ram_gb=max(2.0, ram_total - 3.0), ram_min_gb=2.0,
                       ram_max_gb=max(2.0, ram_total - 3.0),
                       vram_gb=vram_total - DESKTOP_VRAM_RESERVE_GB,
                       threads=cores, nice=0, io_class=2, io_level=4,
                       cpu_weight=100, ctx=ctx or 8192)
    if name == "baseline":
        # Stock llama.cpp with no QuietMoE limits, used to compare in benchmarks.
        return Profile("baseline", ram_gb=ram_total, ram_min_gb=ram_total, ram_max_gb=ram_total,
                       vram_gb=vram_total - DESKTOP_VRAM_RESERVE_GB, threads=cores, nice=0,
                       io_class=2, io_level=4, cpu_weight=100, ctx=ctx or 8192, polite=False)
    raise ValueError(f"unknown profile '{name}' (use light, balanced, max or baseline)")


@dataclass
class Plan:
    n_gpu_layers: int      # -ngl: layers whose shared weights live on the GPU
    n_cpu_moe: int         # --n-cpu-moe: first N layers keep their experts on the CPU/RAM side
    gpu_expert_layers: int
    vram_used_gb: float
    notes: list[str]


def plan(info: ModelInfo, profile: Profile) -> Plan:
    notes: list[str] = []
    if info.max_ctx and profile.ctx > info.max_ctx:
        notes.append(f"context lowered to {info.max_ctx}, the most this model supports")
        profile.ctx = info.max_ctx
    kv_gb = info.kv_bytes_per_token * profile.ctx / GB
    budget = profile.vram_gb - COMPUTE_RESERVE_GB - kv_gb
    shared_gb = info.shared_bytes / GB

    if not info.is_moe:
        # Dense model: offload as many whole layers as fit.
        per_layer = info.total_bytes / max(1, info.n_layers) / GB
        ngl = max(0, min(info.n_layers, int(budget / per_layer)))
        notes.append("dense model: offloading whole layers, the rest runs on the CPU")
        return Plan(ngl, 0, 0, ngl * per_layer + kv_gb + COMPUTE_RESERVE_GB, notes)

    if shared_gb > budget:
        # Even the shared weights don't fit: put only some layers on the GPU.
        per_layer = shared_gb / max(1, info.n_layers)
        ngl = max(0, int(budget / per_layer))
        notes.append(f"shared weights ({shared_gb:.1f} GB) don't fit the {profile.vram_gb:.1f} GB "
                     f"VRAM budget, only {ngl}/{info.n_layers} layers on GPU")
        return Plan(ngl, info.n_layers, 0, ngl * per_layer + kv_gb + COMPUTE_RESERVE_GB, notes)

    free = budget - shared_gb
    gpu_layers = 0
    # --n-cpu-moe keeps the FIRST N layers' experts on CPU, so the GPU gets the last ones.
    for layer in sorted(info.expert_bytes_per_layer, reverse=True):
        size = info.expert_bytes_per_layer[layer] / GB
        if size > free:
            break
        free -= size
        gpu_layers += 1

    n_cpu_moe = info.n_layers - gpu_layers
    used = profile.vram_gb - free
    notes.append(f"{gpu_layers}/{info.n_layers} expert layers on the GPU, "
                 f"{n_cpu_moe} streamed from RAM/disk")
    if kv_gb > 0.5:
        notes.append(f"context {profile.ctx} uses up to {kv_gb:.1f} GB of VRAM")
    return Plan(999, n_cpu_moe, gpu_layers, used, notes)
