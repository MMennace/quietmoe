"""Startup banner shown in the terminal."""

import os
import sys

from . import __version__, __author__, __github__

LOGO = r"""
   ___        _      _   __  __       ___
  / _ \ _   _(_) ___| |_|  \/  | ___ | __|
 | | | | | | | |/ _ \ __| |\/| |/ _ \| _|
 | |_| | |_| | |  __/ |_| |  | | (_) | |__
  \__\_\\__,_|_|\___|\__|_|  |_|\___/|____|
"""


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text: str, code: str) -> str:
    """Wrap text in an ANSI color code if the terminal supports it."""
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


# Logo colors (256-color approximations of #22d3ee cyan and #f97316 orange).
CYAN = "38;5;45"
ORANGE = "38;5;208"


def print_banner(hw=None, limits=None) -> None:
    print(c(LOGO, "1;" + CYAN))
    name = c("Quiet", "1;97") + c("MoE", "1;" + CYAN)
    print(f"  {name} v{__version__} · big MoE models, light on your PC")
    print(f"  by {c('@' + __author__, '1;' + ORANGE)} · {__github__}")
    print()
    if hw is not None:
        gpu = hw.gpu_name or "no GPU detected"
        vram = f" ({hw.vram_gb:.0f} GB)" if hw.vram_gb else ""
        vk = c("Vulkan ✓", "32") if hw.vulkan else c("Vulkan ✗", "31")
        print(f"  GPU:   {gpu}{vram} · {vk}")
        ram_line = f"{hw.ram_gb:.0f} GB"
        if limits is not None:
            ram_line += f"  (QuietMoE limit: {limits.ram_gb:g} GB)"
        print(f"  RAM:   {ram_line}")
        print(f"  CPU:   {hw.cpu_name} · {hw.cores} cores / {hw.threads} threads")
        if hw.disk:
            print(f"  Disk:  {hw.disk}")
    if limits is not None:
        print(f"  Mode:  {c(limits.name.capitalize(), '1;' + ORANGE)} · priority: {limits.priority}")
    print()
