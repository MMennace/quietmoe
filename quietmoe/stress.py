"""Simulated heavy app for `quietmoe bench --stress`.

A real PC under load has apps that (1) hold a lot of RAM and (2) keep re-reading their own
files (browser cache, game assets, libraries). If the AI pushes that data out of RAM, the app
has to wait for the disk, and you feel stutter when switching windows or loading things.

This process does exactly that and measures it:
  * "memory touch": time to touch a random 256 MB part of its RAM
  * "file reload":  time to re-read its ~1 GB working file
Both stay fast if the app's data stays in RAM, and jump if the AI pushed it out.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import statistics
import time

from . import polite

GB = 1024**3
MB = 1024**2
PAGE = 4096

CACHE_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                         "quietmoe")


def safe_sizes(anon_gb: float | None, file_gb: float = 1.0) -> tuple[float, float]:
    """Pick sizes that create real memory pressure but never risk freezing the PC."""
    free = polite.mem_available_gb()
    if anon_gb is None:
        anon_gb = free * 0.5
    # Always leave at least 3 GB for the system, the desktop and the AI's fixed memory.
    anon_gb = max(0.0, min(anon_gb, free - file_gb - 3.0))
    return round(anon_gb, 1), file_gb


def _touch(buf: bytearray, start: int, length: int) -> None:
    end = min(len(buf), start + length)
    for i in range(start, end, PAGE):
        buf[i] = (buf[i] + 1) & 0xFF


def _worker(anon_gb: float, file_gb: float, interval: float, cache_dir: str,
            ready, stop, out) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"stress-{os.getpid()}.bin")
    reload_ms: list[float] = []
    touch_ms: list[float] = []
    try:
        block = os.urandom(MB)
        with open(path, "wb") as f:
            for _ in range(int(file_gb * 1024)):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        with open(path, "rb") as f:            # pull it into the page cache
            while f.read(8 * MB):
                pass

        buf = bytearray(int(anon_gb * GB)) if anon_gb > 0 else bytearray()
        _touch(buf, 0, len(buf))                # actually commit the memory
        ready.set()

        chunk = 256 * MB
        while not stop.wait(interval):
            t0 = time.perf_counter()
            with open(path, "rb") as f:
                while f.read(8 * MB):
                    pass
            reload_ms.append((time.perf_counter() - t0) * 1000)
            if buf:
                start = random.randrange(0, max(1, len(buf) - chunk))
                t0 = time.perf_counter()
                _touch(buf, start, chunk)
                touch_ms.append((time.perf_counter() - t0) * 1000)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        out.put({"reload_ms": reload_ms, "touch_ms": touch_ms})


class StressApp:
    def __init__(self, anon_gb: float | None = None, file_gb: float = 1.0, interval: float = 1.0):
        self.anon_gb, self.file_gb = safe_sizes(anon_gb, file_gb)
        self.interval = interval
        ctx = mp.get_context("spawn")
        self._ready = ctx.Event()
        self._stop = ctx.Event()
        self._out = ctx.Queue()
        self._proc = ctx.Process(target=_worker, daemon=True,
                                 args=(self.anon_gb, self.file_gb, interval, CACHE_DIR,
                                       self._ready, self._stop, self._out))
        self.results: dict = {}

    def start(self, timeout: float = 300) -> bool:
        self._proc.start()
        return self._ready.wait(timeout)

    def stop(self) -> dict:
        self._stop.set()
        try:
            self.results = self._out.get(timeout=120)
        except Exception:
            self.results = {"reload_ms": [], "touch_ms": []}
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.kill()
        return self.summary()

    def summary(self) -> dict:
        def stats(vals):
            if not vals:
                return None, None
            return round(statistics.mean(vals), 1), round(max(vals), 1)
        r_avg, r_max = stats(self.results.get("reload_ms", []))
        t_avg, t_max = stats(self.results.get("touch_ms", []))
        return {"app_ram_gb": self.anon_gb, "app_file_gb": self.file_gb,
                "app_reload_avg_ms": r_avg, "app_reload_max_ms": r_max,
                "app_touch_avg_ms": t_avg, "app_touch_max_ms": t_max,
                "samples": len(self.results.get("reload_ms", []))}
