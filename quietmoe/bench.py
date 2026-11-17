"""Benchmark: how fast is the model, AND how usable does the PC stay while it runs?

"PC usability" is measured with a responsiveness probe: a normal-priority thread (like
your browser or game) asks to wake up every 10 ms and we record how late it actually
wakes up. Big delays = stutter. We also sample free RAM and Linux pressure stats.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time

from . import polite
from .banner import c
from .gguf_info import GB
from .hardware import vram_used_gb
from .llama import BENCH_DIR

PROMPTS = {
    "explain": "Explain in simple words how a refrigerator keeps food cold.",
    "code": "Write a Python function that checks if a string is a palindrome, with tests.",
    "summary": ("Summarize the following text in three bullet points.\n\n" +
                ("Local language models let people run AI on their own computers. They protect "
                 "privacy because nothing leaves the machine, they work offline, and they cost "
                 "nothing per message. The downside is that big models need a lot of memory, "
                 "so most people can only run small ones. Mixture-of-experts models change "
                 "this because only a few experts are used for each token. ") * 6),
}


class Probe(threading.Thread):
    """Measures wake-up lateness and samples system stats once per second."""

    def __init__(self, unit: str | None):
        super().__init__(daemon=True)
        self.unit = unit
        self.lateness_ms: list[float] = []
        self.samples: list[dict] = []
        self._halt = threading.Event()
        self._cgroup = polite.scope_cgroup(unit) if unit else None
        self._swap_start = polite.swap_out_pages()
        self.swapped_mb = 0.0

    def stop(self) -> None:
        self._halt.set()
        self.swapped_mb = (polite.swap_out_pages() - self._swap_start) * 4096 / 1024**2

    def run(self) -> None:
        tick = 0.010
        last_sample = 0.0
        while not self._halt.is_set():
            t0 = time.perf_counter()
            time.sleep(tick)
            self.lateness_ms.append((time.perf_counter() - t0 - tick) * 1000)
            now = time.time()
            if now - last_sample >= 1.0:
                last_sample = now
                self.samples.append(self._sample())

    def _sample(self) -> dict:
        if not self._cgroup and self.unit:
            self._cgroup = polite.scope_cgroup(self.unit)
        s = {
            "free_ram_gb": round(polite.mem_available_gb(), 2),
            "mem_full": polite.pressure("memory")["full"],
            "io_some": polite.pressure("io")["some"],
            "cpu_some": polite.pressure("cpu")["some"],
            "vram_used_gb": round(vram_used_gb(), 2),
        }
        if self._cgroup:
            try:
                with open(os.path.join(self._cgroup, "memory.current")) as f:
                    s["quietmoe_ram_gb"] = round(int(f.read()) / GB, 2)
            except OSError:
                pass
        return s

    def summary(self) -> dict:
        lat = sorted(self.lateness_ms) or [0.0]
        def pct(p):
            return round(lat[min(len(lat) - 1, int(len(lat) * p))], 2)
        def avg(key):
            vals = [s[key] for s in self.samples if key in s]
            return round(statistics.mean(vals), 2) if vals else None
        def peak(key, fn=max):
            vals = [s[key] for s in self.samples if key in s]
            return round(fn(vals), 2) if vals else None
        return {
            "lag_p50_ms": pct(0.50), "lag_p99_ms": pct(0.99), "lag_max_ms": round(lat[-1], 2),
            "stutters_over_50ms": sum(1 for x in lat if x > 50),
            "min_free_ram_gb": peak("free_ram_gb", min),
            "avg_mem_pressure": avg("mem_full"), "avg_io_pressure": avg("io_some"),
            "avg_cpu_pressure": avg("cpu_some"),
            "peak_vram_gb": peak("vram_used_gb"),
            "peak_quietmoe_ram_gb": peak("quietmoe_ram_gb"),
            "swapped_out_mb": round(self.swapped_mb),
        }


def run(server, n_predict: int = 128, rounds: int = 1, log=print,
        stress: bool = False, stress_gb: float | None = None) -> dict:
    """Start the server, run the prompts, stop it. Returns a result dict.

    With stress=True a simulated heavy app runs the whole time (see stress.py).
    """
    os.makedirs(BENCH_DIR, exist_ok=True)
    log_path = os.path.join(BENCH_DIR, f"server-{server.profile.name}-{int(time.time())}.log")
    app = None
    if stress:
        from .stress import StressApp
        app = StressApp(stress_gb)
        log(f"  starting simulated heavy app ({app.anon_gb:g} GB RAM + "
            f"{app.file_gb:g} GB working file) ...")
        if not app.start():
            log("  ! simulated app failed to start, continuing without it")
            app.stop()
            app = None
    t0 = time.time()
    server.start(log_path=log_path)
    probe = Probe(server.launch.unit if server.launch else None)
    probe.start()
    result: dict = {"profile": server.profile.name, "model": os.path.basename(server.model_path),
                    "plan": {"n_cpu_moe": server.plan.n_cpu_moe,
                             "gpu_expert_layers": server.plan.gpu_expert_layers,
                             "threads": server.profile.threads, "ctx": server.profile.ctx,
                             "ram_cap_gb": server.profile.ram_gb if server.profile.polite else None},
                    "limited": bool(server.launch and server.launch.limited), "runs": []}
    try:
        log(f"  loading model ({server.profile.name}) ...")
        if not server.wait_ready():
            raise RuntimeError(f"llama-server failed to start, see log: {log_path}")
        result["load_s"] = round(time.time() - t0, 1)
        log(f"  ready in {result['load_s']} s")
        for r in range(rounds):
            for name, prompt in PROMPTS.items():
                out = server.complete(prompt, n_predict=n_predict)
                t = out.get("timings", {})
                run_res = {"prompt": name, "round": r,
                           "gen_tok_s": round(t.get("predicted_per_second", 0), 2),
                           "prompt_tok_s": round(t.get("prompt_per_second", 0), 1),
                           "tokens": t.get("predicted_n")}
                result["runs"].append(run_res)
                log(f"    {name:8s} {run_res['gen_tok_s']:6.2f} tok/s  "
                    f"(prompt {run_res['prompt_tok_s']} tok/s)")
    finally:
        probe.stop()
        server.stop()
        if app:
            result["stress"] = app.stop()
    gens = [r["gen_tok_s"] for r in result["runs"]]
    result["avg_gen_tok_s"] = round(statistics.mean(gens), 2) if gens else 0
    result["system"] = probe.summary()
    if server.gov:
        result["governor_changes"] = len(server.gov.history)
    path = os.path.join(BENCH_DIR, f"bench-{server.profile.name}-{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    result["saved_to"] = path
    return result


def report(results: list[dict]) -> str:
    rows = [
        ("Speed (tokens/s)", "avg_gen_tok_s", None),
        ("Load time (s)", "load_s", None),
        ("Lag typical (ms)", "lag_p50_ms", "system"),
        ("Lag worst 1% (ms)", "lag_p99_ms", "system"),
        ("Stutters > 50 ms", "stutters_over_50ms", "system"),
        ("Lowest free RAM (GB)", "min_free_ram_gb", "system"),
        ("QuietMoE RAM peak (GB)", "peak_quietmoe_ram_gb", "system"),
        ("VRAM peak (GB)", "peak_vram_gb", "system"),
        ("Memory pressure (%)", "avg_mem_pressure", "system"),
        ("Disk pressure (%)", "avg_io_pressure", "system"),
        ("Apps pushed to swap (MB)", "swapped_out_mb", "system"),
        ("App file reload avg (ms)", "app_reload_avg_ms", "stress"),
        ("App file reload worst (ms)", "app_reload_max_ms", "stress"),
        ("App memory touch worst (ms)", "app_touch_max_ms", "stress"),
    ]
    if not any("stress" in r for r in results):
        rows = [row for row in rows if row[2] != "stress"]
    names = [r["profile"] for r in results]
    width = max(12, *(len(n) for n in names))
    label_w = 29
    lines = ["  " + "".ljust(label_w) + "".join(c(n.ljust(width), "1") for n in names)]
    for label, key, sub in rows:
        vals = []
        for r in results:
            v = (r.get(sub, {}) if sub else r).get(key)
            vals.append("-" if v is None else str(v))
        lines.append("  " + label.ljust(label_w) + "".join(v.ljust(width) for v in vals))
    lines.append("")
    lines.append(c("  Lower lag / stutters / pressure = PC stays smoother. Higher free RAM = "
                   "more room for your apps.", "2"))
    if any("stress" in r for r in results):
        lines.append(c("  App reload / touch = how a heavy app feels while the AI runs. Lower = "
                       "your apps keep their data in RAM.", "2"))
    return "\n".join(lines)
