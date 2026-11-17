"""Start llama-server with the QuietMoE plan and limits."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request

from . import polite
from .banner import c
from .gguf_info import ModelInfo
from .llama import find_binary, supports_flag
from .profiles import Plan, Profile


class Server:
    def __init__(self, model_path: str, info: ModelInfo, profile: Profile, plan: Plan,
                 port: int = 8080, extra_args: list[str] | None = None, quiet_logs: bool = True,
                 governor: bool = True, verbose: bool = True):
        self.model_path = model_path
        self.info = info
        self.profile = profile
        self.plan = plan
        self.port = port
        self.extra_args = extra_args or []
        self.quiet_logs = quiet_logs
        self.use_governor = governor
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self.launch: polite.Launch | None = None
        self.gov: polite.Governor | None = None
        self.log_path: str | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        binary = find_binary("llama-server")
        if not binary:
            raise SystemExit("llama-server not found. Run:  quietmoe setup")
        cmd = [binary, "-m", self.model_path,
               "-ngl", str(self.plan.n_gpu_layers),
               "-c", str(self.profile.ctx),
               "-t", str(self.profile.threads),
               "--host", "127.0.0.1", "--port", str(self.port)]
        if self.info.is_moe and self.plan.n_cpu_moe:
            cmd += ["--n-cpu-moe", str(self.plan.n_cpu_moe)]
        if self.profile.polite and supports_flag(binary, "--no-repack"):
            # Repacking copies CPU-side weights into fixed RAM. We want them to stay
            # memory-mapped so Linux can drop and re-read them (= streaming).
            cmd.append("--no-repack")
        return cmd + self.extra_args

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def start(self, log_path: str | None = None) -> None:
        if self.profile.polite:
            # Don't start above what's really free once other apps' memory is counted.
            ceiling = polite.ceiling_gb(polite.meminfo(), 0.0)
            if 0 < ceiling < self.profile.ram_gb:
                self.profile.ram_gb = round(max(self.profile.ram_min_gb, ceiling), 1)
                self._log(f"  RAM cap starts at {self.profile.ram_gb:.1f} GB "
                          f"(your other apps are using a lot of memory)")
        self.launch = polite.wrap(self.command(), self.profile)
        for w in self.launch.warnings:
            self._log(c(f"  ! {w}", "33"))
        self.log_path = log_path
        out = open(log_path, "w") if log_path else (subprocess.DEVNULL if self.quiet_logs else None)
        self.proc = subprocess.Popen(self.launch.cmd, stdout=out, stderr=subprocess.STDOUT
                                     if out is not None else None, start_new_session=True)
        if self.launch.limited and self.use_governor and self.launch.unit:
            def changed(old, new, reason):
                self._log(c(f"  ~ RAM cap {old:.1f} -> {new:.1f} GB ({reason})", "2"))
            self.gov = polite.Governor(self.launch.unit, self.profile, on_change=changed)
            self.gov.start()

    def wait_ready(self, timeout: float = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if json.loads(r.read() or b"{}").get("status") == "ok":
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def complete(self, prompt: str, n_predict: int = 128, temperature: float = 0.0) -> dict:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": temperature,
                           "cache_prompt": False}).encode()
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        if self.gov:
            self.gov.stop()
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
