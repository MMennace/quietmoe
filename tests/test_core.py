"""Tests that run anywhere (no GPU needed). Run with:  python -m pytest tests"""

import os

import numpy as np
import pytest

from quietmoe import gguf_info, llama, polite
from quietmoe.hardware import Hardware
from quietmoe.profiles import make_profile, plan

MB = 1024**2


def make_fake_moe(path, n_layers=8, expert_mb=4, shared_mb=2):
    import gguf
    w = gguf.GGUFWriter(path, "fakemoe")
    w.add_block_count(n_layers)
    w.add_uint32("fakemoe.expert_count", 64)
    w.add_uint32("fakemoe.expert_used_count", 4)
    w.add_uint32("fakemoe.attention.head_count", 8)
    w.add_uint32("fakemoe.attention.head_count_kv", 2)
    w.add_uint32("fakemoe.embedding_length", 512)
    per = expert_mb * MB // 3 // 4  # float32 elements per expert tensor
    for i in range(n_layers):
        for kind in ("gate", "up", "down"):
            w.add_tensor(f"blk.{i}.ffn_{kind}_exps.weight", np.zeros(per, dtype=np.float32))
        w.add_tensor(f"blk.{i}.attn_q.weight", np.zeros(shared_mb * MB // 4 // n_layers,
                                                         dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture
def fake_model(tmp_path):
    p = str(tmp_path / "fake.gguf")
    make_fake_moe(p)
    return p


def hw(vram=8.0, ram=16.0):
    return Hardware("AMD Radeon RX 6600 XT", vram, 0.5, True, ram, 12.0, "Ryzen 7", 8, 16,
                    "NVMe SSD (nvme0n1)")


def test_inspect(fake_model):
    info = gguf_info.inspect(fake_model)
    assert info.is_moe
    assert info.n_layers == 8
    assert info.n_experts == 64 and info.n_experts_used == 4
    assert len(info.expert_bytes_per_layer) == 8
    assert info.shared_bytes < info.expert_bytes
    assert info.active_bytes_per_token < info.total_bytes
    assert "Experts" in gguf_info.describe(info)


def test_plan_fits_everything_with_big_budget(fake_model):
    info = gguf_info.inspect(fake_model)
    p = plan(info, make_profile("max", hw()))
    assert p.n_cpu_moe == 0 and p.gpu_expert_layers == 8


def test_plan_puts_last_layers_on_gpu():
    info = gguf_info.ModelInfo("x.gguf", "fake", 40, 21 * 1024**3, int(1.5 * 1024**3),
                               {i: int(0.48 * 1024**3) for i in range(40)}, 256, 8, 0)
    light = plan(info, make_profile("light", hw()))
    balanced = plan(info, make_profile("balanced", hw()))
    assert 0 < light.gpu_expert_layers < balanced.gpu_expert_layers < 40
    assert light.n_cpu_moe == 40 - light.gpu_expert_layers
    assert light.vram_used_gb <= make_profile("light", hw()).vram_gb + 1e-6


def test_plan_shared_too_big():
    info = gguf_info.ModelInfo("x.gguf", "fake", 40, 30 * 1024**3, 10 * 1024**3,
                               {i: 1024**3 for i in range(40)}, 128, 8, 0)
    p = plan(info, make_profile("light", hw()))
    assert p.n_gpu_layers < 40 and p.n_cpu_moe == 40


def test_split_files(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"m-0000{i}-of-00003.gguf").write_bytes(b"")
    parts = gguf_info.split_files(str(tmp_path / "m-00001-of-00003.gguf"))
    assert len(parts) == 3


def test_pick_files():
    files = ["Qwen-Q4_K_M.gguf", "Qwen-UD-Q4_K_XL.gguf", "mmproj-F16.gguf",
             "Q8_0/Qwen-Q8_0-00001-of-00002.gguf", "Q8_0/Qwen-Q8_0-00002-of-00002.gguf"]
    assert llama._pick_files(files, "Q4_K_M") == ["Qwen-Q4_K_M.gguf"]
    assert len(llama._pick_files(files, "Q8_0")) == 2


class FakeGov(polite.Governor):
    def _apply(self, new_cap, reason):
        new_cap = min(self.profile.ram_max_gb, max(self.profile.ram_min_gb, new_cap))
        self.history.append((0, new_cap))
        self.cap = new_cap


def test_governor_shrinks_and_grows():
    prof = make_profile("light", hw())
    g = FakeGov("u", prof, calm_secs=10)
    g.step(free_gb=0.8, now=0)                      # PC needs memory
    assert g.cap < prof.ram_gb
    for _ in range(10):
        g.step(free_gb=0.5, now=0)
    assert g.cap == prof.ram_min_gb                 # never below the floor
    t = 0
    for _ in range(40):                             # long calm period
        t += 10
        g.step(free_gb=10, now=t)
    assert g.cap == prof.ram_max_gb                 # never above the ceiling


def test_governor_ignores_its_own_pressure():
    """Regression: lots of free RAM must never shrink the cap (bench on RX 6600 XT, v1.2)."""
    prof = make_profile("light", hw())
    g = FakeGov("u", prof, calm_secs=10)
    start = g.cap
    for t in range(0, 300, 2):
        g.step(free_gb=13.0, now=t)
    assert g.cap >= start


def test_wrap_without_limits():
    prof = make_profile("baseline", hw())
    launch = polite.wrap(["llama-server"], prof)
    assert launch.cmd == ["llama-server"] and not launch.limited


def test_wrap_light_has_low_priority():
    launch = polite.wrap(["llama-server"], make_profile("light", hw()))
    assert launch.cmd[-1] == "llama-server"
    if "nice" in launch.cmd:
        assert launch.cmd[launch.cmd.index("nice") + 2] == "19"


# ---------- v1.4: start-up dialog, KV estimate, stress app ----------

def test_kv_counts_only_attention_layers(tmp_path):
    import gguf
    path = str(tmp_path / "hybrid.gguf")
    w = gguf.GGUFWriter(path, "hyb")
    w.add_block_count(8)
    w.add_uint32("hyb.attention.head_count", 8)
    w.add_uint32("hyb.attention.head_count_kv", 2)
    w.add_uint32("hyb.attention.key_length", 128)
    w.add_uint32("hyb.context_length", 32768)
    for i in range(8):
        w.add_tensor(f"blk.{i}.ffn_up_exps.weight", np.zeros(256, dtype=np.float32))
        if i % 4 == 3:  # full attention only every 4th layer, like Qwen3.5
            w.add_tensor(f"blk.{i}.attn_k.weight", np.zeros(64, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    info = gguf_info.inspect(path)
    assert info.attn_layers == 2
    assert info.kv_bytes_per_token == 2 * 2 * (128 + 128) * 2
    assert info.max_ctx == 32768


def test_ctx_is_clamped_to_model_max():
    info = gguf_info.ModelInfo("x.gguf", "fake", 40, 21 * 1024**3, int(1.5 * 1024**3),
                               {i: int(0.48 * 1024**3) for i in range(40)}, 256, 8, 0,
                               max_ctx=16384)
    prof = make_profile("light", hw(), ctx=65536)
    p = plan(info, prof)
    assert prof.ctx == 16384 and any("lowered" in n for n in p.notes)


def test_light_starts_high_and_uses_all_cores():
    h = hw()
    light = make_profile("light", h)
    assert light.threads == h.cores and light.nice == 19
    assert light.ram_gb > 5            # 12 GB free -> starts around 9 GB
    assert light.ram_gb <= light.ram_max_gb


def test_wizard_questions_and_saved_settings(tmp_path, monkeypatch):
    from quietmoe import wizard
    monkeypatch.setattr(wizard, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(wizard, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(wizard, "interactive", lambda: True)
    answers = iter(["3", "2"])                       # documents, balanced
    s = wizard.resolve(None, None, False, False, reader=lambda _: next(answers))
    assert s == {"profile": "balanced", "ctx": 32768}
    assert wizard.load()["profile"] == "balanced"
    # next start: saved settings are reused without questions
    s2 = wizard.resolve(None, None, False, False, reader=lambda _: pytest.fail("asked again"))
    assert s2 == s
    # flags always win
    assert wizard.resolve("max", None, False, False)["profile"] == "max"
    # --setup asks again; empty answers pick the defaults
    s3 = wizard.resolve(None, None, True, False, reader=lambda _: "")
    assert s3 == {"profile": "light", "ctx": 8192}


def test_wizard_rejects_bad_input():
    from quietmoe import wizard
    answers = iter(["abc", "9", "2"])
    assert wizard.choose("q?", ["a", "b", "c"], reader=lambda _: next(answers)) == 1


def test_pick_model(tmp_path):
    from quietmoe import wizard
    assert wizard.pick_model([], None) is None
    assert wizard.pick_model(["/m/a.gguf"], None) == "/m/a.gguf"
    assert wizard.pick_model(["/m/a.gguf", "/m/b.gguf"], "/m/b.gguf") == "/m/b.gguf"


def test_stress_app_runs(tmp_path, monkeypatch):
    from quietmoe import stress
    monkeypatch.setattr(stress, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(stress, "safe_sizes", lambda a, f: (0.05, 0.02))
    app = stress.StressApp(interval=0.2)
    assert app.start(timeout=60)
    assert list(tmp_path.glob("stress-*.bin"))       # the file really is in our folder
    import time
    time.sleep(1.2)
    s = app.stop()
    assert s["samples"] >= 2 and s["app_reload_avg_ms"] is not None
    assert s["app_touch_max_ms"] is not None
    assert tmp_path.exists() and not list(tmp_path.glob("stress-*.bin"))  # used + cleaned up


# ---------- v1.5: swap-aware governor ----------

def test_ceiling_counts_other_apps():
    info = {"MemTotal": 15.2, "AnonPages": 9.5, "Shmem": 0.5}   # 6.9 GB app + desktop + us
    assert polite.ceiling_gb(info, our_anon_gb=1.0) == pytest.approx(15.2 - 9.0 - 3.0)


def test_governor_shrinks_when_apps_get_swapped():
    """Regression for the v1.4 stress run: free RAM looked high because the app was swapped out."""
    prof = make_profile("light", hw())
    g = FakeGov("u", prof, calm_secs=10)
    start = g.cap
    g.step(free_gb=12.7, now=0, swapping=True, ceiling=11.0)
    assert g.cap < start


def test_governor_respects_ceiling():
    prof = make_profile("light", hw())
    g = FakeGov("u", prof, calm_secs=10)
    g.step(free_gb=9.0, now=0, ceiling=3.5)          # apps now use most of the RAM
    assert g.cap == pytest.approx(max(prof.ram_min_gb, 3.5))
    for t in range(0, 200, 10):                      # calm, but ceiling still low
        g.step(free_gb=9.0, now=t, ceiling=3.5)
    assert g.cap <= 3.5 + 1e-6


def test_governor_grows_up_to_ceiling_only():
    prof = make_profile("light", hw())
    g = FakeGov("u", prof, calm_secs=10)
    g.cap = 4.0
    for t in range(0, 300, 10):
        g.step(free_gb=10.0, now=t, ceiling=6.0)
    assert g.cap == pytest.approx(6.0)


# ---------- v1.6: no false swap alarms ----------

def test_small_or_short_swapping_is_ignored():
    """Regression for the v1.5 idle run: background swapping must not shrink the cap."""
    g = FakeGov("u", make_profile("light", hw()))
    small = 1000                                   # ~4 MB per tick: normal background noise
    big = g.swap_threshold_pages * 2
    assert not any(g.swap_signal(small) for _ in range(20))
    assert not g.swap_signal(big)                  # one spike is not enough
    assert not g.swap_signal(small)                # streak broken
    assert not g.swap_signal(big)
    assert g.swap_signal(big)                      # sustained heavy swapping -> react


def test_floor_protects_quietmoe_itself():
    g = FakeGov("u", make_profile("light", hw()))
    g.floor = 3.5                                  # QuietMoE's fixed memory + 1 GB
    real_apply = polite.Governor._apply
    calls = []
    def fake_run(*a, **k):
        calls.append(a)
    import subprocess as sp
    old = sp.run
    sp.run = fake_run
    try:
        for _ in range(10):
            real_apply(g, g.cap * 0.5, "test")
    finally:
        sp.run = old
    assert g.cap == pytest.approx(3.5)


def test_scope_never_swaps():
    launch = polite.wrap(["llama-server"], make_profile("light", hw()))
    if launch.limited:
        assert "MemorySwapMax=0" in launch.cmd


def test_wizard_mixed_option(tmp_path, monkeypatch):
    from quietmoe import wizard
    monkeypatch.setattr(wizard, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(wizard, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(wizard, "interactive", lambda: True)
    answers = iter(["4", "1"])
    s = wizard.resolve(None, None, True, False, reader=lambda _: next(answers))
    assert s == {"profile": "light", "ctx": 24576}
    assert "A bit of everything" in wizard.describe(s)
