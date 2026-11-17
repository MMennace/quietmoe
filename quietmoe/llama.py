"""Find / build llama.cpp (Vulkan) and download models."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

DATA_DIR = os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
                        "quietmoe")
LLAMA_DIR = os.path.join(DATA_DIR, "llama.cpp")
MODELS_DIR = os.path.join(DATA_DIR, "models")
BENCH_DIR = os.path.join(DATA_DIR, "bench")

SETUP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_llamacpp.sh")

# Short names -> (Hugging Face repo, default quant)
MODELS = {
    "qwen3.5-35b-a3b": ("unsloth/Qwen3.5-35B-A3B-GGUF", "Q4_K_M"),
    "qwen3-30b-a3b": ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Q4_K_M"),
    "gpt-oss-20b": ("unsloth/gpt-oss-20b-GGUF", "Q4_K_M"),
}


def find_binary(name: str) -> str | None:
    """Look for a llama.cpp binary: $QUIETMOE_LLAMA_BIN dir, our own build, then PATH."""
    candidates = []
    if os.environ.get("QUIETMOE_LLAMA_BIN"):
        candidates.append(os.path.join(os.environ["QUIETMOE_LLAMA_BIN"], name))
    candidates.append(os.path.join(LLAMA_DIR, "build", "bin", name))
    for path in candidates:
        if os.access(path, os.X_OK):
            return path
    return shutil.which(name)


_HELP_CACHE: dict[str, str] = {}


def supports_flag(binary: str, flag: str) -> bool:
    """Check `binary --help` for a flag (llama.cpp flags change between versions)."""
    if binary not in _HELP_CACHE:
        try:
            r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
            _HELP_CACHE[binary] = r.stdout + r.stderr
        except (OSError, subprocess.SubprocessError):
            _HELP_CACHE[binary] = ""
    return flag in _HELP_CACHE[binary]


def be_polite() -> None:
    """Lower this process's CPU and disk priority, so long jobs don't slow the PC down."""
    try:
        os.nice(15)
    except OSError:
        pass
    if shutil.which("ionice"):
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], capture_output=True)


def run_setup(jobs: int | None = None) -> int:
    if not os.path.exists(SETUP_SCRIPT):
        print(f"setup script not found: {SETUP_SCRIPT}", file=sys.stderr)
        return 1
    be_polite()  # the compiler inherits the low priority
    env = dict(os.environ)
    # Leave one core free so the desktop stays responsive while compiling.
    env["QUIETMOE_JOBS"] = str(jobs or max(1, (os.cpu_count() or 2) - 1))
    return subprocess.call(["bash", SETUP_SCRIPT, LLAMA_DIR], env=env)


def _pick_files(files: list[str], quant: str) -> list[str]:
    quant_l = quant.lower()
    picked = []
    for f in files:
        base = os.path.basename(f).lower()
        if not base.endswith(".gguf") or "mmproj" in base:
            continue
        if base.endswith(f"-{quant_l}.gguf") or f"-{quant_l}-000" in base:
            picked.append(f)
    return sorted(picked)


def download(model: str, quant: str | None = None, fast: bool = False) -> str:
    """Download a GGUF (all split parts) and return the path of the first file.

    By default it runs at low priority and uses plain HTTP downloads. Hugging Face's
    Xet transfer is faster on some connections but can use 100% of the CPU.
    """
    be_polite()
    if not fast:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit("huggingface_hub is missing: pip install huggingface_hub")

    repo, default_quant = MODELS.get(model.lower(), (model, "Q4_K_M"))
    quant = quant or default_quant
    files = _pick_files(list_repo_files(repo), quant)
    if not files:
        raise SystemExit(f"no {quant} GGUF found in {repo}. Try --quant with another value.")

    target = os.path.join(MODELS_DIR, repo.replace("/", "__"))
    os.makedirs(target, exist_ok=True)
    paths = []
    for f in files:
        print(f"  downloading {repo}/{f} ...")
        paths.append(hf_hub_download(repo, f, local_dir=target))
    return paths[0]


def list_local_models() -> list[str]:
    found = []
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f.lower():
                if "-of-" in f and "-00001-of-" not in f:
                    continue
                found.append(os.path.join(root, f))
    return sorted(found)


def resolve_model(model: str) -> str:
    """Accept a path, or a short name / repo that was downloaded before."""
    if os.path.exists(model):
        return model
    key = MODELS.get(model.lower(), (model, ""))[0].replace("/", "__").lower()
    for path in list_local_models():
        if key in path.lower():
            return path
    raise SystemExit(f"model '{model}' not found. Download it first:  quietmoe download {model}")
