"""Command line interface: quietmoe <command>."""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__, bench, hardware, llama, wizard
from .banner import c, print_banner
from . import polite
from .gguf_info import describe, inspect
from .profiles import make_profile, plan
from .runner import Server

PROFILES = ["light", "balanced", "max"]


def _load_model(name: str, hw):
    path = llama.resolve_model(name)
    info = inspect(path)
    return path, info


def _show_plan(info, profile, p) -> None:
    print(f"  Plan ({profile.name}):")
    print(f"    VRAM budget {profile.vram_gb:.1f} GB · RAM cap {profile.ram_gb:.1f} GB "
          f"(auto {profile.ram_min_gb:.1f}-{profile.ram_max_gb:.1f} GB) · "
          f"{profile.threads} threads · context {profile.ctx}")
    for n in p.notes:
        print(f"    - {n}")


def cmd_info(args) -> int:
    hw = hardware.detect(llama.MODELS_DIR)
    print_banner(hw)
    if not hw.vulkan:
        print(c("  ! Vulkan not detected. Install mesa-vulkan-drivers + vulkan-tools.", "33"))
    if hw.disk and "HDD" in hw.disk:
        print(c("  ! Models on a hard drive will be very slow. Use an SSD (NVMe is best).", "33"))
    server = llama.find_binary("llama-server")
    print(f"  llama.cpp: {server or c('not installed (run: quietmoe setup)', '33')}")
    models = llama.list_local_models()
    print(f"  Models:    {len(models)} downloaded")
    for m in models:
        print(f"    - {m}")
    if args.model:
        path, info = _load_model(args.model, hw)
        print()
        print(describe(info))
        for name in PROFILES:
            prof = make_profile(name, hw, args.ctx)
            print()
            _show_plan(info, prof, plan(info, prof))
    _print_next_steps()
    return 0


def cmd_setup(args) -> int:
    print_banner()
    print("  Building llama.cpp with Vulkan (this takes a few minutes) ...\n")
    code = llama.run_setup(args.jobs)
    if code != 0:
        print(c("\n  Build failed. Fix the problem shown above, then run:  quietmoe setup\n", "31"))
        return code
    print(c("\n  ✓ Engine ready!", "1;32"))
    _print_next_steps()
    return 0


def _print_next_steps() -> None:
    """Tell the user exactly what to run next, based on what is already done."""
    models = llama.list_local_models()
    print("\n  Next steps:")
    if not llama.find_binary("llama-server"):
        print("    1. quietmoe setup                          # build the AI engine")
    elif not models:
        print("    1. quietmoe download qwen3.5-35b-a3b       # download the model (~21 GB)")
        print("    2. quietmoe run                            # asks a few questions the first time")
        print("    3. quietmoe bench qwen3.5-35b-a3b --compare --stress  # optional: speed + smoothness test")
    else:
        name = "qwen3.5-35b-a3b" if any("qwen3.5-35b-a3b" in m.lower() for m in models) \
            else models[0]
        print(f"    1. quietmoe run                  # asks a few questions the first time")
        print(f"    2. quietmoe bench {name} --compare --stress  # optional: speed + smoothness test")
    print()


def cmd_download(args) -> int:
    print_banner()
    if not args.fast:
        print(c("  Low-priority download (your PC stays usable). Use --fast for full speed.\n", "2"))
    path = llama.download(args.model, args.quant, fast=args.fast)
    print(c(f"\n  Saved: {path}", "32"))
    info = inspect(path)
    print(describe(info))
    print(f"\n  Next steps:")
    print(f"    1. quietmoe run                  # asks a few questions the first time")
    print(f"    2. quietmoe bench {args.model} --compare --stress  # optional: speed + smoothness test\n")
    return 0


def cmd_models(args) -> int:
    print("Known short names:")
    for k, (repo, q) in llama.MODELS.items():
        print(f"  {k:18s} {repo} ({q})")
    print("\nDownloaded:")
    for m in llama.list_local_models() or ["  (none)"]:
        print(f"  {m}")
    return 0


def cmd_run(args) -> int:
    hw = hardware.detect(llama.MODELS_DIR)
    print_banner(hw)

    # Which model: argument > last used > the only one downloaded > ask.
    model = args.model
    if not model:
        model = wizard.pick_model(llama.list_local_models(), wizard.load().get("model"))
        if not model:
            print(c("  No model downloaded yet. Run:  quietmoe download qwen3.5-35b-a3b\n", "33"))
            return 1

    settings = wizard.resolve(args.profile, args.ctx, args.setup, args.no_ask)
    prof = make_profile(settings["profile"], hw, settings["ctx"])
    path, info = _load_model(model, hw)
    wizard.save({"model": path})  # remembered for next time
    print(describe(info))
    p = plan(info, prof)
    _show_plan(info, prof, p)
    server = Server(path, info, prof, p, port=args.port, extra_args=args.extra,
                    quiet_logs=not args.logs, governor=not args.no_governor)
    if args.dry_run:
        print("\n  Command:\n    " + " ".join(polite.wrap(server.command(), prof).cmd))
        return 0
    print("\n  Loading model ...")
    server.start()
    try:
        if not server.wait_ready():
            print(c("  llama-server stopped. Run again with --logs to see why.", "31"))
            return 1
        print(c(f"\n  Ready! Chat in your browser: {server.url}", "1;32"))
        print(f"  Memory for the chat:         {prof.ctx} tokens ({wizard.words(prof.ctx)}, "
              f"PDFs and messages included)")
        print(f"  OpenAI-compatible API:       {server.url}/v1")
        if server.launch and server.launch.unit:
            mem = polite.scope_memory(server.launch.unit)
            if mem:
                print(f"  RAM now: {mem.get('file', 0):.1f} GB model cache (can be dropped) + "
                      f"{mem.get('anon', 0):.1f} GB fixed")
                if mem.get("anon", 0) > prof.ram_gb:
                    print(c("  ! Fixed RAM is above the cap: weights were copied into RAM "
                            "instead of memory-mapped. Please report this on GitHub.", "33"))
        print(c("  Press Ctrl+C to stop.\n", "2"))
        while server.proc and server.proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping ...")
    finally:
        server.stop()
    return 0


def cmd_bench(args) -> int:
    hw = hardware.detect(llama.MODELS_DIR)
    print_banner(hw)
    path, info = _load_model(args.model, hw)
    print(describe(info))
    profiles = ["baseline", args.profile] if args.compare else [args.profile]
    results = []
    for name in profiles:
        prof = make_profile(name, hw, args.ctx)
        p = plan(info, prof)
        print()
        _show_plan(info, prof, p)
        server = Server(path, info, prof, p, port=args.port)
        results.append(bench.run(server, n_predict=args.tokens, rounds=args.rounds,
                                 stress=args.stress is not None,
                                 stress_gb=None if args.stress in (None, "auto")
                                 else float(args.stress)))
        print(f"  saved: {results[-1]['saved_to']}")
    print()
    print(bench.report(results))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quietmoe",
                                 description="Big MoE models, light on your PC (AMD + Vulkan).",
                                 epilog="Extra llama-server flags: quietmoe run MODEL -- --flag value")
    ap.add_argument("--version", action="version", version=f"quietmoe {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="show hardware, and the plan for a model")
    s.add_argument("model", nargs="?")
    s.add_argument("--ctx", type=int)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("setup", help="build llama.cpp with Vulkan")
    s.add_argument("--jobs", type=int, help="parallel compile jobs (default: CPU threads - 1)")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("download", help="download a model (GGUF)")
    s.add_argument("model", help="short name (see: quietmoe models) or a Hugging Face repo")
    s.add_argument("--quant", help="quantization, default Q4_K_M")
    s.add_argument("--fast", action="store_true",
                   help="use Hugging Face Xet transfer: can be faster but uses a lot of CPU")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("models", help="list known and downloaded models")
    s.set_defaults(func=cmd_models)

    for name, func, helptext in (("run", cmd_run, "run a model with a chat web UI"),
                                 ("bench", cmd_bench, "measure speed and PC responsiveness")):
        s = sub.add_parser(name, help=helptext)
        if name == "run":
            s.add_argument("model", nargs="?", help="model name or path (default: last used)")
            s.add_argument("--profile", choices=PROFILES,
                           help="light / balanced / max (default: your saved choice)")
        else:
            s.add_argument("model")
            s.add_argument("--profile", choices=PROFILES, default="light")
        s.add_argument("--ctx", type=int, help="context length")
        s.add_argument("--port", type=int, default=8080)
        s.set_defaults(func=func)
        if name == "run":
            s.add_argument("--logs", action="store_true", help="show llama.cpp output")
            s.add_argument("--no-governor", action="store_true",
                           help="keep the RAM cap fixed")
            s.add_argument("--dry-run", action="store_true",
                           help="only print the command that would run")
            s.add_argument("--setup", action="store_true",
                           help="ask the start-up questions again")
            s.add_argument("--no-ask", action="store_true",
                           help="never ask questions (use saved settings or defaults)")
            s.set_defaults(extra=[])
        else:
            s.add_argument("--compare", action="store_true",
                           help="also run stock llama.cpp without limits, side by side")
            s.add_argument("--stress", nargs="?", const="auto", metavar="GB",
                           help="simulate a heavy app using RAM while testing "
                                "(default size: half of your free RAM)")
            s.add_argument("--tokens", type=int, default=128)
            s.add_argument("--rounds", type=int, default=1)

    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:  # everything after "--" goes straight to llama-server
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if extra:
        args.extra = extra
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
