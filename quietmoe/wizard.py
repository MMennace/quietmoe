"""Friendly first-run questions for `quietmoe run`, and the saved settings file."""

from __future__ import annotations

import json
import os
import sys

from .banner import c

CONFIG_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                          "quietmoe")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# What the user wants to do -> context length (how much text the AI can keep in mind at once).
USES = [
    ("chat", "Chatting and quick questions", 8192),
    ("code", "Coding (longer files and conversations)", 16384),
    ("docs", "Long documents: PDFs, school projects, reports", 32768),
    ("mixed", "A bit of everything (mixed)", 24576),
]

# How much of the PC the AI may use -> profile.
PC_USE = [
    ("light", "Yes, I'll use my PC normally (browser, videos, apps)"),
    ("balanced", "Only light stuff (music, a few tabs)"),
    ("max", "No, give the AI everything (fastest)"),
]

PROFILE_NAMES = {"light": "Light", "balanced": "Balanced", "max": "Max"}


def words(ctx: int) -> str:
    """Rough number of English words that fit in a context (1 token ~ 0.75 words)."""
    return f"~{int(ctx * 0.75 / 1000)},000 words"


def describe(settings: dict) -> str:
    use = next((u for u in USES if u[2] == settings.get("ctx")), None)
    use_txt = use[1].split(" (")[0].split(":")[0] if use else "Custom context"
    return f"{PROFILE_NAMES.get(settings.get('profile'), settings.get('profile'))} mode · " \
           f"{use_txt} ({settings.get('ctx')} tokens, {words(settings.get('ctx', 8192))})"


def load() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(settings: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = load()
    data.update(settings)
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def choose(question: str, options: list[str], default: int = 1, reader=input) -> int:
    """Ask a numbered question. Returns the chosen index (0-based)."""
    print(f"  {c(question, '1')}")
    for i, text in enumerate(options, 1):
        mark = c(" (default)", "2") if i == default else ""
        print(f"    {i}) {text}{mark}")
    while True:
        try:
            answer = reader(f"  Choose 1-{len(options)} [{default}]: ").strip()
        except EOFError:
            answer = ""
        if not answer:
            print()
            return default - 1
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            print()
            return int(answer) - 1
        print(c("  Please type a number from the list.", "33"))


def yes_no(question: str, default: bool = True, reader=input) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = reader(f"  {question} {hint} ").strip().lower()
    except EOFError:
        answer = ""
    return default if not answer else answer.startswith("y")


def ask_settings(reader=input) -> dict:
    print(c("  A few quick questions so QuietMoE fits what you want to do.", "2"))
    print(c("  (Your answers are saved. Change them anytime with: quietmoe run --setup)\n", "2"))
    use = USES[choose("What will you mainly use the AI for?", [u[1] for u in USES], 1, reader)]
    pc = PC_USE[choose("Will you use your PC for other things while the AI runs?",
                       [p[1] for p in PC_USE], 1, reader)]
    return {"profile": pc[0], "ctx": use[2]}


def pick_model(models: list[str], saved: str | None, reader=input) -> str | None:
    """Choose which downloaded model to run."""
    if saved and any(saved == m for m in models):
        return saved
    if len(models) == 1:
        return models[0]
    if not models:
        return None
    if not interactive():
        return models[0]
    idx = choose("Which model do you want to run?", [os.path.basename(m) for m in models], 1,
                 reader)
    return models[idx]


def resolve(args_profile: str | None, args_ctx: int | None, force_setup: bool,
            no_ask: bool, reader=input) -> dict:
    """Decide profile + context: command-line flags > saved settings > questions > defaults."""
    saved = load()
    if args_profile or args_ctx:
        return {"profile": args_profile or saved.get("profile", "light"),
                "ctx": args_ctx or saved.get("ctx", 8192)}
    if not interactive() or no_ask:
        return {"profile": saved.get("profile", "light"), "ctx": saved.get("ctx", 8192)}
    if saved.get("profile") and saved.get("ctx") and not force_setup:
        print(f"  Settings: {c(describe(saved), '1')}")
        print(c("  (change with: quietmoe run --setup)\n", "2"))
        return {"profile": saved["profile"], "ctx": saved["ctx"]}
    settings = ask_settings(reader)
    save(settings)
    print(f"  Saved: {c(describe(settings), '1')}\n")
    return settings
