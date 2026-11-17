<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-wordmark-dark.png">
  <img src="assets/logo-wordmark-light.png" alt="QuietMoE" width="560">
</picture>

### Run a 35B AI on your gaming PC. Your PC stays usable.

GPT-5-mini-class AI on a normal AMD graphics card and 16 GB of RAM.
Free, private, offline. No cloud, no subscription, no ROCm.

[![License: MIT](https://img.shields.io/badge/license-MIT-22d3ee)](LICENSE)
![Linux](https://img.shields.io/badge/platform-Linux-22d3ee)
![AMD GPUs via Vulkan](https://img.shields.io/badge/GPU-AMD%20%C2%B7%20Vulkan-f97316)
![Model](https://img.shields.io/badge/model-Qwen3.5%2035B--A3B-f97316)

```bash
curl -fsSL https://raw.githubusercontent.com/MMennace/quietmoe/main/install.sh | bash
```

by [@MMennace](https://github.com/MMennace)

</div>

---

## Why you'll like it

* 🧠 **A big model on normal hardware.** Qwen3.5-35B-A3B scores close to GPT-5-mini, and runs at
  **~11 tokens/s** on an RX 6600 XT with 16 GB RAM.
* 🖥️ **Your PC stays usable.** Under memory pressure, QuietMoE pushed **5.6× less of your apps
  to swap** than running without limits, at about the same AI speed.
* 🔒 **100% local.** Your chats, code and documents never leave your computer. Works offline.
* 🎮 **Made for the cards people actually have.** AMD Radeon through Vulkan, including cards
  AMD's ROCm doesn't support. No CUDA, no ROCm.
* ⚡ **One-command install** on Ubuntu, Debian, Kali, Mint, Fedora, Arch and openSUSE. It asks
  two questions the first time, then just runs.

## What you can do with it

* chat and ask questions, in a ChatGPT-style page in your browser
* write, explain and debug code
* summarize and explain long documents, PDFs and school projects (up to ~24,000 words)
* plug it into other tools through its OpenAI-compatible API

## What you need

| | Minimum | Recommended |
|---|---|---|
| **GPU** | AMD Radeon with 6 GB VRAM (RX 400 series or newer) | 8 GB or more |
| **RAM** | 16 GB | 16 GB or more |
| **Disk** | SSD with ~25 GB free | NVMe SSD |
| **OS** | Linux | |

QuietMoE measures your card and adjusts automatically. More VRAM = more of the model on the GPU =
faster. 4 GB cards work too, but slowly (a smaller model fits them better). Tested on an
RX 6600 XT. Other cards welcome: [share your benchmark](#benchmark-speed-and-smoothness)!

## Results on a real gaming PC

**RX 6600 XT 8 GB · Ryzen 7 5700G · 16 GB RAM · NVMe SSD · Qwen3.5-35B-A3B (Q4_K_M, 20.5 GB)**

Idle PC (`quietmoe bench --compare`):

| | No limits | QuietMoE `light` |
|---|---|---|
| AI speed | 9.7–15.5 tok/s (varies per run) | 11.2 tok/s |
| Apps pushed to swap | 591 MB | 132 MB |
| VRAM used | 6.7 GB | 5.7 GB |

Busy PC (`quietmoe bench --compare --stress`, a simulated app using 6.9 GB RAM + a 1 GB file):

| | No limits | QuietMoE `light` | |
|---|---|---|---|
| AI speed | 6.0 tok/s | 5.7 tok/s | about the same |
| Apps pushed to swap | 17,616 MB | 3,166 MB | **5.6× less** |
| App file reload (average) | 621 ms | 230 ms | **2.7× faster** |
| App's worst wait for its memory | 7.9 s | 1.8 s | **4.4× shorter** |

In short: when the PC is idle, QuietMoE runs about as fast as no limits. When your apps need
the RAM, the AI runs at about the same speed, but your apps stay responsive instead of being
pushed out to disk.

<sub>`--compare` always runs "no limits" first, with a colder cache, so its speed varies more
between runs. Numbers from v1.6.0.</sub>

## Quick start (Linux)

One command does everything: it detects your distro, installs the needed packages, installs
QuietMoE, builds the engine, sets up the RAM limit and offers to download a model.

```bash
curl -fsSL https://raw.githubusercontent.com/MMennace/quietmoe/main/install.sh | bash
```

Or from a downloaded copy:

```bash
git clone https://github.com/MMennace/quietmoe && cd quietmoe
./install.sh
```

Supported: **Ubuntu / Debian / Mint / Pop!_OS**, **Fedora / Nobara**, **Arch / Manjaro /
EndeavourOS / CachyOS** and **openSUSE**. Installer options: `--yes` (no questions),
`--no-model`, `--model NAME`, `--dry-run` (show what it would do).

Then:

```bash
quietmoe run        # chat UI at http://127.0.0.1:8080
```

The first time, it asks two quick questions and remembers your answers:

```
  What will you mainly use the AI for?
    1) Chatting and quick questions                       -> 8K tokens  (~6,000 words)
    2) Coding (longer files and conversations)            -> 16K tokens (~12,000 words)
    3) Long documents: PDFs, school projects, reports     -> 32K tokens (~24,000 words)
    4) A bit of everything (mixed)                        -> 24K tokens (~18,000 words)

  Will you use your PC for other things while the AI runs?
    1) Yes, I'll use my PC normally        -> light
    2) Only light stuff                    -> balanced
    3) No, give the AI everything          -> max
```

Change your answers anytime with `quietmoe run --setup`.

<details>
<summary>Manual install (any distro)</summary>

1. Install: git, cmake, a C++ compiler, Vulkan headers + loader, SPIRV-Headers, `glslc` (shaderc), the Mesa
   RADV Vulkan driver, vulkan-tools, Python 3.10+ and pipx.
2. `pipx install git+https://github.com/MMennace/quietmoe`
3. `quietmoe setup` builds llama.cpp with Vulkan
4. `quietmoe download qwen3.5-35b-a3b`
5. Optional RAM cap: see Troubleshooting below.

</details>

## Profiles

| Profile | RAM cache | VRAM budget | CPU | For |
|---|---|---|---|---|
| `light` (default) | starts at free RAM − 3 GB, shrinks to 2 GB when apps need memory | VRAM − 2 GB | all cores, lowest priority | using your PC normally while it runs |
| `balanced` | same, shrinks to 3 GB at most | VRAM − 1 GB | all cores, low priority | PC mostly idle |
| `max` | up to 80% of RAM | VRAM − 1 GB | all cores, normal priority | overnight jobs |

Low priority means QuietMoE uses the whole CPU while you're not using it, and steps aside the
moment you are.

```bash
quietmoe run --profile balanced --ctx 16384   # skip the questions, choose directly
quietmoe info qwen3.5-35b-a3b           # show the plan for every profile
quietmoe run qwen3.5-35b-a3b --dry-run  # print the exact command without running it
quietmoe run qwen3.5-35b-a3b -- --temp 0.7   # pass extra flags to llama-server
```

## How it works

```
            ┌──────────── GPU (VRAM) ────────────┐
 token ──►  │ shared layers + attention (always) │
            │ as many expert layers as fit       │  ◄── planned from the GGUF file
            │ ~2 GB left free for your desktop   │
            └────────────────┬───────────────────┘
                             │ other experts
            ┌──────── RAM (cache with a cap) ────┐
            │ recently used experts (page cache) │  ◄── big when idle, shrinks when
            └────────────────┬───────────────────┘      your apps need memory
                             │ cache miss
            ┌──────────── NVMe SSD ──────────────┐
            │ the full model file (memory-mapped)│
            └────────────────────────────────────┘
```

* **Planner.** QuietMoE reads the GGUF file, measures the shared weights and every expert layer,
  and works out exactly how many expert layers fit in your VRAM budget.
* **Streaming for free.** The model is memory-mapped and runs inside a systemd scope with
  `MemoryHigh=<cap>`. Linux keeps the experts you use most in RAM and drops the rest. Dropped
  experts are re-read from the SSD when needed, so RAM use stays under the cap no matter how big
  the model is.
* **Governor.** It starts with a big cache (fast while the PC is idle) and checks every 2 s:
  * if Linux keeps moving your apps' memory to swap (sustained, not small background swaps),
    or less than 2 GB is free, it shrinks the cache by 20%
  * QuietMoE's own memory never goes to swap: at the cap, Linux only drops model cache, which
    is re-read from the SSD
  * it never lets the cache grow past *total RAM − what your apps use − 3 GB*
  * after 10 s of calm it grows the cache again by 1 GB
* **Low priority.** It uses `nice`, `ionice` and cgroup CPU/IO weights, so your apps always come
  first.
* **Engine.** Inference runs on [llama.cpp](https://github.com/ggml-org/llama.cpp) with the
  Vulkan backend, which works on basically every AMD card, including ones ROCm doesn't support.

## Benchmark: speed AND smoothness

```bash
quietmoe bench qwen3.5-35b-a3b --compare --stress
```

This runs the model with no limits (`baseline`) and then with QuietMoE (`light`), and measures:

* **tokens per second**
* **lag**: a normal-priority thread (standing in for your browser or game) wakes every 10 ms,
  and we record how late it wakes up. Higher lag means more stutter.
* the lowest free RAM, QuietMoE's RAM peak, VRAM peak, and Linux memory/disk pressure
* how much of your apps' memory Linux pushed to swap (disk) during the test
* with `--stress`: a simulated heavy app (half your free RAM + a 1 GB working file, like a
  browser's cache) runs the whole time. We measure how long it takes to reload its file and
  touch its memory. If the AI pushes the app's data out of RAM, these times jump.

Results are saved as JSON in `~/.local/share/quietmoe/bench/`.

Ran it on your PC? Open an issue with your `bench-*.json` and I'll add it to the table.

## Models

| Short name | Model | Notes |
|---|---|---|
| `qwen3.5-35b-a3b` | Qwen3.5 35B-A3B (Q4_K_M) | recommended: 35B total, 3B active |
| `qwen3-30b-a3b` | Qwen3 30B-A3B Instruct 2507 | older, very well tested |
| `gpt-oss-20b` | OpenAI gpt-oss 20B | smaller MoE |

Any GGUF works: `quietmoe download <hf-user>/<repo> --quant Q4_K_M`, or `quietmoe run /path/to/model.gguf`.

## Troubleshooting

**"systemd user memory limits unavailable"**: your system doesn't let normal users cap their own
memory. Enable it once:

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nDelegate=cpu cpuset io memory pids\n' | sudo tee /etc/systemd/system/user@.service.d/delegate.conf
sudo systemctl daemon-reload
# then log out and back in
```

**Vulkan not detected**: install the Mesa RADV driver and tools
(`mesa-vulkan-drivers vulkan-tools` on Ubuntu/Debian, `vulkan-radeon vulkan-tools` on Arch), then
check with `vulkaninfo --summary`.

**Slow on first answers**: that's normal. The first tokens read experts from the SSD. It speeds
up as the most-used experts get cached.

**Model on a hard drive**: streaming needs an SSD. NVMe is best.

## Roadmap

- [x] GGUF planner: exact expert-layer placement for your VRAM
- [x] RAM cap + streaming through the page cache
- [x] Pressure-aware governor
- [x] Speed + responsiveness benchmark
- [ ] Smart prefetch: tell Linux which experts will be needed next (`madvise`)
- [ ] Pin the hottest experts, measured from real usage
- [ ] Speculative decoding with a small draft model (or the model's MTP head)
- [ ] Pause and resume when a game starts
- [ ] Windows support

## Credits

* [llama.cpp](https://github.com/ggml-org/llama.cpp) does the actual inference.
* [AirLLM](https://github.com/lyogavin/airllm) inspired the idea of running models bigger than
  your memory.
* [Unsloth](https://huggingface.co/unsloth) provides the GGUF quantizations.

## License

MIT © 2026 MMennace
