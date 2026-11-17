"""Read a GGUF model and measure how big the expert vs. shared weights are.

This is what lets QuietMoE decide exactly how many expert layers fit on the GPU.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_\w*_exps\.")
ATTN_K_RE = re.compile(r"^blk\.(\d+)\.attn_k\.")

GB = 1024**3


@dataclass
class ModelInfo:
    path: str
    arch: str
    n_layers: int
    total_bytes: int
    shared_bytes: int                      # everything that is NOT a routed expert
    expert_bytes_per_layer: dict[int, int] = field(default_factory=dict)
    n_experts: int = 0
    n_experts_used: int = 0
    kv_bytes_per_token: int = 0            # f16 KV cache, only layers with full attention
    max_ctx: int = 0                       # longest context the model was trained for
    attn_layers: int = 0

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_bytes_per_layer)

    @property
    def expert_bytes(self) -> int:
        return sum(self.expert_bytes_per_layer.values())

    @property
    def active_bytes_per_token(self) -> int:
        """Rough bytes touched per token: shared weights + the chosen experts."""
        if not self.is_moe or not self.n_experts:
            return self.total_bytes
        return self.shared_bytes + int(self.expert_bytes * self.n_experts_used / self.n_experts)


def split_files(path: str) -> list[str]:
    """Return all parts of a split GGUF (model-00001-of-00003.gguf) or just [path]."""
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    parts = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))
    return parts or [path]


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:
        return default


def inspect(path: str) -> ModelInfo:
    from gguf import GGUFReader  # imported lazily: only needed here

    parts = split_files(path)
    first = GGUFReader(parts[0])
    arch = _field(first, "general.architecture", "unknown")
    n_layers = int(_field(first, f"{arch}.block_count", 0) or 0)
    n_experts = int(_field(first, f"{arch}.expert_count", 0) or 0)
    n_used = int(_field(first, f"{arch}.expert_used_count", 0) or 0)

    kv_heads = _field(first, f"{arch}.attention.head_count_kv", 0)
    heads = _field(first, f"{arch}.attention.head_count", 0)
    if isinstance(kv_heads, list):
        kv_heads = max(kv_heads) if kv_heads else 0
    if isinstance(heads, list):
        heads = max(heads) if heads else 0
    emb = int(_field(first, f"{arch}.embedding_length", 0) or 0)
    key_len = int(_field(first, f"{arch}.attention.key_length", 0) or 0)
    if not key_len and heads:
        key_len = emb // int(heads)
    val_len = int(_field(first, f"{arch}.attention.value_length", 0) or key_len)
    max_ctx = int(_field(first, f"{arch}.context_length", 0) or 0)

    total = shared = 0
    per_layer: dict[int, int] = {}
    attn_layers: set[int] = set()
    for i, part in enumerate(parts):
        reader = first if i == 0 else GGUFReader(part)
        for t in reader.tensors:
            nbytes = int(t.n_bytes)
            total += nbytes
            a = ATTN_K_RE.match(t.name)
            if a:
                attn_layers.add(int(a.group(1)))
            m = EXPERT_RE.match(t.name)
            if m:
                layer = int(m.group(1))
                per_layer[layer] = per_layer.get(layer, 0) + nbytes
            else:
                shared += nbytes

    n_layers = n_layers or (max(per_layer) + 1 if per_layer else 0)
    # Hybrid models (like Qwen3.5) only have full attention in some layers; only those
    # layers need a KV cache. If we can't tell, assume every layer does (safe upper bound).
    kv_layers = len(attn_layers) or n_layers
    kv_per_token = kv_layers * int(kv_heads or 0) * (key_len + val_len) * 2

    return ModelInfo(
        path=os.path.abspath(path),
        arch=arch,
        n_layers=n_layers,
        total_bytes=total,
        shared_bytes=shared,
        expert_bytes_per_layer=per_layer,
        n_experts=n_experts,
        n_experts_used=n_used,
        kv_bytes_per_token=kv_per_token,
        max_ctx=max_ctx,
        attn_layers=kv_layers,
    )


def describe(info: ModelInfo) -> str:
    lines = [
        f"  Model:   {os.path.basename(info.path)}  ({info.arch}, {info.n_layers} layers)",
        f"  Size:    {info.total_bytes / GB:.1f} GB total",
    ]
    if info.is_moe:
        lines += [
            f"  Shared:  {info.shared_bytes / GB:.2f} GB  (used by every token, lives on the GPU)",
            f"  Experts: {info.expert_bytes / GB:.1f} GB  ({info.n_experts} per layer, "
            f"{info.n_experts_used} used per token)",
            f"  Per token it needs about {info.active_bytes_per_token / GB:.1f} GB of weights",
        ]
    else:
        lines.append("  Not a MoE model: QuietMoE works best with MoE models.")
    return "\n".join(lines)
