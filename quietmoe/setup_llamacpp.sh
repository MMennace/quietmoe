#!/usr/bin/env bash
# Build llama.cpp with the Vulkan backend (works on basically every AMD GPU, no ROCm needed).
# Usage: setup_llamacpp.sh <install-dir>
set -euo pipefail

DIR="${1:-$HOME/.local/share/quietmoe/llama.cpp}"

need() { command -v "$1" >/dev/null 2>&1; }

missing=()
for tool in git cmake make; do need "$tool" || missing+=("$tool"); done
need c++ || need g++ || need clang++ || missing+=("c++ compiler")
need glslc || missing+=("glslc (shader compiler)")
if ! { [ -f /usr/include/vulkan/vulkan.h ] || pkg-config --exists vulkan 2>/dev/null; }; then
  missing+=("vulkan headers")
fi

# Newer llama.cpp needs the SPIR-V headers to build the Vulkan backend.
if ! { [ -f /usr/include/spirv/unified1/spirv.hpp ] || compgen -G "/usr/share/cmake*/SPIRV-Headers*" >/dev/null \
       || compgen -G "/usr/lib*/cmake/SPIRV-Headers*" >/dev/null \
       || compgen -G "/usr/lib/*/cmake/SPIRV-Headers*" >/dev/null; }; then
  missing+=("spirv-headers")
fi

if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing: ${missing[*]}"
  echo
  echo "Install them with one of these, then run 'quietmoe setup' again:"
  echo "  Ubuntu/Debian/Mint: sudo apt install git cmake build-essential libvulkan-dev glslc spirv-headers mesa-vulkan-drivers vulkan-tools"
  echo "  Fedora:             sudo dnf install git cmake gcc-c++ vulkan-headers vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers vulkan-tools"
  echo "  openSUSE:           sudo zypper install git cmake gcc-c++ vulkan-devel spirv-headers shaderc libvulkan_radeon vulkan-tools"
  echo "  Arch/Manjaro:       sudo pacman -S git cmake base-devel vulkan-headers vulkan-icd-loader spirv-headers shaderc vulkan-radeon vulkan-tools"
  exit 1
fi

if [ -d "$DIR/.git" ]; then
  echo "Updating llama.cpp in $DIR"
  git -C "$DIR" pull -q --ff-only
else
  echo "Cloning llama.cpp into $DIR"
  mkdir -p "$(dirname "$DIR")"
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$DIR"
fi

cd "$DIR"
LOG="$DIR/build.log"
: > "$LOG"

# Show the last lines of the log and stop (only used when something really fails).
die_with_log() {
  echo
  echo "Build failed. Last lines of the log ($LOG):"
  echo "------------------------------------------------------------"
  tail -n 40 "$LOG"
  echo "------------------------------------------------------------"
  exit 1
}

echo "Configuring ..."
# GGML_CCACHE=OFF: ccache is optional, skip its "not found" warning.
# -Wno-dev and -w: hide developer and compiler warnings (they are harmless; errors still stop the build).
cmake -B build -Wno-dev -DGGML_VULKAN=ON -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DCMAKE_C_FLAGS="-w" -DCMAKE_CXX_FLAGS="-w" >> "$LOG" 2>&1 || die_with_log

JOBS="${QUIETMOE_JOBS:-$(nproc)}"
echo "Compiling with $JOBS parallel jobs at low priority (your PC stays usable)."
echo "Full output is saved to $LOG"

set +e
cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli llama-bench 2>&1 \
  | tee -a "$LOG" \
  | grep --line-buffered -oE '^\[ *[0-9]+%\]' \
  | while read -r pct; do printf '\r  Building %s ' "$pct"; done
status=${PIPESTATUS[0]}
set -e
echo
[ "$status" -eq 0 ] || die_with_log

echo
echo "Done. Binaries are in $DIR/build/bin"
