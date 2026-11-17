#!/usr/bin/env bash
# QuietMoE installer - big MoE models, light on your PC
# https://github.com/MMennace/quietmoe
#
# One-line install:
#   curl -fsSL https://raw.githubusercontent.com/MMennace/quietmoe/main/install.sh | bash
#
# Or from a cloned / unzipped folder:
#   ./install.sh
#
# Options:
#   --yes          answer "yes" to every question
#   --no-model     don't offer to download a model
#   --model NAME   model to download (default: qwen3.5-35b-a3b)
#   --dry-run      only print what would be done

set -euo pipefail

REPO_URL="https://github.com/MMennace/quietmoe"
MODEL="qwen3.5-35b-a3b"
ASSUME_YES=0
SKIP_MODEL=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) ASSUME_YES=1 ;;
    --no-model) SKIP_MODEL=1 ;;
    --model) MODEL="${2:?--model needs a name}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# ---------- pretty output ----------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  CYAN=$'\033[1;38;5;45m'; ORANGE=$'\033[1;38;5;208m'; GREEN=$'\033[1;32m'
  YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  CYAN=""; ORANGE=""; GREEN=""; YELLOW=""; RED=""; BOLD=""; DIM=""; RESET=""
fi

step()  { echo; echo "${CYAN}==>${RESET} ${BOLD}$*${RESET}"; }
ok()    { echo "  ${GREEN}✓${RESET} $*"; }
warn()  { echo "  ${YELLOW}!${RESET} $*"; }
fail()  { echo; echo "  ${RED}✗ $*${RESET}"; exit 1; }
run()   {
  echo "  ${DIM}\$ $*${RESET}"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# Ask a yes/no question. Works even when the script is piped from curl.
ask() {
  local question="$1" default="${2:-y}" answer
  if [ "$ASSUME_YES" -eq 1 ]; then return 0; fi
  if [ ! -r /dev/tty ]; then [ "$default" = "y" ]; return; fi
  local hint="[Y/n]"; [ "$default" = "n" ] && hint="[y/N]"
  printf "  %s %s " "$question" "$hint" > /dev/tty
  read -r answer < /dev/tty || answer=""
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy] ]]
}

cat <<EOF
${CYAN}
   ___        _      _   __  __       ___
  / _ \\ _   _(_) ___| |_|  \\/  | ___ | __|
 | | | | | | | |/ _ \\ __| |\\/| |/ _ \\| _|
 | |_| | |_| | |  __/ |_| |  | | (_) | |__
  \\__\\_\\\\__,_|_|\\___|\\__|_|  |_|\\___/|____|
${RESET}
  ${BOLD}Quiet${RESET}${CYAN}MoE${RESET} installer · big MoE models, light on your PC
  by ${ORANGE}@MMennace${RESET} · ${REPO_URL}
EOF
[ "$DRY_RUN" -eq 1 ] && warn "dry run: nothing will be changed"

# ---------- basic checks ----------
[ "$(uname -s)" = "Linux" ] || fail "QuietMoE needs Linux (Windows support is on the roadmap)."
[ "$(id -u)" -ne 0 ] || [ "$DRY_RUN" -eq 1 ] || fail "Don't run this as root. Run it as your normal user; it will ask for sudo when needed."
command -v sudo >/dev/null || fail "sudo is required."

# ---------- 1. detect the distro ----------
step "Detecting your system"
OS_RELEASE="${QUIETMOE_OS_RELEASE:-/etc/os-release}"   # override only used for testing
[ -r "$OS_RELEASE" ] || fail "Can't read $OS_RELEASE, unknown Linux distribution."
# shellcheck disable=SC1090
. "$OS_RELEASE"
DISTRO_NAME="${PRETTY_NAME:-$ID}"
FAMILY=""
for id in ${ID:-} ${ID_LIKE:-}; do
  case "$id" in
    ubuntu|debian|linuxmint|pop|elementary|zorin|neon|kali|raspbian) FAMILY="debian"; break ;;
    fedora|rhel|centos|rocky|almalinux|nobara|ultramarine) FAMILY="fedora"; break ;;
    arch|manjaro|endeavouros|garuda|cachyos|artix) FAMILY="arch"; break ;;
    opensuse*|suse|sles) FAMILY="suse"; break ;;
  esac
done
[ -n "$FAMILY" ] || fail "Unsupported distribution: $DISTRO_NAME. Install the packages from the README manually."
ok "$DISTRO_NAME (family: $FAMILY)"

if command -v lspci >/dev/null; then
  GPU_LINE="$(lspci | grep -Ei 'vga|3d|display' | grep -Ei 'amd|ati|radeon' | head -1 || true)"
  if [ -n "$GPU_LINE" ]; then
    ok "AMD GPU: ${GPU_LINE#*: }"
  else
    warn "No AMD GPU found. QuietMoE is built for AMD cards, but continuing anyway."
  fi
fi

# ---------- 2. system packages ----------
step "Installing system packages (needs sudo)"
case "$FAMILY" in
  debian)
    PKGS=(git cmake build-essential pkg-config libvulkan-dev glslc spirv-headers mesa-vulkan-drivers vulkan-tools
          python3 python3-pip pipx pciutils)
    run sudo apt-get update
    run sudo apt-get install -y "${PKGS[@]}"
    ;;
  fedora)
    PKGS=(git cmake gcc-c++ make pkgconf vulkan-headers vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers
          vulkan-tools python3 python3-pip pipx pciutils)
    run sudo dnf install -y "${PKGS[@]}"
    ;;
  arch)
    PKGS=(git cmake base-devel pkgconf vulkan-headers vulkan-icd-loader spirv-headers shaderc vulkan-radeon vulkan-tools
          python python-pip python-pipx pciutils)
    run sudo pacman -S --needed --noconfirm "${PKGS[@]}"
    ;;
  suse)
    PKGS=(git cmake gcc-c++ make pkg-config vulkan-devel spirv-headers shaderc libvulkan_radeon vulkan-tools
          python3 python3-pip python3-pipx pciutils)
    run sudo zypper --non-interactive install "${PKGS[@]}"
    ;;
esac
ok "system packages installed"

# ---------- 3. check Vulkan + GPU access ----------
step "Checking Vulkan"
if [ "$DRY_RUN" -eq 0 ] && command -v vulkaninfo >/dev/null; then
  VK_GPU="$(vulkaninfo --summary 2>/dev/null | grep -m1 deviceName | sed 's/.*= *//' || true)"
  if [ -n "$VK_GPU" ]; then ok "Vulkan sees: $VK_GPU"; else warn "Vulkan found no GPU. A reboot may help after installing drivers."; fi
fi
if [ -e /dev/dri/renderD128 ] && [ ! -r /dev/dri/renderD128 ]; then
  warn "Your user can't access the GPU device."
  if ask "Add $USER to the 'render' and 'video' groups? (log out/in afterwards)"; then
    run sudo usermod -aG render,video "$USER"
    NEED_RELOGIN=1
  fi
fi

# ---------- 4. install QuietMoE itself ----------
step "Installing QuietMoE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
run pipx ensurepath >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ] && grep -q 'name = "quietmoe"' "$SCRIPT_DIR/pyproject.toml"; then
  ok "using the local folder: $SCRIPT_DIR"
  run pipx install --force -e "$SCRIPT_DIR"
else
  ok "downloading from GitHub"
  run pipx install --force "git+${REPO_URL}.git"
fi
if [ "$DRY_RUN" -eq 0 ]; then
  command -v quietmoe >/dev/null || fail "quietmoe was installed but isn't on PATH. Open a new terminal and run: quietmoe info"
fi
ok "quietmoe command installed"

# ---------- 5. build llama.cpp with Vulkan ----------
step "Building the AI engine (llama.cpp + Vulkan). This takes 5-15 minutes."
run quietmoe setup
ok "engine ready"

# ---------- 6. RAM limit permission ----------
step "RAM limit (keeps your PC usable)"
UID_NUM="$(id -u)"
CTRL_FILE="/sys/fs/cgroup/user.slice/user-${UID_NUM}.slice/user@${UID_NUM}.service/cgroup.controllers"
if [ -r "$CTRL_FILE" ] && grep -qw memory "$CTRL_FILE"; then
  ok "already enabled"
elif ! command -v systemctl >/dev/null; then
  warn "systemd not found: QuietMoE will run with low priority only (no RAM cap)."
else
  echo "  QuietMoE needs permission to put a memory limit on itself."
  echo "  This is a small, safe systemd setting (user processes may limit their own memory)."
  if ask "Enable it now?"; then
    run sudo mkdir -p /etc/systemd/system/user@.service.d
    if [ "$DRY_RUN" -eq 0 ]; then
      printf '[Service]\nDelegate=cpu cpuset io memory pids\n' | sudo tee /etc/systemd/system/user@.service.d/delegate.conf >/dev/null
    else
      echo "  ${DIM}\$ write /etc/systemd/system/user@.service.d/delegate.conf${RESET}"
    fi
    run sudo systemctl daemon-reload
    ok "enabled (active after you log out and back in)"
    NEED_RELOGIN=1
  else
    warn "skipped: QuietMoE will still run, with low priority but no RAM cap"
  fi
fi

# ---------- 7. optional model download ----------
if [ "$SKIP_MODEL" -eq 0 ]; then
  step "Model"
  echo "  Recommended: $MODEL (about 21 GB download)"
  if ask "Download it now?"; then
    run quietmoe download "$MODEL"
  else
    echo "  Later:  quietmoe download $MODEL"
  fi
fi

# ---------- done ----------
echo
echo "${GREEN}  ✓ QuietMoE is installed!${RESET}"
echo
if [ "${NEED_RELOGIN:-0}" -eq 1 ]; then
  echo "  ${YELLOW}Log out and back in (or reboot) once so the new settings apply.${RESET}"
  echo
fi
echo "  Next steps:"
echo "    quietmoe info                    ${DIM}# check your hardware${RESET}"
echo "    quietmoe run                     ${DIM}# chat at http://127.0.0.1:8080 (asks 2 questions the first time)${RESET}"
echo "    quietmoe bench $MODEL --compare --stress   ${DIM}# optional: speed + smoothness test${RESET}"
echo
echo "  ${DIM}If 'quietmoe' isn't found, open a new terminal first.${RESET}"
echo
