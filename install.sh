#!/usr/bin/env bash
# One-command installer for vox-terminal on macOS.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/jad/vox-terminal/main/install.sh | bash
#
# What it does:
#   1. Installs portaudio via Homebrew (required for microphone access)
#   2. Installs vox-terminal via pipx (isolated Python env)
#   3. Prints next steps
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

info()  { echo -e "${GREEN}==> $*${RESET}"; }
warn()  { echo -e "${RED}==> $*${RESET}"; }
dim()   { echo -e "${DIM}    $*${RESET}"; }

# --- Prerequisites -----------------------------------------------------------

if [[ "$(uname)" != "Darwin" ]]; then
    warn "This installer is macOS-only. On Linux, install manually:"
    dim "pip install vox-terminal"
    exit 1
fi

if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Install it first:"
    dim 'https://brew.sh'
    exit 1
fi

# --- Install portaudio -------------------------------------------------------

if ! brew list portaudio &>/dev/null; then
    info "Installing portaudio (required for microphone access)..."
    brew install portaudio
else
    dim "portaudio already installed."
fi

# --- Install vox-terminal -----------------------------------------------------

if command -v pipx &>/dev/null; then
    info "Installing vox-terminal via pipx..."
    pipx install vox-terminal 2>/dev/null || pipx upgrade vox-terminal
elif command -v uvx &>/dev/null; then
    info "Installing vox-terminal via uv..."
    uv tool install vox-terminal 2>/dev/null || uv tool upgrade vox-terminal
else
    info "Installing pipx..."
    brew install pipx
    pipx ensurepath
    info "Installing vox-terminal via pipx..."
    pipx install vox-terminal
fi

# --- Done ---------------------------------------------------------------------

echo ""
info "vox-terminal installed successfully!"
echo ""
echo "  Next steps:"
echo ""
echo '  1. Set your API key:'
echo '     export VOX_TERMINAL_LLM__API_KEY="your-anthropic-key"'
echo ""
echo '  2. Start from any project:'
echo '     cd /path/to/your/project'
echo '     vox-terminal start .'
echo ""
dim "Add the API key export to ~/.zshrc to persist across sessions."
