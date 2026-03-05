# Vox-Terminal

Voice-powered coding assistant for IDEs like Cursor, VS Code, and CLI agents like Claude Code. Ask questions about your project by speaking and receive spoken answers — powered by Claude, Whisper, and ElevenLabs/macOS TTS.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [Engine Options](#engine-options)
- [MCP Server (Cursor Integration)](#mcp-server-cursor-integration)
- [Conversation Memory](#conversation-memory)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Development](#development)
- [Feature Tiers](#feature-tiers)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Hands-free voice loop** — auto-listen, silence detection, transcribe, LLM, TTS, repeat
- **Streaming TTS** — response starts playing while the LLM is still generating
- **Sentence-buffered playback** — LLM chunks are buffered and spoken at sentence boundaries for natural delivery
- **Barge-in support** (opt-in) — interrupt TTS mid-speech by talking, with hysteresis to prevent false triggers
- **Conversation memory** — SQLite-backed history persists across sessions per project
- **Project context injection** — git state, directory tree, config files, and README summary sent with every query
- **Swappable engines** — STT, TTS, VAD, and LLM providers are behind abstract interfaces
- **Graceful degradation** — mic fails → text input, cloud TTS fails → macOS `say` fallback, API errors → clean messages
- **MCP server** — expose tools to Cursor's agent via Model Context Protocol
- **Text sanitization** — strips markdown, commit prefixes, and ticket tags before TTS

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Vox-Terminal CLI                         │
│                                                              │
│  ┌────────────┐   ┌──────────┐   ┌───────────────────────┐  │
│  │   Audio     │──▶│   STT    │──▶│  Context Assembler    │  │
│  │  Capture    │   │(Whisper) │   │  (git + tree + cfgs)  │  │
│  │  + VAD      │   └──────────┘   └──────────┬────────────┘  │
│  └────────────┘                              │               │
│       ▲                                      ▼               │
│       │ barge-in   ┌──────────┐   ┌───────────────────────┐  │
│       │ (opt-in)   │   TTS    │◀──│  LLM Client (Claude)  │  │
│       └────────────│(ElevenLabs│   │  streaming response   │  │
│                    │ /say/OAI)│   └───────────────────────┘  │
│                    └──────────┘                               │
│                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────┐    │
│  │  Memory (SQLite)  │   │  MCP Server (optional/stdio) │    │
│  └──────────────────┘   └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User speaks → float32 audio (16kHz mono)
  → VAD detects speech, silence triggers stop
  → Whisper transcription → text
  → Context assembled: git + tree + config contents + docs + README + user files
  → System prompt + context + user question → Claude API (streaming)
  → LLM chunks buffered at sentence boundaries → sanitized → TTS engine
  → Audio playback (afplay/ffplay/say)
  → Loop returns to "Listening..."
```

### Design Principles

- **Modularity** — every pipeline stage (STT, TTS, VAD, LLM, context) is behind an abstract interface with a factory function
- **Streaming-first** — LLM output streams into TTS for lower perceived latency
- **Offline-capable STT** — default is local Whisper, no network calls for transcription
- **Fail gracefully** — never crash the session; fall back or surface clean error messages

---

## Installation

### Prerequisites

- Python 3.10+ (tested on 3.10–3.14)
- macOS (for `afplay`/`say` audio playback; Linux needs `ffplay`)
- PortAudio (`brew install portaudio` for `sounddevice`)

### Install

```bash
# Clone the repo
git clone https://github.com/alediez2048/vox-terminal.git
cd vox-terminal

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Optional: install ML-based voice activity detection
pip install -e ".[vad]"
```

### API Keys

Set at minimum your Anthropic API key:

```bash
export VOX_TERMINAL_LLM__API_KEY="sk-ant-..."
```

For ElevenLabs TTS:

```bash
export VOX_TERMINAL_TTS__ENGINE="elevenlabs"
export VOX_TERMINAL_TTS__ELEVENLABS_API_KEY="your-key"
```

For OpenAI STT or TTS:

```bash
export VOX_TERMINAL_STT__OPENAI_API_KEY="sk-..."
export VOX_TERMINAL_TTS__OPENAI_API_KEY="sk-..."
```

Add these to `~/.zshrc` or `~/.bashrc` to persist across sessions.

---

## Quick Start

### Interactive voice mode (default)

```bash
vox-terminal
```

Speak naturally — recording starts automatically. Silence detection stops recording, transcribes, sends to Claude, and speaks the answer. Loop repeats.

### Single text question

```bash
vox-terminal ask --text "What does this project do?"
```

### Preview project context

```bash
vox-terminal context --preview
```

### Start MCP server

```bash
vox-terminal serve
```

---

## CLI Commands

| Command | Description |
|---|---|
| `vox-terminal` | Interactive voice loop (default) |
| `vox-terminal --text "..."` | Ask a single text question |
| `vox-terminal ask --text "..."` | Ask a single text question (explicit subcommand) |
| `vox-terminal context [--preview/--no-preview]` | Show project context sent to the LLM |
| `vox-terminal serve` | Start the MCP server (stdio transport) |

---

## Configuration

Configuration is resolved in priority order:

1. **Environment variables** — prefix `VOX_TERMINAL_`, nested delimiter `__`
2. **TOML file** — `vox-terminal.toml` in the project root
3. **Defaults** — hardcoded in `config.py`

### All Settings

#### General

| Env Var | Type | Default | Description |
|---|---|---|---|
| `VOX_TERMINAL_GENERAL__PROJECT_ROOT` | `Path` | Current dir | Root directory for context assembly |
| `VOX_TERMINAL_GENERAL__VERBOSE` | `bool` | `false` | Verbose output |
| `VOX_TERMINAL_GENERAL__LOG_LEVEL` | `str` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `VOX_TERMINAL_GENERAL__MEMORY_ENABLED` | `bool` | `true` | Persist conversation history |
| `VOX_TERMINAL_GENERAL__MEMORY_DB_PATH` | `Path` | `~/.vox-terminal/conversations.db` | SQLite DB location |
| `VOX_TERMINAL_GENERAL__MEMORY_MAX_MESSAGES` | `int` | `40` | Max messages loaded from history |
| `VOX_TERMINAL_GENERAL__BARGE_IN_ENABLED` | `bool` | `false` | Allow voice interruption during TTS (use with headphones) |

#### Speech-to-Text (STT)

| Env Var | Type | Default | Description |
|---|---|---|---|
| `VOX_TERMINAL_STT__ENGINE` | `str` | `whisper_local` | `whisper_local` or `openai` |
| `VOX_TERMINAL_STT__WHISPER_MODEL` | `str` | `base.en` | Model size: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3` |
| `VOX_TERMINAL_STT__WHISPER_DEVICE` | `str` | `cpu` | `cpu` or `cuda` |
| `VOX_TERMINAL_STT__WHISPER_COMPUTE_TYPE` | `str` | `int8` | Compute type for faster-whisper |
| `VOX_TERMINAL_STT__SAMPLE_RATE` | `int` | `16000` | Audio sample rate (Hz) |
| `VOX_TERMINAL_STT__OPENAI_API_KEY` | `str` | `""` | OpenAI API key (for cloud STT) |
| `VOX_TERMINAL_STT__SILENCE_THRESHOLD` | `float` | `0.01` | RMS energy threshold for silence |
| `VOX_TERMINAL_STT__SILENCE_DURATION` | `float` | `1.5` | Seconds of silence before stop |
| `VOX_TERMINAL_STT__MAX_RECORD_DURATION` | `float` | `30.0` | Max recording length (seconds) |
| `VOX_TERMINAL_STT__VAD_ENGINE` | `str` | `silero` | `silero` (ML) or `energy` (RMS) |
| `VOX_TERMINAL_STT__VAD_THRESHOLD` | `float` | `0.5` | VAD confidence threshold |

#### LLM

| Env Var | Type | Default | Description |
|---|---|---|---|
| `VOX_TERMINAL_LLM__API_KEY` | `str` | `""` | Anthropic API key **(required)** |
| `VOX_TERMINAL_LLM__PROVIDER` | `str` | `claude` | LLM provider (only `claude` currently) |
| `VOX_TERMINAL_LLM__MODEL` | `str` | `claude-sonnet-4-20250514` | Claude model ID |
| `VOX_TERMINAL_LLM__MAX_TOKENS` | `int` | `1024` | Max response tokens |
| `VOX_TERMINAL_LLM__TEMPERATURE` | `float` | `0.3` | Sampling temperature |
| `VOX_TERMINAL_LLM__MAX_HISTORY_TURNS` | `int` | `10` | Conversation history depth (turns) |
| `VOX_TERMINAL_LLM__STREAM_TIMEOUT` | `float` | `60.0` | Stream response timeout (seconds) |

#### Text-to-Speech (TTS)

| Env Var | Type | Default | Description |
|---|---|---|---|
| `VOX_TERMINAL_TTS__ENGINE` | `str` | `macos_say` | `macos_say`, `openai`, `elevenlabs`, or `piper` |
| `VOX_TERMINAL_TTS__MACOS_VOICE` | `str` | `Samantha` | macOS `say` voice name |
| `VOX_TERMINAL_TTS__MACOS_RATE` | `int` | `200` | Words per minute |
| `VOX_TERMINAL_TTS__OPENAI_API_KEY` | `str` | `""` | OpenAI API key (for TTS) |
| `VOX_TERMINAL_TTS__OPENAI_VOICE` | `str` | `alloy` | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `VOX_TERMINAL_TTS__OPENAI_MODEL` | `str` | `tts-1` | `tts-1` or `tts-1-hd` |
| `VOX_TERMINAL_TTS__ELEVENLABS_API_KEY` | `str` | `""` | ElevenLabs API key |
| `VOX_TERMINAL_TTS__ELEVENLABS_VOICE_ID` | `str` | `JBFqnCBsd6RMkjVDRZzb` | ElevenLabs voice ID |
| `VOX_TERMINAL_TTS__ELEVENLABS_MODEL_ID` | `str` | `eleven_flash_v2_5` | ElevenLabs model |
| `VOX_TERMINAL_TTS__ELEVENLABS_SPEED` | `float` | `1.0` | Playback speed multiplier (via `afplay -r`) |
| `VOX_TERMINAL_TTS__ELEVENLABS_OUTPUT_FORMAT` | `str` | `mp3_44100_128` | Audio output format |

#### MCP

| Env Var | Type | Default | Description |
|---|---|---|---|
| `VOX_TERMINAL_MCP__ENABLED` | `bool` | `true` | Enable MCP server |
| `VOX_TERMINAL_MCP__TRANSPORT` | `str` | `stdio` | MCP transport protocol |
| `VOX_TERMINAL_MCP__INCLUDE_GIT` | `bool` | `true` | Include git metadata in context |
| `VOX_TERMINAL_MCP__INCLUDE_TREE` | `bool` | `true` | Include directory tree in context |
| `VOX_TERMINAL_MCP__TREE_DEPTH` | `int` | `3` | Directory tree depth |

### TOML Configuration

Create `vox-terminal.toml` in your project root:

```toml
[general]
verbose = true
barge_in_enabled = false

[stt]
engine = "whisper_local"
whisper_model = "small.en"
silence_duration = 2.5
silence_threshold = 0.015
vad_engine = "silero"

[llm]
model = "claude-sonnet-4-20250514"
max_tokens = 2048
temperature = 0.3

[tts]
engine = "elevenlabs"
elevenlabs_speed = 1.2

[mcp]
tree_depth = 3
```

---

## Engine Options

### Speech-to-Text

| Engine | Type | Latency | Quality | Dependencies |
|---|---|---|---|---|
| `whisper_local` | Local | ~1-3s | Good | `faster-whisper` (included) |
| `openai` | Cloud | ~2-4s | Excellent | `openai` SDK + API key |

**Whisper models** (local): `tiny.en` (fastest) → `base.en` (default) → `small.en` → `medium.en` → `large-v3` (most accurate). Model is lazy-loaded on first use and preloaded concurrently during startup.

### Text-to-Speech

| Engine | Type | Latency | Quality | Dependencies |
|---|---|---|---|---|
| `macos_say` | Local | Instant | Basic | macOS built-in (no API key) |
| `elevenlabs` | Cloud | ~200ms | Excellent | `elevenlabs` SDK + API key |
| `openai` | Cloud | ~1-2s | Very good | `openai` SDK + API key |
| `piper` | Local | — | — | Not yet implemented |

Cloud TTS engines (`elevenlabs`, `openai`) are automatically wrapped with a `MacOSSayTTS` fallback — if the cloud API fails, speech continues via macOS `say`.

**ElevenLabs playback modes:**
- **macOS (default):** Buffers full audio, plays via `afplay -r <speed>`
- **Linux/other:** Streams chunks to `ffplay` stdin for lower first-chunk latency

### Voice Activity Detection (VAD)

| Engine | Type | Accuracy | Dependencies |
|---|---|---|---|
| `silero` | ML (PyTorch) | High | `silero-vad`, `torch` (optional `[vad]` extra) |
| `energy` | RMS threshold | Basic | None (built-in) |

Silero is preferred but falls back to energy VAD if `torch` is not installed.

### LLM

| Provider | Models | Notes |
|---|---|---|
| `claude` | `claude-sonnet-4-20250514`, `claude-opus-4-6`, etc. | Streaming, project context injected into system prompt |

The system prompt instructs Claude to respond in plain conversational language (no markdown, no special formatting) since responses are spoken aloud via TTS.

---

## MCP Server (Cursor Integration)

Vox-Terminal can run as an MCP (Model Context Protocol) server, exposing tools to Cursor's agent.

### Setup

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "vox-terminal": {
      "command": "vox-terminal",
      "args": ["serve"]
    }
  }
}
```

### Exposed Tools

| Tool | Description |
|---|---|
| `ask_project_question(question, include_git?, include_tree?)` | Ask Claude a question with full project context |
| `get_project_summary()` | Get the assembled project context (git + tree + configs) |

---

## Conversation Memory

Conversation history is stored in a SQLite database at `~/.vox-terminal/conversations.db` (configurable via `VOX_TERMINAL_GENERAL__MEMORY_DB_PATH`).

- **Per-project isolation** — messages are scoped by `project_root`
- **Cross-session persistence** — previous session messages are loaded at startup
- **Configurable depth** — `memory_max_messages` controls how many messages are loaded (default 40)
- **WAL mode** — safe for concurrent reads

### Schema

```
sessions: id (UUID), started_at (ISO 8601), project_root
messages: id (UUID), session_id (FK), role, content, timestamp
```

Disable with `VOX_TERMINAL_GENERAL__MEMORY_ENABLED=false`.

---

## Project Structure

```
vox-terminal/
├── src/vox_terminal/
│   ├── __init__.py                  # Package root, __version__ = "0.1.0"
│   ├── cli.py                       # Typer app, interactive loop, CLI commands
│   ├── config.py                    # Pydantic-settings (env vars + TOML)
│   ├── retry.py                     # Async retry with exponential backoff
│   ├── audio_capture.py             # Sounddevice recording + VAD integration
│   ├── mcp_server.py                # FastMCP server (stdio)
│   ├── context/
│   │   ├── assembler.py             # Orchestrates all context sources
│   │   └── sources/
│   │       ├── git.py               # Branch, status, commits, remote
│   │       ├── tree.py              # Depth-limited directory tree
│   │       ├── configs.py           # Config file detection + README summary
│   │       └── files.py             # File content reading with size/safety guards
│   ├── llm/
│   │   ├── base.py                  # Abstract LLMClient, Message, LLMResponse
│   │   └── claude.py                # Anthropic Claude (streaming)
│   ├── stt/
│   │   ├── base.py                  # Abstract STTEngine, TranscriptionResult
│   │   ├── whisper_local.py         # faster-whisper (local, lazy model load)
│   │   └── openai_stt.py            # OpenAI Whisper API
│   ├── tts/
│   │   ├── base.py                  # Abstract TTSEngine, FallbackTTSEngine, sanitizer
│   │   ├── macos_say.py             # macOS `say` command
│   │   ├── openai_tts.py            # OpenAI Audio API
│   │   ├── elevenlabs_tts.py        # ElevenLabs (buffered afplay / streaming ffplay)
│   │   └── piper.py                 # Piper TTS (stub, not yet implemented)
│   ├── vad/
│   │   ├── base.py                  # Abstract VADEngine, VADResult
│   │   ├── silero.py                # Silero VAD (PyTorch)
│   │   └── energy.py                # RMS energy threshold
│   └── memory/
│       └── store.py                 # SQLite conversation store
├── tests/
│   ├── conftest.py                  # Shared fixtures (project_root, settings, clean_env)
│   ├── unit/                        # 130+ unit tests
│   │   ├── test_config.py
│   │   ├── test_cli.py
│   │   ├── test_audio_capture.py
│   │   ├── test_stt.py
│   │   ├── test_tts.py
│   │   ├── test_vad.py
│   │   ├── test_llm.py
│   │   ├── test_context.py
│   │   ├── test_memory.py
│   │   └── test_retry.py
│   ├── integration/
│   │   └── test_pipeline.py         # STT → LLM → TTS pipeline tests
│   └── contracts/
│       └── test_mcp_server.py       # MCP tool schema/contract tests
├── pyproject.toml                   # Dependencies, build config, tool settings
├── .pre-commit-config.yaml          # ruff, mypy, detect-secrets, conventional commits
├── .gitignore
├── VoxCursor_PRD.md                 # Product requirements document
└── README.md
```

---

## Module Reference

### `cli.py` — Entry Point

The Typer app wires all components together. Key functions:

- `_interactive_loop()` — main voice conversation loop. Handles recording, transcription, LLM streaming, TTS playback, and optional barge-in
- `_ask_and_speak()` — streams an LLM response and pipes it to TTS with sentence buffering
- `_ask_once()` — single-shot question with context assembly

### `audio_capture.py` — Microphone Recording

- **Push-to-talk:** `start()` / `stop()` returns a numpy array
- **Hands-free:** `record_until_silence()` listens until speech is detected, then stops after configurable silence duration
- **Barge-in:** exposes a `speech_started` event that the main loop monitors during TTS playback
- Falls back to text input after 3 consecutive mic errors

### `context/assembler.py` — Project Context

Gathers and formats context injected into the LLM system prompt:

- **Git:** branch, remote URL, working tree status, last 5 commits
- **Tree:** depth-limited directory listing (ignores `node_modules`, `.git`, `__pycache__`, etc.)
- **Configs:** detects and reads `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, etc. (excludes `.env`)
- **README:** full content (configurable, can revert to 500-char excerpt)
- **Docs:** auto-detects `CHANGELOG.md`, `CONTRIBUTING.md`, `ARCHITECTURE.md`, etc.
- **User files:** custom glob patterns via `[context] include_files` config
- **Safety:** binary extension blocklist, `.env` exclusion, per-file (50KB) and total (200KB) size caps

### `llm/claude.py` — Claude Client

- Streaming via `anthropic.AsyncAnthropic.messages.stream()`
- System prompt template injects project context and instructs plain conversational output
- Configurable timeout, temperature, max tokens, and history depth

### `tts/base.py` — TTS Base Class

- `speak_streamed()` buffers LLM chunks and flushes at sentence boundaries (`.!?\n`)
- `_sanitize_for_speech()` strips markdown, commit prefixes (`feat:`, `fix:`, etc.), and ticket tags (`(MVP-017)`)
- `FallbackTTSEngine` wraps a primary engine with automatic fallback on error

### `retry.py` — Async Retry

```python
await retry_async(fn, *args, max_attempts=3, base_delay=1.0)
```

Exponential backoff (1s → 2s → 4s). Retries `OSError`, `TimeoutError`, `ConnectionError` by default.

---

## Development

### Setup

```bash
pip install -e ".[dev,vad]"
```

### Run Tests

```bash
# All tests (140 tests)
pytest tests/ -v

# With coverage
pytest tests/ --cov=vox_terminal --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Contract tests (MCP server)
pytest tests/contracts/ -v
```

### Lint & Type Check

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

Hooks enforce: ruff (lint + format), mypy (strict), detect-secrets, and conventional commit messages.

### Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(tts): add ElevenLabs streaming support
fix(audio): handle PortAudio device errors gracefully
test(llm): add timeout contract test
docs: update configuration table
chore(ci): add Python 3.14 to test matrix
```

### Tech Stack

| Category | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Best ecosystem for audio, ML, and LLM SDKs |
| Build | `hatchling` via `pyproject.toml` | Modern Python packaging |
| CLI | `typer` + `rich` | Type-annotated CLI with rich terminal output |
| Config | `pydantic-settings` | Validated, typed config with env var + TOML |
| LLM | `anthropic` SDK | Claude streaming with large context window |
| STT (local) | `faster-whisper` | CTranslate2-optimized, fast on CPU |
| STT (cloud) | `openai` SDK | Higher accuracy, no local model |
| TTS (local) | macOS `say` | Zero setup, low latency |
| TTS (cloud) | `elevenlabs`, `openai` | High quality natural voices |
| VAD | `silero-vad` / energy | ML-based or simple RMS threshold |
| Audio | `sounddevice` + `numpy` | Cross-platform mic capture |
| MCP | `mcp[cli]` (FastMCP) | Official Model Context Protocol SDK |
| Testing | `pytest` + `pytest-asyncio` + `pytest-mock` | Async-native test suite |
| Linting | `ruff` | Fast all-in-one linter + formatter |

### Dependencies

**Core:**
```
anthropic>=0.40.0       faster-whisper>=1.1.0    numpy>=1.26.0
openai>=1.50.0          pydantic>=2.9.0          pydantic-settings>=2.6.0
sounddevice>=0.5.0      typer>=0.15.0            rich>=13.9.0
mcp[cli]>=1.0.0         elevenlabs>=1.0.0
```

**Optional (`[vad]`):**
```
silero-vad>=5.0         torch>=2.0
```

**Dev (`[dev]`):**
```
pytest>=8.0             pytest-asyncio>=0.24.0   pytest-cov>=6.0
pytest-mock>=3.14       ruff>=0.8.0              mypy>=1.13
pre-commit>=4.0         detect-secrets>=1.5
```

---

## Feature Tiers

### Base (Free)

All core functionality ships free:

| Feature | Description |
|---|---|
| Voice loop | Hands-free listen → transcribe → LLM → TTS → repeat |
| Streaming TTS | Response plays while LLM is still generating |
| Barge-in | Interrupt TTS mid-speech (opt-in, headphones recommended) |
| Conversation memory | SQLite-backed history, per-project, cross-session |
| Project context | Git state, directory tree, config files, README, doc files |
| File content reading | Auto-read configs, docs, and user-specified globs into context |
| Swappable engines | STT (Whisper/OpenAI), TTS (say/ElevenLabs/OpenAI), VAD (Silero/energy) |
| Graceful degradation | Mic → text fallback, cloud TTS → local fallback, clean error messages |
| MCP server | Expose tools to Cursor/VS Code via Model Context Protocol |
| Claude LLM | Streaming responses with project context injection |

### Premium

Power features for professional use:

| Feature | Description | Status |
|---|---|---|
| Multi-model support | Switch between Claude, GPT-5.3, Gemini, Kimi, and other LLM providers | Planned |
| Question-aware retrieval | Lightweight RAG — greps codebase for terms in your question, includes matching file excerpts in context automatically | Planned |
| Wake-word activation | "Hey Vox" hands-free trigger without keyboard interaction | Planned |
| Custom context plugins | Pull context from external sources (Jira, CI status, Slack, etc.) | Planned |

---

## Roadmap

### Completed (v0.1.0)

- [x] Hands-free voice loop with silence detection
- [x] Local Whisper + OpenAI STT engines
- [x] macOS say + OpenAI + ElevenLabs TTS engines
- [x] Silero + energy VAD engines
- [x] Claude LLM with streaming
- [x] Project context assembly (git, tree, configs, README, file contents)
- [x] Configurable file content reading (globs, size caps, binary/sensitive exclusion)
- [x] MCP server for Cursor integration
- [x] Conversation memory (SQLite, per-project)
- [x] Barge-in support (opt-in)
- [x] TTS text sanitization
- [x] Subprocess timeout hardening
- [x] Mic error fallback to text input
- [x] FallbackTTSEngine (cloud → local auto-switch)
- [x] Pre-commit hooks (ruff, mypy, detect-secrets, conventional commits)
- [x] 160+ tests (unit, integration, contract)

### Planned (Base)

- [ ] **Menubar app** — SwiftUI/Tauri wrapper with global hotkey and system tray presence
- [ ] **Push-to-talk hotkey** — global keyboard shortcut to toggle listening
- [ ] **Piper TTS** — fully local, cross-platform TTS (no API key needed)
- [ ] **CI/CD pipeline** — GitHub Actions for lint, test, build, and PyPI release
- [ ] **Cross-platform audio** — Linux/Windows playback support beyond macOS

### Planned (Premium)

- [ ] **Multi-model support** — switch between Claude, GPT-5.3, Gemini, Kimi, and other providers
- [ ] **Question-aware retrieval** — lightweight keyword-based RAG that greps the codebase for terms in your question and includes matching file excerpts in context
- [ ] **Wake-word activation** — "Hey Vox" hands-free trigger
- [ ] **Custom context plugins** — user-defined context sources (Jira, CI status, etc.)

---

## License

MIT
