# Product Requirements Document: VoxCursor

**Terminal Voice Assistant for Cursor IDE**

| Field          | Detail                              |
|----------------|-------------------------------------|
| Author         | Alex                                |
| Date           | March 3, 2026                       |
| Version        | 1.0 (Draft)                         |
| Status         | Proposal                            |

---

## 1. Overview

VoxCursor is a terminal-based voice assistant that lets developers ask questions about their active project and receive spoken answers — all without leaving Cursor IDE. It combines speech-to-text, project context gathering, LLM-powered reasoning, and text-to-speech into a single CLI tool that can optionally be exposed as an MCP server for deeper Cursor integration.

### 1.1 Problem Statement

Developers working in Cursor often need quick answers about their project — status updates, architectural decisions, tech stack details, dependency info — but must context-switch to read docs, grep files, or type queries. There is no native voice interface for querying project state hands-free.

### 1.2 Proposed Solution

A Python CLI tool that:

1. Captures voice input from the developer's microphone
2. Transcribes it to text using Whisper
3. Automatically gathers relevant project context (git state, directory structure, config files, etc.)
4. Sends the question + context to an LLM (Claude API)
5. Speaks the response back using TTS
6. Optionally runs as an MCP server for native Cursor integration

### 1.3 Target Users

- Developers using Cursor IDE who want hands-free project queries
- Teams who want quick verbal status checks during development sessions
- Developers with accessibility needs who benefit from voice interaction

### 1.4 Success Metrics

- End-to-end latency under 5 seconds for simple project questions (local STT/TTS)
- Accurate transcription rate above 95% for common developer terminology
- User can ask and receive an answer without touching the keyboard
- MCP server discoverable and usable within Cursor's agent

---

## 2. Systems Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   VoxCursor CLI                      │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Audio    │──▶│   STT    │──▶│  Context       │  │
│  │  Capture  │   │ (Whisper)│   │  Assembler     │  │
│  └──────────┘   └──────────┘   └───────┬────────┘  │
│                                         │            │
│                                         ▼            │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Audio    │◀──│   TTS    │◀──│  LLM Client    │  │
│  │  Playback │   │ (Engine) │   │  (Claude API)  │  │
│  └──────────┘   └──────────┘   └────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           MCP Server (optional)               │   │
│  │   Exposes tools to Cursor's agent             │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.2 Module Breakdown

| Module               | Responsibility                                       | Key Dependencies             |
|----------------------|------------------------------------------------------|------------------------------|
| `audio_capture`      | Record mic input (push-to-talk or wake-word)         | `sounddevice`, `numpy`       |
| `stt_engine`         | Transcribe audio to text                             | `faster-whisper` or OpenAI   |
| `context_assembler`  | Gather project state (git, tree, configs)            | `subprocess`, `pathlib`      |
| `llm_client`         | Send question + context to Claude, receive answer    | `anthropic` SDK              |
| `tts_engine`         | Convert answer text to speech and play it            | `piper-tts`, `say`, or OpenAI|
| `mcp_server`         | Expose voice tools via MCP protocol                  | `mcp` SDK (Python)           |
| `config`             | Manage user preferences (API keys, TTS engine, etc.) | `pydantic-settings`, TOML    |
| `cli`                | Entry point, argument parsing, interactive loop      | `click` or `typer`           |

### 2.3 Design Principles

- **Modularity**: Each stage of the pipeline (STT, context, LLM, TTS) is behind an abstract interface so engines can be swapped without touching other code.
- **Streaming-first**: Where possible, stream LLM output into TTS to reduce perceived latency. The user should start hearing the answer before the LLM finishes generating.
- **Offline-capable STT**: Default to local Whisper so voice input works without network calls. Only the LLM step requires internet.
- **Fail gracefully**: If TTS fails, print the text answer. If STT fails, fall back to typed input. Never crash the session.

### 2.4 Data Flow

```
User speaks ──▶ WAV buffer (16kHz, mono)
    ──▶ Whisper transcription (text)
    ──▶ Context Assembler collects:
         • git status, branch, recent commits
         • directory tree (depth-limited)
         • package.json / pyproject.toml / Cargo.toml
         • README summary (first 500 chars)
         • .cursor/ config if present
    ──▶ Prompt constructed: system prompt + context + user question
    ──▶ Claude API call (streaming)
    ──▶ TTS engine converts chunks to audio
    ──▶ Audio playback to speakers
```

### 2.5 Configuration Schema

```toml
# ~/.voxcursor/config.toml

[general]
project_root = "."            # auto-detected or explicit
verbose = false

[stt]
engine = "faster-whisper"     # "faster-whisper" | "openai"
model_size = "base.en"        # tiny, base, small, medium, large
language = "en"

[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
max_tokens = 1024
temperature = 0.3

[tts]
engine = "say"                # "say" (macOS) | "piper" | "openai"
voice = "Samantha"            # engine-specific
speed = 1.1

[mcp]
enabled = false
port = 8765
```

---

## 3. Functional Requirements

### 3.1 Core Features (MVP)

| ID    | Feature                        | Description                                                                 | Priority |
|-------|--------------------------------|-----------------------------------------------------------------------------|----------|
| F-01  | Push-to-talk recording         | Press Enter to start/stop recording from the terminal                       | P0       |
| F-02  | Speech-to-text transcription   | Transcribe recorded audio using local Whisper                               | P0       |
| F-03  | Project context gathering      | Auto-collect git status, tree, configs before each query                    | P0       |
| F-04  | LLM question answering         | Send transcribed question + context to Claude API                           | P0       |
| F-05  | Text-to-speech response        | Speak the LLM answer aloud via configurable TTS                            | P0       |
| F-06  | Typed input fallback           | Accept typed questions when mic is unavailable                              | P0       |
| F-07  | Configurable engines           | Swap STT/TTS/LLM providers via config file                                 | P1       |

### 3.2 Extended Features (Post-MVP)

| ID    | Feature                        | Description                                                                 | Priority |
|-------|--------------------------------|-----------------------------------------------------------------------------|----------|
| F-08  | MCP server mode                | Expose `ask_project_question` and `get_project_status` as MCP tools         | P1       |
| F-09  | Streaming TTS                  | Begin speaking while LLM is still generating                                | P1       |
| F-10  | Conversation memory            | Maintain short-term context across multiple questions in a session           | P2       |
| F-11  | Custom context plugins         | Let users define additional context sources (e.g., Jira, CI status)         | P2       |
| F-12  | Wake-word activation           | "Hey Vox" hands-free activation instead of push-to-talk                     | P3       |

---

## 4. Non-Functional Requirements

| Requirement      | Target                                                                |
|------------------|-----------------------------------------------------------------------|
| Latency (local)  | < 5s end-to-end with local STT + macOS `say`                         |
| Latency (cloud)  | < 8s end-to-end with OpenAI STT + OpenAI TTS                         |
| Memory usage     | < 500MB RSS with `base.en` Whisper model loaded                      |
| Compatibility    | macOS 13+, Ubuntu 22.04+, Windows 11 (WSL2)                          |
| Python version   | 3.10+                                                                 |
| Security         | API keys stored in env vars or OS keychain, never in config files     |

---

## 5. Test-Driven Development Strategy

### 5.1 Testing Philosophy

All modules are developed test-first. Write the test, watch it fail, implement the minimum code to pass, then refactor. Tests are not an afterthought — they are the specification.

### 5.2 Test Layers

**Unit Tests** — isolated, fast, no I/O

- `test_context_assembler.py`: Mock `subprocess` calls, verify correct parsing of git output, tree output, and config file extraction.
- `test_llm_client.py`: Mock the Anthropic SDK. Verify prompt construction, token limits, and error handling (rate limits, timeouts).
- `test_config.py`: Validate TOML parsing, defaults, and schema validation for all config fields.
- `test_tts_engine.py`: Mock system calls. Verify correct command construction for each TTS backend.
- `test_stt_engine.py`: Mock Whisper model. Verify audio preprocessing (sample rate, channels) and transcription output format.

**Integration Tests** — real I/O, containerized where possible

- Record a known audio sample, run through STT, verify transcription accuracy.
- Given a known git repo fixture, verify context assembler output matches expected structure.
- End-to-end pipeline test with a fixture audio file: STT → context → mock LLM → TTS file generation.

**Contract Tests** — for the MCP server

- Verify MCP tool schemas match the protocol specification.
- Test that Cursor can discover and call tools via the MCP handshake.
- Validate JSON-RPC request/response format for each exposed tool.

### 5.3 Test Tooling

| Tool             | Purpose                                          |
|------------------|--------------------------------------------------|
| `pytest`         | Test runner and assertion library                 |
| `pytest-mock`    | Mocking subprocess, APIs, and hardware access     |
| `pytest-cov`     | Coverage reporting (target: 85%+ on core modules) |
| `pytest-asyncio` | Async tests for streaming LLM and MCP server      |
| `hypothesis`     | Property-based tests for config parsing edge cases |

### 5.4 Coverage Requirements

- Core modules (`context_assembler`, `llm_client`, `config`): 90%+ line coverage
- Engine wrappers (`stt_engine`, `tts_engine`): 80%+ (hardware-dependent code is mocked)
- CLI layer: 70%+ (primarily integration-tested)
- MCP server: 85%+ (contract tests are critical)

---

## 6. Source Control Best Practices

### 6.1 Repository Structure

```
voxcursor/
├── src/
│   └── voxcursor/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── audio_capture.py
│       ├── stt/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract STT interface
│       │   ├── whisper_local.py
│       │   └── openai_stt.py
│       ├── context/
│       │   ├── __init__.py
│       │   ├── assembler.py
│       │   └── sources/         # git.py, tree.py, configs.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract LLM interface
│       │   └── claude.py
│       ├── tts/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract TTS interface
│       │   ├── macos_say.py
│       │   ├── piper.py
│       │   └── openai_tts.py
│       └── mcp_server.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contracts/
│   ├── fixtures/               # sample audio, git repos, configs
│   └── conftest.py
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── .gitignore
├── .pre-commit-config.yaml
└── .github/
    └── workflows/
        ├── ci.yml
        └── release.yml
```

### 6.2 Branching Strategy

Use **GitHub Flow** (simple, suited for a single-product repo):

- `main` — always deployable, protected branch
- `feature/<description>` — short-lived feature branches (e.g., `feature/streaming-tts`)
- `fix/<description>` — bug fix branches
- `chore/<description>` — tooling, CI, dependency updates

Rules:
- No direct pushes to `main`. All changes go through pull requests.
- PRs require at least one approval and passing CI before merge.
- Squash-merge to `main` to keep history linear and readable.
- Delete branches after merge.

### 6.3 Commit Convention

Follow **Conventional Commits** for automated changelogs and semantic versioning:

```
feat(stt): add faster-whisper local transcription engine
fix(context): handle repos with no commits gracefully
test(llm): add contract tests for prompt token limits
docs: add setup instructions for macOS TTS
chore(ci): add Python 3.12 to test matrix
```

### 6.4 Pre-Commit Hooks

Enforced via `.pre-commit-config.yaml`:

- `ruff` — linting and formatting (replaces flake8 + black + isort)
- `mypy` — static type checking (strict mode on core modules)
- `pytest` — run unit tests (fast subset only, full suite in CI)
- `detect-secrets` — prevent accidental API key commits
- `conventional-pre-commit` — enforce commit message format

### 6.5 CI/CD Pipeline

```yaml
# Triggered on: push to any branch, PR to main
steps:
  - lint (ruff check + ruff format --check)
  - typecheck (mypy --strict src/)
  - test (pytest tests/unit/ --cov)
  - integration test (pytest tests/integration/ — on PR to main only)
  - build (pyproject.toml → wheel)
  - release (on tag push: publish to PyPI, generate changelog)
```

### 6.6 Versioning

Follow **Semantic Versioning** (SemVer):

- `0.x.y` during initial development (breaking changes allowed in minor bumps)
- `1.0.0` when MCP server integration is stable and tested with Cursor
- Automated via `python-semantic-release` reading conventional commits

---

## 7. API & Interface Design

### 7.1 CLI Interface

```bash
# Interactive voice loop (default)
voxcursor

# Single question via voice
voxcursor ask

# Single question via text (no mic needed)
voxcursor ask --text "What's the current tech stack?"

# Start as MCP server
voxcursor serve --port 8765

# Show project context that would be sent
voxcursor context --preview
```

### 7.2 MCP Tool Definitions

```json
{
  "tools": [
    {
      "name": "ask_project_question",
      "description": "Ask a natural language question about the current project",
      "inputSchema": {
        "type": "object",
        "properties": {
          "question": { "type": "string" },
          "include_git": { "type": "boolean", "default": true },
          "include_tree": { "type": "boolean", "default": true }
        },
        "required": ["question"]
      }
    },
    {
      "name": "get_project_summary",
      "description": "Get a structured summary of the project state",
      "inputSchema": {
        "type": "object",
        "properties": {}
      }
    }
  ]
}
```

---

## 8. Security Considerations

- API keys (Anthropic, OpenAI) must be stored in environment variables or the OS keychain, never committed to the repo or written to config files.
- Audio recordings are held in memory only and never written to disk by default. An opt-in `--save-audio` flag can be added for debugging.
- The MCP server binds to `localhost` only by default. Remote access requires explicit opt-in and should use TLS.
- No project file contents are sent to the LLM — only structural metadata (file names, git status, config summaries). A `--include-file <path>` flag can opt into sending specific file contents.

---

## 9. Dependencies & Tech Stack

| Category       | Choice                   | Rationale                                          |
|----------------|--------------------------|----------------------------------------------------|
| Language       | Python 3.10+             | Best ecosystem for audio, ML, and LLM SDKs         |
| Build system   | `pyproject.toml` + `hatch` | Modern Python packaging                           |
| STT (local)    | `faster-whisper`         | CTranslate2-optimized, fast on CPU                  |
| STT (cloud)    | OpenAI Whisper API       | Higher accuracy, no local model needed              |
| LLM            | Anthropic Claude API     | Strong reasoning, large context window              |
| TTS (macOS)    | Built-in `say`           | Zero setup, low latency                             |
| TTS (cross-platform) | `piper-tts`        | Open-source, fast, decent quality                   |
| TTS (cloud)    | OpenAI TTS API           | Best voice quality                                  |
| MCP SDK        | `mcp` (Python)           | Official MCP protocol implementation                |
| CLI framework  | `typer`                  | Type-annotated CLI with minimal boilerplate         |
| Config         | `pydantic-settings`      | Validated, typed configuration with TOML support    |

---

## 10. Milestones & Delivery Plan

| Milestone     | Scope                                                 | Target     |
|---------------|-------------------------------------------------------|------------|
| M1 — MVP      | Push-to-talk → Whisper → Claude → `say` (macOS)      | Week 2     |
| M2 — Config   | TOML config, swappable engines, typed input fallback  | Week 3     |
| M3 — Polish   | Streaming TTS, error handling, cross-platform TTS     | Week 5     |
| M4 — MCP      | MCP server mode, Cursor integration testing           | Week 7     |
| M5 — Release   | PyPI package, docs, CI/CD, v1.0.0 tag               | Week 8     |

---

## 11. Open Questions

1. Should the default context window include file contents or just structural metadata? (Privacy vs. answer quality tradeoff)
2. Is wake-word detection worth the complexity for V1, or is push-to-talk sufficient?
3. Should conversation history persist across CLI sessions or remain ephemeral?
4. What is the maximum acceptable cost-per-query budget for cloud STT + LLM + TTS?

---

## 12. Appendix: Quick-Start (Post-Build)

```bash
# Install
pip install voxcursor

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run from your project root
cd ~/my-project
voxcursor

# Ask away
> 🎤 Press Enter to speak...
> "What branch am I on and what did I last commit?"
> 🔊 "You're on feature/auth-flow. Your last commit was 'add JWT
>     token refresh logic' about 2 hours ago."
```
