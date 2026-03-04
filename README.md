# VoxCursor

Terminal voice assistant for IDEs Like VS Code, Cursor and Coding Agents like Claude Code. Ask questions about your project via voice and receive spoken answers — powered by Claude, Whisper, and macOS TTS.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and macOS (for `say` TTS). Set your Anthropic API key:

```bash
export VOXCURSOR_LLM__API_KEY="sk-ant-..."
```

## Quickstart

### Interactive voice mode

```bash
voxcursor
```

Press Enter to start recording, speak your question, press Enter again to stop. VoxCursor transcribes your speech, sends it to Claude with project context, and speaks the answer aloud.

### Single text question

```bash
voxcursor ask --text "What does this project do?"
```

### Preview project context

```bash
voxcursor context --preview
```

### MCP server for Cursor

```bash
voxcursor serve
```

## Configuration

VoxCursor uses environment variables with the `VOXCURSOR_` prefix and `__` as a nested delimiter:

| Variable | Default | Description |
|---|---|---|
| `VOXCURSOR_LLM__API_KEY` | *(required)* | Anthropic API key |
| `VOXCURSOR_LLM__MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `VOXCURSOR_LLM__MAX_TOKENS` | `1024` | Max response tokens |
| `VOXCURSOR_STT__ENGINE` | `whisper_local` | STT engine: `whisper_local` or `openai` |
| `VOXCURSOR_STT__WHISPER_MODEL` | `base.en` | Whisper model size |
| `VOXCURSOR_TTS__ENGINE` | `macos_say` | TTS engine: `macos_say` or `openai` |
| `VOXCURSOR_TTS__MACOS_VOICE` | `Samantha` | macOS voice name |
| `VOXCURSOR_TTS__MACOS_RATE` | `200` | Speech rate (words per minute) |
| `VOXCURSOR_MCP__TREE_DEPTH` | `3` | Directory tree depth in context |

You can also create a `voxcursor.toml` in your project root:

```toml
[general]
verbose = true

[llm]
model = "claude-sonnet-4-20250514"
max_tokens = 2048

[tts]
engine = "macos_say"
macos_voice = "Samantha"
macos_rate = 180

[stt]
engine = "whisper_local"
whisper_model = "small.en"
```

## MCP Setup for Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "voxcursor": {
      "command": "voxcursor",
      "args": ["serve"]
    }
  }
}
```

This exposes two tools to Cursor:
- `ask_project_question` — ask Claude a question with full project context
- `get_project_summary` — get the project's structure and metadata

## Architecture

```
Audio → STT (Whisper/OpenAI) → Context Assembly → Claude API → TTS (macOS say/OpenAI)
                                     ↑
                            Git + Tree + Configs
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ --cov
ruff check src/
```

## License

MIT
