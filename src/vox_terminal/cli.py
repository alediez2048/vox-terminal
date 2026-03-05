"""CLI entry point — Typer app wiring all Vox-Terminal components."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator

import typer
from rich.console import Console
from rich.panel import Panel

from vox_terminal.config import VoxTerminalSettings, load_settings
from vox_terminal.context.assembler import ContextAssembler
from vox_terminal.llm import Message, create_llm_client
from vox_terminal.llm.base import LLMClient
from vox_terminal.stt import create_stt_engine
from vox_terminal.tts import create_tts_engine
from vox_terminal.tts.base import TTSEngine
from vox_terminal.tui.state import DisplayState

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="vox-terminal",
    help="Voice-powered terminal assistant.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    """Set up root logger and quiet noisy third-party loggers."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "anthropic", "openai", "elevenlabs"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _tee_stream(
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Yield chunks while also printing them to the console in real time."""
    async for chunk in stream:
        console.print(chunk, end="", highlight=False)
        yield chunk
    console.print()  # newline after streamed response


async def _ask_and_speak(
    question: str,
    llm: LLMClient,
    tts: TTSEngine,
    history: list[Message] | None = None,
    display_state: DisplayState | None = None,
) -> str:
    """Stream an LLM response for *question* and speak it via TTS.

    Returns the full response text.  Handles API / network errors gracefully
    so the caller's loop can ``continue`` without crashing.
    """
    import anthropic
    import httpx

    chunks: list[str] = []

    async def _collecting_tee(stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in stream:
            chunks.append(chunk)
            if display_state is not None:
                display_state.response_chunks.append(chunk)
            else:
                console.print(chunk, end="", highlight=False)
            yield chunk
        if display_state is None:
            console.print()
        if display_state is not None:
            display_state.phase = "speaking"

    try:
        stream = llm.stream(question, history=history or None)
        tee = _collecting_tee(stream)
        await tts.speak_streamed(tee)
    except anthropic.AuthenticationError:
        console.print("[red]Invalid API key — check VOX_TERMINAL_LLM__API_KEY.[/red]")
        return ""
    except anthropic.RateLimitError:
        console.print("[yellow]Rate limited — please wait a moment and try again.[/yellow]")
        return ""
    except anthropic.APIError as exc:
        console.print(f"[red]API error: {exc}[/red]")
        return ""
    except asyncio.CancelledError:
        # Barge-in interrupted playback — not an error
        return "".join(chunks)
    except TypeError as exc:
        if "authentication" in str(exc).lower():
            console.print(
                "[red]No API key set — run:[/red]\n"
                '  export VOX_TERMINAL_LLM__API_KEY="your-key"\n'
                "[dim]Add it to ~/.zshrc to persist across sessions.[/dim]"
            )
        else:
            console.print(f"[red]Unexpected error: {exc}[/red]")
        return ""
    except (TimeoutError, asyncio.TimeoutError):
        console.print("[yellow]Response timed out — try again.[/yellow]")
        return ""
    except httpx.HTTPError as exc:
        console.print(f"[red]Network error: {exc}[/red]")
        return ""

    return "".join(chunks)


async def _ask_once(
    question: str,
    settings: VoxTerminalSettings,
    history: list[Message] | None = None,
) -> str:
    """Ask a single question: context → LLM stream → TTS."""
    assembler = ContextAssembler(
        settings.general, settings.mcp, context_settings=settings.context,
    )
    context = assembler.assemble(
        include_git=settings.mcp.include_git,
        include_tree=settings.mcp.include_tree,
    )

    llm = create_llm_client(settings.llm, project_context=context)
    tts = create_tts_engine(settings.tts)

    console.print(f"\n[bold cyan]You:[/bold cyan] {question}")
    console.print("[bold green]Vox-Terminal:[/bold green] ", end="")

    return await _ask_and_speak(question, llm, tts, history)


async def _interactive_loop(settings: VoxTerminalSettings) -> None:
    """Main interactive voice loop: record → transcribe → ask → speak."""
    from vox_terminal.audio_capture import AudioCapture, AudioCaptureError
    from vox_terminal.tui.display import SessionDisplay
    from vox_terminal.vad import create_vad_engine

    tts = create_tts_engine(settings.tts)

    # Try to set up audio capture, fall back to text input
    use_voice = True
    stt = None
    capture = None
    try:
        vad = create_vad_engine(
            engine=settings.stt.vad_engine,
            threshold=settings.stt.vad_threshold,
            energy_threshold=settings.stt.silence_threshold,
        )
        capture = AudioCapture(sample_rate=settings.stt.sample_rate, vad=vad)
        stt = create_stt_engine(settings.stt)
    except AudioCaptureError:
        use_voice = False
        console.print(
            "[yellow]Microphone unavailable — falling back to text input.[/yellow]"
        )
    except OSError as exc:
        use_voice = False
        console.print(f"[yellow]Audio device error ({exc}) — using text input.[/yellow]")

    # -- TUI display setup ---------------------------------------------------
    display_state = DisplayState(model_name=settings.llm.model)
    display = SessionDisplay(display_state, console=console)

    console.print(
        Panel(
            "[bold]Vox-Terminal[/bold] — Voice assistant for your project\n"
            + (
                "Speak naturally — listening starts automatically.\n"
                "Press [bold]Ctrl+C[/bold] to quit."
                if use_voice
                else "Type your question, or [bold]q[/bold] to quit."
            ),
            title="Vox-Terminal",
            border_style="cyan",
        )
    )
    console.print("[dim]Warming up...[/dim]")

    # Run heavy startup tasks concurrently:
    # 1. Context assembly (git + directory tree)
    # 2. Whisper model preload (if using local whisper)
    import time as _time

    t0 = _time.monotonic()
    loop = asyncio.get_running_loop()

    async def _assemble_context() -> str:
        assembler = ContextAssembler(
            settings.general, settings.mcp, context_settings=settings.context,
        )
        return await loop.run_in_executor(
            None,
            lambda: assembler.assemble(
                include_git=settings.mcp.include_git,
                include_tree=settings.mcp.include_tree,
            ),
        )

    async def _preload_stt() -> None:
        if stt is not None and hasattr(stt, "_load_model") and not stt.model_loaded:
            model = await loop.run_in_executor(None, stt._load_model)
            stt._model = model  # store so transcribe() won't reload

    context_task = asyncio.create_task(_assemble_context())
    stt_task = asyncio.create_task(_preload_stt())

    context, _ = await asyncio.gather(context_task, stt_task)
    logger.info("Startup completed in %.2fs", _time.monotonic() - t0)

    llm = create_llm_client(settings.llm, project_context=context)

    history: list[Message] = []
    max_history = settings.llm.max_history_turns * 2  # user + assistant pairs

    # -- Conversation memory --------------------------------------------------
    store = None
    session_id = None
    project_key = str(settings.general.project_root)
    if settings.general.memory_enabled:
        from vox_terminal.memory import ConversationStore

        store = ConversationStore(db_path=settings.general.memory_db_path)
        session_id = store.create_session(project_key)
        # Load recent messages from previous sessions
        prev = store.get_recent_messages(
            project_key, max_messages=settings.general.memory_max_messages
        )
        if prev:
            for role, content in prev:
                history.append(Message(role=role, content=content))
            if len(history) > max_history:
                history = history[-max_history:]
            console.print(f"[dim]Loaded {len(prev)} messages from previous session.[/dim]")

    mic_errors = 0
    max_mic_errors = 3

    # Start the TUI display
    await display.start(capture=capture if use_voice else None)

    while True:
        try:
            if use_voice:
                display_state.phase = "listening"
                try:
                    audio = await capture.record_until_silence(
                        silence_threshold=settings.stt.silence_threshold,
                        silence_duration=settings.stt.silence_duration,
                        max_duration=settings.stt.max_record_duration,
                    )
                    mic_errors = 0  # reset on success
                except AudioCaptureError as exc:
                    mic_errors += 1
                    display.print_static(f"[red]Recording error:[/red] {exc}")
                    if mic_errors >= max_mic_errors:
                        display.print_static(
                            "[yellow]Microphone failed repeatedly — "
                            "falling back to text input.[/yellow]\n"
                            "[dim]Tip: Check System Settings → Privacy & Security "
                            "→ Microphone and ensure your terminal has access.[/dim]"
                        )
                        use_voice = False
                    else:
                        await asyncio.sleep(1)
                    continue
                except OSError as exc:
                    mic_errors += 1
                    display.print_static(f"[red]Audio device error:[/red] {exc}")
                    if mic_errors >= max_mic_errors:
                        display.print_static(
                            "[yellow]Microphone failed repeatedly — "
                            "falling back to text input.[/yellow]"
                        )
                        use_voice = False
                    else:
                        await asyncio.sleep(1)
                    continue

                if audio.size == 0:
                    display.print_static("[yellow]No audio captured.[/yellow]")
                    continue

                display_state.phase = "transcribing"
                result = await stt.transcribe(audio, settings.stt.sample_rate)
                question = result.text.strip()
                if not question:
                    display.print_static("[yellow]Could not transcribe audio.[/yellow]")
                    continue
            else:
                display.pause()
                question = input("\n> ").strip()
                display.resume()

            if question.lower() in ("q", "quit", "exit"):
                display.print_static("[dim]Goodbye![/dim]")
                break

            if not question:
                continue

            display_state.question = question
            display_state.response_chunks.clear()
            display_state.phase = "thinking"
            display_state.turn_start = _time.monotonic()

            # --- TTS playback (with optional barge-in) ---
            interrupted_audio = None

            if use_voice and settings.general.barge_in_enabled:
                # Barge-in path: monitor mic during TTS (opt-in for headphone users)
                record_task = asyncio.create_task(
                    capture.record_until_silence(
                        silence_threshold=settings.stt.silence_threshold,
                        silence_duration=settings.stt.silence_duration,
                        max_duration=settings.stt.max_record_duration,
                    )
                )

                async def _speak_with_barge_in(
                    q: str = question,
                    h: list[Message] | None = history or None,
                ) -> str:
                    """Run TTS while watching for speech on the mic."""
                    tts_task = asyncio.create_task(
                        _ask_and_speak(q, llm, tts, h, display_state=display_state)
                    )

                    speech_event = capture.speech_started
                    # Grace period: ignore mic input for 1.5s to avoid
                    # TTS speaker audio feeding back into the mic.
                    await asyncio.sleep(1.5)
                    if speech_event is not None:
                        speech_event.clear()

                    # Hysteresis: require 6 consecutive positive VAD checks
                    # (~300ms sustained speech) before triggering interrupt.
                    consecutive_hits = 0
                    required_hits = 6

                    while not tts_task.done():
                        if speech_event is not None and speech_event.is_set():
                            consecutive_hits += 1
                            speech_event.clear()
                            if consecutive_hits >= required_hits:
                                tts.interrupt()
                                break
                        else:
                            consecutive_hits = 0
                        await asyncio.sleep(0.05)

                    return await tts_task

                response_text = await _speak_with_barge_in()

                if capture.speech_started and capture.speech_started.is_set():
                    # User interrupted — finish recording their speech
                    display.print_static("[dim](interrupted)[/dim]")
                    display_state.phase = "listening"
                    display_state.response_chunks.clear()
                    try:
                        interrupted_audio = await asyncio.wait_for(
                            record_task, timeout=5.0
                        )
                    except TimeoutError:
                        logger.warning("Barge-in record timed out — cancelling")
                        capture.cancel_recording()
                        with contextlib.suppress(
                            AudioCaptureError, asyncio.CancelledError
                        ):
                            await record_task
                else:
                    # TTS finished naturally — cancel the mic monitor
                    capture.cancel_recording()
                    with contextlib.suppress(AudioCaptureError, asyncio.CancelledError):
                        try:
                            await asyncio.wait_for(record_task, timeout=5.0)
                        except TimeoutError:
                            record_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await record_task
            else:
                # Default path: no barge-in, TTS plays fully
                response_text = await _ask_and_speak(
                    question, llm, tts, history or None, display_state=display_state,
                )

            # Update history + persist
            display_state.phase = "idle"
            if response_text:
                history.append(Message(role="user", content=question))
                history.append(Message(role="assistant", content=response_text))
                if store is not None and session_id is not None:
                    store.add_message(session_id, "user", question)
                    store.add_message(session_id, "assistant", response_text)
                if len(history) > max_history:
                    history = history[-max_history:]
                display_state.history_count = len(history) // 2

            # If user interrupted via barge-in, process their speech immediately
            if interrupted_audio is not None and interrupted_audio.size > 0:
                display_state.phase = "transcribing"
                result = await stt.transcribe(
                    interrupted_audio, settings.stt.sample_rate
                )
                question = result.text.strip()
                if question and question.lower() not in ("q", "quit", "exit"):
                    display_state.question = question
                    display_state.response_chunks.clear()
                    display_state.phase = "thinking"

                    response_text = await _ask_and_speak(
                        question, llm, tts, history or None,
                        display_state=display_state,
                    )

                    display_state.phase = "idle"
                    if response_text:
                        history.append(Message(role="user", content=question))
                        history.append(
                            Message(role="assistant", content=response_text)
                        )
                        if store is not None and session_id is not None:
                            store.add_message(session_id, "user", question)
                            store.add_message(session_id, "assistant", response_text)
                        if len(history) > max_history:
                            history = history[-max_history:]
                        display_state.history_count = len(history) // 2
                elif question and question.lower() in ("q", "quit", "exit"):
                    display.print_static("[dim]Goodbye![/dim]")
                    break

        except KeyboardInterrupt:
            display.print_static("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            break

    # Clean up display + memory store
    await display.stop()
    if store is not None:
        store.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback()
def main(
    ctx: typer.Context,
    text: str | None = typer.Option(None, "--text", "-t", help="Ask a text question"),
) -> None:
    """Terminal voice assistant for Cursor IDE."""
    if ctx.invoked_subcommand is not None:
        return
    settings = load_settings()
    _configure_logging(settings.general.log_level)
    if text:
        asyncio.run(_ask_once(text, settings))
    else:
        asyncio.run(_interactive_loop(settings))


@app.command()
def ask(
    text: str = typer.Option(..., "--text", "-t", help="Question to ask"),
) -> None:
    """Ask a single question via text."""
    settings = load_settings()
    _configure_logging(settings.general.log_level)
    asyncio.run(_ask_once(text, settings))


@app.command()
def context(
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Print context"),
) -> None:
    """Show the project context that would be sent to the LLM."""
    settings = load_settings()
    assembler = ContextAssembler(
        settings.general, settings.mcp, context_settings=settings.context,
    )
    ctx = assembler.assemble(
        include_git=settings.mcp.include_git,
        include_tree=settings.mcp.include_tree,
    )
    if preview:
        if ctx:
            console.print(Panel(ctx, title="Project Context", border_style="green"))
        else:
            console.print("[yellow]No project context available.[/yellow]")


@app.command()
def serve() -> None:
    """Start the MCP server for Cursor integration."""
    try:
        from vox_terminal.mcp_server import create_mcp_server

        server = create_mcp_server()
        server.run(transport="stdio")
    except ImportError:
        console.print("[red]MCP server dependencies not available.[/red]")
        sys.exit(1)


def app_main() -> None:
    """Entry point for the CLI — called by the console_scripts entrypoint."""
    app()
