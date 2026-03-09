"""CLI entry point — Typer app wiring all Vox-Terminal components."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import select
import sys
import threading
import time as _time
from collections.abc import AsyncIterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from vox_terminal.config import (
    ContextSettings,
    GeneralSettings,
    VoxTerminalSettings,
    load_settings,
)
from vox_terminal.context.assembler import ContextAssembler
from vox_terminal.llm import Message, create_llm_client
from vox_terminal.llm.base import LLMClient
from vox_terminal.observability import (
    TurnContext,
    TurnWaterfall,
    generate_turn_id,
    get_current_turn_id,
)
from vox_terminal.project_root import resolve_project_root
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


class _TurnIDFilter(logging.Filter):
    """Inject the active turn ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        turn_id = get_current_turn_id()
        record.turn_id = turn_id[:8] if turn_id else "-"
        return True


def _resolve_log_file_path(settings: GeneralSettings) -> Path:
    """Resolve the on-disk log path from settings."""
    if settings.log_file is not None:
        return settings.log_file.expanduser()
    return Path.home() / ".vox-terminal" / "vox-terminal.log"


def _configure_logging(settings: GeneralSettings) -> None:
    """Set up root logger and quiet noisy third-party loggers."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    console_handler = logging.StreamHandler()
    console_handler.addFilter(_TurnIDFilter())
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [turn:%(turn_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console_handler)

    log_file = _resolve_log_file_path(settings)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=settings.log_rotate_max_bytes,
            backupCount=settings.log_rotate_backup_count,
            encoding="utf-8",
        )
        file_handler.addFilter(_TurnIDFilter())
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [turn:%(turn_id)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
    except OSError:
        logger.warning("Unable to configure log file handler at %s", log_file)

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


def _log_turn_waterfall(
    waterfall: TurnWaterfall,
    *,
    label: str = "turn_waterfall",
) -> None:
    """Log a compact latency waterfall for the current turn."""
    snapshot = waterfall.snapshot_ms()
    if not snapshot:
        return
    fields = ", ".join(
        f"{key}={snapshot[key]:.1f}" for key in sorted(snapshot)
    )
    logger.info("%s %s", label, fields)


def _spacebar_watcher_thread(
    tts: TTSEngine,
    loop: asyncio.AbstractEventLoop,
    space_pressed: asyncio.Event,
    stop_event: threading.Event,
) -> None:
    """Background thread: wait for spacebar, then interrupt TTS via the main event loop.

    All TTS engines (especially ElevenLabs) expect ``interrupt()`` to run on
    the asyncio thread, so we use ``loop.call_soon_threadsafe`` instead of
    calling it directly from this thread.
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r and sys.stdin in r:
                    key = sys.stdin.read(1)
                    if key == " ":
                        loop.call_soon_threadsafe(tts.interrupt)
                        loop.call_soon_threadsafe(space_pressed.set)
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, OSError, AttributeError):
        pass


async def _wait_for_spacebar_interrupt(
    tts: TTSEngine,
    stop_event: threading.Event,
) -> None:
    """Run until user presses space (then interrupt TTS) or stop_event is set.
    No-op if stdin is not a TTY or on Windows.
    """
    if not sys.stdin.isatty() or sys.platform == "win32":
        await asyncio.Event().wait()  # never completes; caller will cancel us
        return
    loop = asyncio.get_running_loop()
    space_pressed = asyncio.Event()
    thread = threading.Thread(
        target=_spacebar_watcher_thread,
        args=(tts, loop, space_pressed, stop_event),
        daemon=True,
    )
    thread.start()
    try:
        await space_pressed.wait()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()


def _build_interactive_context_settings(base: ContextSettings) -> ContextSettings:
    """Build a compact context profile for conversational voice turns."""
    if not base.interactive_compact_context:
        return base.model_copy(deep=True)

    enabled_sources = (
        list(base.interactive_enabled_sources)
        if base.interactive_enabled_sources
        else list(base.enabled_sources)
    )
    if not enabled_sources:
        enabled_sources = ["project_info", "git", "tree"]

    return ContextSettings(
        include_files=[],
        enabled_sources=enabled_sources,
        max_file_size=base.max_file_size,
        max_context_chars=min(base.max_context_chars, base.interactive_max_context_chars),
        interactive_compact_context=base.interactive_compact_context,
        interactive_max_context_chars=base.interactive_max_context_chars,
        interactive_enabled_sources=list(base.interactive_enabled_sources),
        read_config_files=False,
        read_full_readme=False,
        skip_network_sources=True,
        doc_patterns=[],
    )


async def _ask_and_speak(
    question: str,
    llm: LLMClient,
    tts: TTSEngine,
    history: list[Message] | None = None,
    display_state: DisplayState | None = None,
    log_sensitive: bool = False,
    waterfall: TurnWaterfall | None = None,
) -> str:
    """Stream an LLM response for *question* and speak it via TTS.

    Returns the full response text.  Handles API / network errors gracefully
    so the caller's loop can ``continue`` without crashing.
    """
    import anthropic
    import httpx

    chunks: list[str] = []
    emitted_chars = 0
    llm_started_at = _time.monotonic()
    if waterfall is not None:
        waterfall.mark("llm_request_start_ms", at=llm_started_at)
    logger.info(
        "LLM stage started (question_chars=%d, history_messages=%d)",
        len(question),
        len(history or []),
    )

    def _on_tts_event(event_name: str, at: float) -> None:
        if waterfall is not None:
            waterfall.mark(event_name, at=at)
        if event_name == "tts_first_flush_ms":
            logger.info("TTS first flush ready")
        elif event_name == "tts_first_audio_ms":
            logger.info("TTS first audio started")
        elif event_name == "tts_end_ms":
            logger.info("TTS stage finished")

    async def _collecting_tee(stream: AsyncIterator[str]) -> AsyncIterator[str]:
        nonlocal emitted_chars
        saw_first_token = False
        async for chunk in stream:
            if not saw_first_token:
                saw_first_token = True
                token_at = _time.monotonic()
                if waterfall is not None:
                    waterfall.mark("llm_first_token_ms", at=token_at)
                logger.info(
                    "LLM first token received (elapsed_ms=%.0f)",
                    (token_at - llm_started_at) * 1000,
                )
            emitted_chars += len(chunk)
            chunks.append(chunk)
            if display_state is not None:
                if display_state.phase != "speaking":
                    display_state.phase = "speaking"
                display_state.response_chunks.append(chunk)
            else:
                console.print(chunk, end="", highlight=False)
            yield chunk
        llm_end_at = _time.monotonic()
        if waterfall is not None:
            waterfall.mark("llm_end_ms", at=llm_end_at)
        logger.info(
            "LLM stage finished (response_chars=%d, elapsed_ms=%.0f)",
            emitted_chars,
            (llm_end_at - llm_started_at) * 1000,
        )
        if display_state is None:
            console.print()

    try:
        stream = llm.stream(question, history=history or None)
        if asyncio.iscoroutine(stream):
            stream = await stream
        tee = _collecting_tee(stream)
        await tts.speak_streamed(tee, on_event=_on_tts_event)
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
        partial = "".join(chunks)
        logger.info(
            "LLM/TTS stage interrupted (response_chars=%d)",
            len(partial),
        )
        return partial
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

    response = "".join(chunks)
    if log_sensitive:
        logger.debug("LLM response preview: %s", response[:200])
    return response


async def _ask_once(
    question: str,
    settings: VoxTerminalSettings,
    history: list[Message] | None = None,
) -> str:
    """Ask a single question: context → LLM stream → TTS."""
    context_t0 = _time.monotonic()
    assembler = ContextAssembler(
        settings.general,
        settings.mcp,
        context_settings=settings.context,
    )
    context = assembler.assemble(
        include_git=settings.mcp.include_git,
        include_tree=settings.mcp.include_tree,
    )
    logger.info(
        "Context assembly finished (chars=%d, elapsed_ms=%.0f)",
        len(context),
        (_time.monotonic() - context_t0) * 1000,
    )

    llm = create_llm_client(settings.llm, project_context=context)
    tts = create_tts_engine(settings.tts)

    console.print(f"\n[bold cyan]You:[/bold cyan] {question}")
    console.print("[bold green]Vox-Terminal:[/bold green] ", end="")

    return await _ask_and_speak(
        question,
        llm,
        tts,
        history,
        log_sensitive=settings.general.log_sensitive,
    )


async def _run_diagnostics(
    settings: VoxTerminalSettings,
    *,
    no_audio: bool = False,
    text_only: bool = False,
    minimal_context: bool = False,
) -> list[tuple[str, bool, str, float]]:
    """Run a minimal diagnostics pass across core pipeline stages."""
    results: list[tuple[str, bool, str, float]] = []
    context_settings = settings.context
    if minimal_context:
        context_settings = ContextSettings(
            enabled_sources=["project_info", "git"],
            include_files=[],
            read_config_files=False,
            read_full_readme=False,
            skip_network_sources=True,
            doc_patterns=[],
            max_file_size=settings.context.max_file_size,
            max_context_chars=min(settings.context.max_context_chars, 20_000),
        )

    assembled_context = ""

    # Context
    t0 = _time.monotonic()
    try:
        assembler = ContextAssembler(
            settings.general,
            settings.mcp,
            context_settings=context_settings,
        )
        assembled_context = assembler.assemble(
            include_git=settings.mcp.include_git,
            include_tree=settings.mcp.include_tree,
        )
        results.append(
            (
                "context",
                True,
                f"{len(assembled_context)} chars assembled",
                (_time.monotonic() - t0) * 1000,
            )
        )
    except Exception as exc:
        results.append(("context", False, str(exc), (_time.monotonic() - t0) * 1000))

    # STT
    if text_only:
        results.append(("stt", True, "skipped (--text-only)", 0.0))
    else:
        t0 = _time.monotonic()
        try:
            stt = create_stt_engine(settings.stt)
            if hasattr(stt, "_load_model"):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, stt._load_model)
            results.append(("stt", True, f"engine={settings.stt.engine}", (_time.monotonic() - t0) * 1000))
        except Exception as exc:
            results.append(("stt", False, str(exc), (_time.monotonic() - t0) * 1000))

    # LLM
    t0 = _time.monotonic()
    try:
        llm = create_llm_client(settings.llm, project_context=assembled_context)
        probe = await asyncio.wait_for(
            llm.ask("Reply with exactly this word: ok"),
            timeout=min(settings.llm.stream_timeout, 20.0),
        )
        summary = probe.content.strip().replace("\n", " ")[:80]
        results.append(("llm", True, summary or "ok", (_time.monotonic() - t0) * 1000))
    except Exception as exc:
        results.append(("llm", False, str(exc), (_time.monotonic() - t0) * 1000))

    # TTS
    t0 = _time.monotonic()
    if text_only:
        results.append(("tts", True, "skipped (--text-only)", (_time.monotonic() - t0) * 1000))
    elif no_audio:
        results.append(("tts", True, "skipped (--no-audio)", (_time.monotonic() - t0) * 1000))
    else:
        try:
            tts = create_tts_engine(settings.tts)
            await tts.speak("Diagnostics complete.")
            results.append(("tts", True, f"engine={settings.tts.engine}", (_time.monotonic() - t0) * 1000))
        except Exception as exc:
            results.append(("tts", False, str(exc), (_time.monotonic() - t0) * 1000))

    return results


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
    vad = None
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
        console.print("[yellow]Microphone unavailable — falling back to text input.[/yellow]")
    except OSError as exc:
        use_voice = False
        console.print(f"[yellow]Audio device error ({exc}) — using text input.[/yellow]")

    # -- TUI display setup ---------------------------------------------------
    display_state = DisplayState(model_name=settings.llm.model)
    display = SessionDisplay(display_state, console=console)

    space_hint = (
        " Press [bold]Space[/bold] during a response to interrupt."
        if settings.general.spacebar_interrupt_enabled and sys.stdin.isatty()
        else ""
    )
    console.print(
        Panel(
            "[bold]Vox-Terminal[/bold] — Voice assistant for your project\n"
            + (
                "Speak naturally — listening starts automatically.\n"
                "Press [bold]Ctrl+C[/bold] to quit."
                if use_voice
                else "Type your question, or [bold]q[/bold] to quit."
            )
            + space_hint,
            title="Vox-Terminal",
            border_style="cyan",
        )
    )
    console.print("[dim]Warming up...[/dim]")

    # Run heavy startup tasks concurrently:
    # 1. Context assembly (git + directory tree)
    # 2. Whisper model preload (if using local whisper)
    t0 = _time.monotonic()
    loop = asyncio.get_running_loop()
    interactive_context_settings = _build_interactive_context_settings(settings.context)
    logger.info(
        "Interactive context profile (compact=%s, enabled_sources=%s, max_chars=%d)",
        interactive_context_settings.interactive_compact_context,
        ",".join(interactive_context_settings.enabled_sources)
        if interactive_context_settings.enabled_sources
        else "default",
        interactive_context_settings.max_context_chars,
    )

    async def _assemble_context() -> str:
        assembler = ContextAssembler(
            settings.general,
            settings.mcp,
            context_settings=interactive_context_settings,
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

    async def _preload_vad() -> None:
        if vad is None or not hasattr(vad, "_load_model"):
            return
        model_loaded = bool(getattr(vad, "_model", None))
        if model_loaded:
            return
        try:
            await loop.run_in_executor(None, vad._load_model)
        except Exception as exc:
            logger.warning("VAD warmup failed; continuing with lazy load (%s)", exc)

    context_task = asyncio.create_task(_assemble_context())
    stt_task = asyncio.create_task(_preload_stt())
    vad_task = asyncio.create_task(_preload_vad())

    context, _, _ = await asyncio.gather(context_task, stt_task, vad_task)
    logger.info("Startup completed in %.2fs", _time.monotonic() - t0)
    logger.info("Startup context assembled (chars=%d)", len(context))

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
            turn_id = generate_turn_id()
            turn_waterfall = TurnWaterfall()
            if use_voice:
                display_state.phase = "listening"
                try:
                    audio = await capture.record_until_silence(
                        silence_threshold=settings.stt.silence_threshold,
                        silence_duration=settings.stt.silence_duration,
                        silence_duration_after_speech=settings.stt.silence_duration_after_speech,
                        adaptive_endpointing=settings.stt.adaptive_endpointing,
                        speech_start_threshold=settings.stt.speech_start_threshold,
                        speech_end_threshold=settings.stt.speech_end_threshold,
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

                speech_started_at = capture.last_speech_started_at
                if speech_started_at is not None:
                    turn_waterfall.mark("speech_detected_ms", at=speech_started_at)
                turn_waterfall.mark("record_end_ms")

                display_state.phase = "transcribing"
                with TurnContext(turn_id):
                    stt_t0 = _time.monotonic()
                    turn_waterfall.mark("stt_start_ms", at=stt_t0)
                    result = await stt.transcribe(audio, settings.stt.sample_rate)
                    turn_waterfall.mark("stt_end_ms")
                    logger.info(
                        "STT stage finished (samples=%d, elapsed_ms=%.0f, confidence=%s)",
                        audio.size,
                        (_time.monotonic() - stt_t0) * 1000,
                        f"{result.confidence:.2f}" if result.confidence is not None else "n/a",
                    )
                    if settings.general.log_sensitive:
                        logger.debug("STT transcription: %s", result.text)
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

            with TurnContext(turn_id):
                logger.info(
                    "Turn started (mode=%s, question_chars=%d)",
                    "voice" if use_voice else "text",
                    len(question),
                )
                if settings.general.log_sensitive:
                    logger.debug("User question: %s", question)

            display_state.question = question
            display_state.response_chunks.clear()
            display_state.phase = "thinking"
            display_state.turn_start = _time.monotonic()

            # Inject referenced file contents into the question
            from vox_terminal.context.sources.inline import inject_file_context

            inline_t0 = _time.monotonic()
            enriched_question = inject_file_context(
                question,
                settings.general.project_root,
            )
            turn_waterfall.mark("inline_context_end_ms")
            logger.info(
                "Inline context stage finished (elapsed_ms=%.0f, enriched_chars=%d)",
                (_time.monotonic() - inline_t0) * 1000,
                len(enriched_question),
            )

            # --- TTS playback (with optional barge-in) ---
            interrupted_audio = None
            barge_in_interrupted = False

            if use_voice and settings.general.barge_in_enabled:
                # Barge-in path: monitor mic during TTS (opt-in for headphone users)
                record_task = asyncio.create_task(
                    capture.record_until_silence(
                        silence_threshold=settings.stt.silence_threshold,
                        silence_duration=settings.stt.silence_duration,
                        silence_duration_after_speech=settings.stt.silence_duration_after_speech,
                        adaptive_endpointing=settings.stt.adaptive_endpointing,
                        speech_start_threshold=settings.stt.speech_start_threshold,
                        speech_end_threshold=settings.stt.speech_end_threshold,
                        max_duration=settings.stt.max_record_duration,
                    )
                )

                async def _speak_with_barge_in(
                    q: str = enriched_question,
                    h: list[Message] | None = history or None,
                    current_turn_id: str = turn_id,
                    waterfall: TurnWaterfall = turn_waterfall,
                ) -> tuple[str, bool]:
                    """Run TTS while watching for speech on the mic and/or spacebar."""
                    with TurnContext(current_turn_id):
                        tts_task = asyncio.create_task(
                            _ask_and_speak(
                                q,
                                llm,
                                tts,
                                h,
                                display_state=display_state,
                                log_sensitive=settings.general.log_sensitive,
                                waterfall=waterfall,
                            )
                        )

                    stop_event = threading.Event()
                    space_task: asyncio.Task[None] | None = None
                    if settings.general.spacebar_interrupt_enabled:
                        space_task = asyncio.create_task(
                            _wait_for_spacebar_interrupt(tts, stop_event)
                        )

                    speech_event = capture.speech_started
                    grace_started_at = _time.monotonic()
                    applied_grace_s = settings.general.barge_in_grace_max_seconds
                    while not tts_task.done():
                        first_audio_seen = waterfall.elapsed_ms("tts_first_audio_ms") is not None
                        applied_grace_s = (
                            settings.general.barge_in_grace_min_seconds
                            if first_audio_seen
                            else settings.general.barge_in_grace_max_seconds
                        )
                        if _time.monotonic() - grace_started_at >= applied_grace_s:
                            break
                        await asyncio.sleep(0.02)
                    if speech_event is not None:
                        speech_event.clear()

                    # Hysteresis: require sustained positive VAD checks before interrupt.
                    consecutive_hits = 0
                    required_hits = max(1, settings.general.barge_in_required_hits)
                    poll_interval = max(0.01, settings.general.barge_in_poll_interval_ms / 1000.0)
                    speech_hits = 0
                    speech_first_seen_at: float | None = None
                    interrupt_detected_at: float | None = None
                    interrupted_by_user = False

                    while not tts_task.done():
                        if space_task is not None and space_task.done():
                            interrupted_by_user = True
                            break
                        if speech_event is not None and speech_event.is_set():
                            now = _time.monotonic()
                            speech_hits += 1
                            if speech_first_seen_at is None:
                                speech_first_seen_at = now
                                waterfall.mark("barge_in_speech_detected_ms", at=now)
                            consecutive_hits += 1
                            speech_event.clear()
                            if consecutive_hits >= required_hits:
                                interrupt_detected_at = now
                                waterfall.mark("barge_in_interrupt_ms", at=now)
                                interrupted_by_user = True
                                tts.interrupt()
                                break
                        else:
                            consecutive_hits = 0
                        await asyncio.sleep(poll_interval)

                    stop_event.set()
                    if space_task is not None and not space_task.done():
                        space_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await space_task

                    response = await tts_task
                    if (
                        interrupted_by_user
                        and speech_first_seen_at is not None
                        and interrupt_detected_at is not None
                    ):
                        logger.info(
                            "Barge-in metrics (triggered=true, detection_latency_ms=%.0f, speech_hits=%d, required_hits=%d, grace_ms=%.0f)",
                            (interrupt_detected_at - speech_first_seen_at) * 1000,
                            speech_hits,
                            required_hits,
                            applied_grace_s * 1000,
                        )
                    elif speech_first_seen_at is not None:
                        logger.info(
                            "Barge-in metrics (triggered=false, possible_false_trigger=true, speech_hits=%d, required_hits=%d, grace_ms=%.0f)",
                            speech_hits,
                            required_hits,
                            applied_grace_s * 1000,
                        )
                    else:
                        logger.info(
                            "Barge-in metrics (triggered=false, speech_hits=0, required_hits=%d, grace_ms=%.0f)",
                            required_hits,
                            applied_grace_s * 1000,
                        )
                    return response, interrupted_by_user

                response_text, barge_in_interrupted = await _speak_with_barge_in()

                if barge_in_interrupted:
                    # User interrupted — finish recording their speech
                    display.print_static("[dim](interrupted)[/dim]")
                    display_state.phase = "listening"
                    display_state.response_chunks.clear()
                    try:
                        interrupted_audio = await asyncio.wait_for(record_task, timeout=5.0)
                    except TimeoutError:
                        logger.warning("Barge-in record timed out — cancelling")
                        capture.cancel_recording()
                        with contextlib.suppress(AudioCaptureError, asyncio.CancelledError):
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
                # Default path: TTS with optional spacebar-to-interrupt
                with TurnContext(turn_id):
                    stop_event = threading.Event()
                    tts_task = asyncio.create_task(
                        _ask_and_speak(
                            enriched_question,
                            llm,
                            tts,
                            history or None,
                            display_state=display_state,
                            log_sensitive=settings.general.log_sensitive,
                            waterfall=turn_waterfall,
                        )
                    )
                    if settings.general.spacebar_interrupt_enabled:
                        space_task = asyncio.create_task(
                            _wait_for_spacebar_interrupt(tts, stop_event)
                        )
                        _done, pending = await asyncio.wait(
                            [tts_task, space_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        stop_event.set()
                        for t in pending:
                            t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await t
                    else:
                        await tts_task
                    response_text = await tts_task

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

            with TurnContext(turn_id):
                _log_turn_waterfall(turn_waterfall)

            # If user interrupted via barge-in, process their speech immediately
            if interrupted_audio is not None and interrupted_audio.size > 0:
                display_state.phase = "transcribing"
                follow_up_turn_id = generate_turn_id()
                follow_up_waterfall = TurnWaterfall()
                with TurnContext(follow_up_turn_id):
                    stt_t0 = _time.monotonic()
                    follow_up_waterfall.mark("stt_start_ms", at=stt_t0)
                    result = await stt.transcribe(interrupted_audio, settings.stt.sample_rate)
                    follow_up_waterfall.mark("stt_end_ms")
                    logger.info(
                        "STT stage finished (barge_in=true, samples=%d, elapsed_ms=%.0f, confidence=%s)",
                        interrupted_audio.size,
                        (_time.monotonic() - stt_t0) * 1000,
                        f"{result.confidence:.2f}" if result.confidence is not None else "n/a",
                    )
                    if settings.general.log_sensitive:
                        logger.debug("STT transcription: %s", result.text)
                question = result.text.strip()
                if question and question.lower() not in ("q", "quit", "exit"):
                    with TurnContext(follow_up_turn_id):
                        logger.info(
                            "Turn started (mode=barge-in, question_chars=%d)",
                            len(question),
                        )
                        if settings.general.log_sensitive:
                            logger.debug("User question: %s", question)
                    display_state.question = question
                    display_state.response_chunks.clear()
                    display_state.phase = "thinking"

                    inline_t0 = _time.monotonic()
                    enriched_question = inject_file_context(
                        question,
                        settings.general.project_root,
                    )
                    follow_up_waterfall.mark("inline_context_end_ms")
                    logger.info(
                        "Inline context stage finished (barge_in=true, elapsed_ms=%.0f, enriched_chars=%d)",
                        (_time.monotonic() - inline_t0) * 1000,
                        len(enriched_question),
                    )
                    with TurnContext(follow_up_turn_id):
                        response_text = await _ask_and_speak(
                            enriched_question,
                            llm,
                            tts,
                            history or None,
                            display_state=display_state,
                            log_sensitive=settings.general.log_sensitive,
                            waterfall=follow_up_waterfall,
                        )

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
                    with TurnContext(follow_up_turn_id):
                        _log_turn_waterfall(follow_up_waterfall)
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
# Shared option — resolved once in the callback, stored in ctx.obj
# ---------------------------------------------------------------------------

_project_root_option = typer.Option(
    None,
    "--project-root",
    "-p",
    help="Explicit project root directory. "
    "Resolved automatically via vox-terminal.toml / git root / cwd when omitted.",
    exists=True,
    file_okay=False,
    resolve_path=True,
)

_project_path_argument = typer.Argument(
    None,
    help="Path to the project directory (default: auto-detect from cwd).",
    exists=True,
    file_okay=False,
    resolve_path=True,
)


def _resolve_settings(
    project_root: Path | None = None,
) -> VoxTerminalSettings:
    """Load settings with the resolved project root threaded in."""
    resolved = resolve_project_root(project_root)
    return load_settings(project_root=resolved)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback()
def main(
    ctx: typer.Context,
    text: str | None = typer.Option(None, "--text", "-t", help="Ask a text question"),
    project_root: Path | None = _project_root_option,
) -> None:
    """Terminal voice assistant for Cursor IDE."""
    if ctx.invoked_subcommand is not None:
        # Store for subcommands that need it
        ctx.ensure_object(dict)
        ctx.obj["project_root"] = project_root
        return
    settings = _resolve_settings(project_root)
    _configure_logging(settings.general)
    if text:
        asyncio.run(_ask_once(text, settings))
    else:
        asyncio.run(_interactive_loop(settings))


@app.command()
def start(
    ctx: typer.Context,
    project_path: Path | None = _project_path_argument,
) -> None:
    """Bootstrap and launch vox-terminal for a project.

    Resolves the project root, validates prerequisites, then starts the
    interactive voice loop.  Designed for one-command usage from any repo:

        vox-terminal start .
        vox-terminal start /path/to/my-project
    """
    parent_root = (ctx.obj or {}).get("project_root")
    explicit = project_path or parent_root
    settings = _resolve_settings(explicit)
    _configure_logging(settings.general)

    root = settings.general.project_root
    console.print(
        Panel(
            f"[bold]Project:[/bold] {root}\n"
            f"[bold]LLM:[/bold]     {settings.llm.model}\n"
            f"[bold]STT:[/bold]     {settings.stt.engine}\n"
            f"[bold]TTS:[/bold]     {settings.tts.engine}",
            title="Vox-Terminal Bootstrap",
            border_style="cyan",
        )
    )

    if not settings.llm.api_key:
        console.print(
            "[red]No API key configured.[/red]\n"
            '  export VOX_TERMINAL_LLM__API_KEY="your-anthropic-key"\n'
            "[dim]Add it to ~/.zshrc to persist across sessions.[/dim]"
        )
        raise typer.Exit(code=1)

    asyncio.run(_interactive_loop(settings))


@app.command()
def ask(
    ctx: typer.Context,
    text: str = typer.Option(..., "--text", "-t", help="Question to ask"),
    project_root: Path | None = _project_root_option,
) -> None:
    """Ask a single question via text."""
    explicit = project_root or (ctx.obj or {}).get("project_root")
    settings = _resolve_settings(explicit)
    _configure_logging(settings.general)
    asyncio.run(_ask_once(text, settings))


@app.command()
def context(
    ctx: typer.Context,
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Print context"),
    project_root: Path | None = _project_root_option,
) -> None:
    """Show the project context that would be sent to the LLM."""
    explicit = project_root or (ctx.obj or {}).get("project_root")
    settings = _resolve_settings(explicit)
    assembler = ContextAssembler(
        settings.general,
        settings.mcp,
        context_settings=settings.context,
    )
    ctx_text = assembler.assemble(
        include_git=settings.mcp.include_git,
        include_tree=settings.mcp.include_tree,
    )
    if preview:
        if ctx_text:
            console.print(Panel(ctx_text, title="Project Context", border_style="green"))
        else:
            console.print("[yellow]No project context available.[/yellow]")


@app.command()
def logs(
    tail: bool = typer.Option(False, "--tail", help="Follow log output"),
    lines: int = typer.Option(100, "--lines", "-n", min=1, help="Show last N lines"),
) -> None:
    """Show Vox-Terminal logs from the rotating log file."""
    settings = load_settings()
    log_path = _resolve_log_file_path(settings.general)

    if not log_path.is_file():
        console.print(f"[yellow]No log file found at {log_path}[/yellow]")
        return

    def _read_lines() -> list[str]:
        try:
            return log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    content = _read_lines()
    for line in content[-lines:]:
        console.print(line, highlight=False)

    if not tail:
        return

    console.print(f"[dim]Tailing {log_path}. Press Ctrl+C to stop.[/dim]")
    shown = len(content)
    try:
        while True:
            _time.sleep(0.5)
            updated = _read_lines()
            if len(updated) > shown:
                for line in updated[shown:]:
                    console.print(line, highlight=False)
                shown = len(updated)
    except KeyboardInterrupt:
        return


@app.command()
def diagnose(
    ctx: typer.Context,
    no_audio: bool = typer.Option(
        False,
        "--no-audio",
        help="Skip audible TTS playback check",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        min=1,
        help="Run diagnostics repeatedly for latency benchmarking",
    ),
    text_only: bool = typer.Option(
        False,
        "--text-only",
        help="Skip STT/TTS and benchmark the text pipeline only",
    ),
    minimal_context: bool = typer.Option(
        False,
        "--minimal-context",
        help="Use a lightweight context profile during diagnostics",
    ),
    project_root: Path | None = _project_root_option,
) -> None:
    """Run a quick health check across context, STT, LLM, and TTS."""
    explicit = project_root or (ctx.obj or {}).get("project_root")
    settings = _resolve_settings(explicit)
    _configure_logging(settings.general)

    logger.info(
        "Diagnose benchmark started (iterations=%d, no_audio=%s, text_only=%s, minimal_context=%s)",
        iterations,
        no_audio,
        text_only,
        minimal_context,
    )

    all_runs: list[list[tuple[str, bool, str, float]]] = []
    has_failure = False
    for index in range(iterations):
        results = asyncio.run(
            _run_diagnostics(
                settings,
                no_audio=no_audio,
                text_only=text_only,
                minimal_context=minimal_context,
            )
        )
        all_runs.append(results)
        if iterations > 1:
            console.print(f"[bold]Iteration {index + 1}/{iterations}[/bold]")
        for stage, ok, detail, elapsed_ms in results:
            status = "OK" if ok else "FAIL"
            style = "green" if ok else "red"
            if not ok:
                has_failure = True
            console.print(
                f"[{style}]{stage.upper():<8} {status}[/{style}] "
                f"{detail} ({elapsed_ms:.0f} ms)"
            )
            logger.info(
                "Diagnose benchmark result (iteration=%d, stage=%s, ok=%s, elapsed_ms=%.0f, detail=%s)",
                index + 1,
                stage,
                ok,
                elapsed_ms,
                detail,
            )

    if iterations > 1:
        stage_times: dict[str, list[float]] = {}
        stage_failures: dict[str, int] = {}
        for run in all_runs:
            for stage, ok, _detail, elapsed_ms in run:
                stage_times.setdefault(stage, []).append(elapsed_ms)
                if not ok:
                    stage_failures[stage] = stage_failures.get(stage, 0) + 1
        console.print("\n[bold]Benchmark summary[/bold]")
        for stage in sorted(stage_times):
            samples = stage_times[stage]
            avg_ms = sum(samples) / len(samples)
            min_ms = min(samples)
            max_ms = max(samples)
            failures = stage_failures.get(stage, 0)
            console.print(
                f"{stage.upper():<8} avg={avg_ms:.0f} ms min={min_ms:.0f} ms max={max_ms:.0f} ms failures={failures}/{len(samples)}"
            )
            logger.info(
                "Diagnose benchmark summary (stage=%s, avg_ms=%.0f, min_ms=%.0f, max_ms=%.0f, failures=%d, samples=%d)",
                stage,
                avg_ms,
                min_ms,
                max_ms,
                failures,
                len(samples),
            )

    if has_failure:
        raise typer.Exit(code=1)


@app.command()
def serve(
    ctx: typer.Context,
    project_root: Path | None = _project_root_option,
) -> None:
    """Start the MCP server for Cursor integration."""
    try:
        from vox_terminal.mcp_server import create_mcp_server

        explicit = project_root or (ctx.obj or {}).get("project_root")
        resolved = resolve_project_root(explicit)
        server = create_mcp_server(project_root=resolved)
        server.run(transport="stdio")
    except ImportError:
        console.print("[red]MCP server dependencies not available.[/red]")
        sys.exit(1)


def app_main() -> None:
    """Entry point for the CLI — called by the console_scripts entrypoint."""
    app()
