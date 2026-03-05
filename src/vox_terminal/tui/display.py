"""SessionDisplay — orchestrates a single Rich Live layout for the TUI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from vox_terminal.tui.state import DisplayState
from vox_terminal.tui.widgets import ResponsePanel, SoundBar, StateSpinner, StatusBar

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 1.0 / 15  # ~15 fps


class SessionDisplay:
    """Manages a persistent Rich Live display for the interactive session.

    One Live display runs for the entire ``_interactive_loop`` lifetime.
    A Layout splits into *body* (swaps per state) and *footer* (always-visible
    status bar).  An async refresh loop at ~15 fps reads shared state and
    rebuilds the layout.
    """

    def __init__(
        self,
        state: DisplayState,
        console: Console | None = None,
    ) -> None:
        self.state = state
        self._console = console or Console()
        self._levels: deque[float] = deque(maxlen=60)
        self._capture: Any | None = None
        self._live: Live | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, capture: Any | None = None) -> None:
        """Enter the Live display and start the async refresh loop."""
        self._capture = capture
        layout = Layout()
        layout.split_column(
            Layout(name="body", ratio=1),
            Layout(name="footer", size=1),
        )
        self._live = Live(
            layout,
            console=self._console,
            refresh_per_second=15,
            transient=False,
        )
        self._live.start()
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        """Cancel the refresh loop and exit the Live display."""
        self._running = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def pause(self) -> None:
        """Temporarily stop Live rendering (for raw ``input()`` calls)."""
        if self._live is not None:
            self._live.stop()

    def resume(self) -> None:
        """Re-enter Live rendering after a ``pause()``."""
        if self._live is not None:
            self._live.start()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def print_static(self, text: str) -> None:
        """Print a message outside the Live context (errors, goodbye, etc.)."""
        if self._live is not None and self._live.is_started:
            self._live.console.print(text)
        else:
            self._console.print(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        """Rebuild the layout at ~15 fps from shared state."""
        while self._running:
            try:
                self._update_audio_level()
                if self._live is not None and self._live.is_started:
                    renderable = self._live.renderable
                    if isinstance(renderable, Layout):
                        renderable["body"].update(self._render_body())
                        renderable["footer"].update(StatusBar(self.state))
            except Exception:
                logger.debug("Display refresh error", exc_info=True)
            await asyncio.sleep(_REFRESH_INTERVAL)

    def _update_audio_level(self) -> None:
        """Read the current audio level from the capture device."""
        if self._capture is not None and hasattr(self._capture, "audio_level"):
            level = self._capture.audio_level
            self._levels.append(level)
            self.state.audio_level = level
        else:
            self._levels.append(0.0)

    def _render_body(self) -> Any:
        """Return the Rich renderable for the current phase."""
        phase = self.state.phase

        if phase == "listening":
            return Columns(
                [StateSpinner("listening"), SoundBar(self._levels)],
                expand=True,
            )

        if phase == "transcribing":
            return StateSpinner("transcribing")

        if phase == "thinking":
            return Group(
                Text(f"You: {self.state.question}", style="bold cyan"),
                StateSpinner("thinking"),
            )

        if phase == "speaking":
            return ResponsePanel(self.state.response_chunks, self.state.question)

        # idle
        return Text("")
