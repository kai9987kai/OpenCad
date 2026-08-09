"""Undo/redo history built from commands rather than whole-scene snapshots.

The original implementation deep-copied every mesh in the scene on every edit
and kept forty of those copies.  A single TPMS lattice can be several hundred
megabytes of triangles, so the history could outgrow the model it was meant to
protect.  This module keeps the same ergonomics but charges each entry for the
memory it actually holds, trims by *budget* instead of by count, and lets cheap
edits (nudging a spin box, dragging a gizmo) record something proportional to
the change rather than to the scene.

Nothing here imports Qt or PyVista, so the whole thing is unit testable.
"""

from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager

__all__ = [
    "CallbackCommand",
    "Command",
    "History",
    "MacroCommand",
    "SnapshotCommand",
]

# A conservative default: enough for a long editing session on a normal model,
# small enough that history alone will not exhaust a 16 GB machine.
DEFAULT_MEMORY_BUDGET = 512 * 1024 * 1024
DEFAULT_LIMIT = 200
# Two consecutive edits closer together than this are candidates for merging,
# so dragging a slider produces one undo step instead of two hundred.
DEFAULT_MERGE_WINDOW = 0.75


class Command:
    """A reversible edit.

    Subclasses implement :meth:`undo` and :meth:`redo`.  ``redo`` must be safe
    to call immediately after ``undo`` any number of times - the history never
    calls ``redo`` to *perform* an edit for the first time, only to replay one.
    """

    __slots__ = ("label", "merge_key", "timestamp")

    def __init__(self, label="Edit", merge_key=None):
        self.label = str(label)
        self.merge_key = merge_key
        self.timestamp = 0.0

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError

    def memory_bytes(self):
        """Approximate retained bytes, used for budget-based trimming."""
        return 0

    def merge_with(self, other):
        """Absorb a newer command of the same kind; return True if absorbed.

        Used to collapse a stream of continuous edits into one undo step.  The
        default refuses, which is always safe.
        """
        return False

    def __repr__(self):
        return f"{type(self).__name__}({self.label!r})"


class CallbackCommand(Command):
    """A command defined by two thunks.

    This is the workhorse: call sites capture just the before/after values they
    changed, so an edit to one actor's colour costs a few dozen bytes instead of
    a copy of the scene.
    """

    __slots__ = ("_bytes", "_merge", "_redo", "_undo")

    def __init__(self, label, undo, redo, memory_bytes=0, merge_key=None, merge=None):
        super().__init__(label, merge_key)
        if not callable(undo) or not callable(redo):
            raise TypeError("undo and redo must be callables.")
        self._undo = undo
        self._redo = redo
        self._bytes = int(memory_bytes)
        self._merge = merge

    def undo(self):
        self._undo()

    def redo(self):
        self._redo()

    def memory_bytes(self):
        return self._bytes

    def merge_with(self, other):
        """Keep this command's *undo* and adopt the newer command's *redo*.

        That is what collapsing a drag means: rewind to where the gesture
        started, fast-forward to where it ended.
        """
        if self.merge_key is None or self.merge_key != other.merge_key:
            return False
        if self._merge is not None and not self._merge(other):
            return False
        self._redo = other._redo
        self._bytes = max(self._bytes, other.memory_bytes())
        self.label = other.label
        return True


class SnapshotCommand(Command):
    """Restore opaque before/after states through a single restore function.

    Kept for edits that genuinely rewrite the whole scene - opening a project,
    or a rebuild that touches every feature - where a targeted command would be
    more fragile than simply remembering both states.
    """

    __slots__ = ("_after", "_before", "_restore", "_sizer")

    def __init__(self, label, before, after, restore, sizer=None):
        super().__init__(label)
        if not callable(restore):
            raise TypeError("restore must be callable.")
        self._before = before
        self._after = after
        self._restore = restore
        self._sizer = sizer

    def undo(self):
        self._restore(self._before)

    def redo(self):
        self._restore(self._after)

    def memory_bytes(self):
        if self._sizer is None:
            return 0
        return int(self._sizer(self._before)) + int(self._sizer(self._after))


class MacroCommand(Command):
    """Several commands treated as one undo step.

    Undo runs the children in reverse so a compound edit unwinds in the order
    it was applied.
    """

    __slots__ = ("commands",)

    def __init__(self, label, commands=()):
        super().__init__(label)
        self.commands = list(commands)

    def add(self, command):
        self.commands.append(command)

    def undo(self):
        for command in reversed(self.commands):
            command.undo()

    def redo(self):
        for command in self.commands:
            command.redo()

    def memory_bytes(self):
        return sum(command.memory_bytes() for command in self.commands)

    def __len__(self):
        return len(self.commands)


class History:
    """A bounded undo/redo stack.

    ``on_change`` is invoked after every mutation so a menu or toolbar can
    refresh its enabled state and labels without polling.
    """

    def __init__(
        self,
        limit=DEFAULT_LIMIT,
        memory_budget=DEFAULT_MEMORY_BUDGET,
        merge_window=DEFAULT_MERGE_WINDOW,
        on_change=None,
        clock=time.monotonic,
    ):
        self.limit = max(int(limit), 1)
        self.memory_budget = max(int(memory_budget), 0)
        self.merge_window = float(merge_window)
        self.on_change = on_change
        self._clock = clock
        self._undo = []
        self._redo = []
        self._macro = None
        self._suspended = 0
        self._trimmed = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def push(self, command):
        """Record an edit that has *already been applied*.

        Returns the command that ended up on the stack, which may be an earlier
        one that absorbed this edit.
        """
        if not isinstance(command, Command):
            raise TypeError("History only accepts Command instances.")
        if self._suspended:
            return command

        command.timestamp = self._clock()

        if self._macro is not None:
            self._macro.add(command)
            return command

        if self._undo:
            top = self._undo[-1]
            recent = (command.timestamp - top.timestamp) <= self.merge_window
            if recent and top.merge_with(command):
                top.timestamp = command.timestamp
                self._redo.clear()
                self._notify()
                return top

        self._undo.append(command)
        self._redo.clear()
        self._enforce_limits()
        self._notify()
        return command

    def record(self, label, undo, redo, memory_bytes=0, merge_key=None):
        """Convenience wrapper around :class:`CallbackCommand`."""
        return self.push(
            CallbackCommand(
                label, undo, redo, memory_bytes=memory_bytes, merge_key=merge_key
            )
        )

    @contextmanager
    def macro(self, label):
        """Group everything pushed inside the block into one undo step.

        Nested macros flatten into the outermost one, and an empty macro is
        discarded so a no-op operation does not leave a dead entry.
        """
        if self._macro is not None:
            yield self._macro  # already grouping; nesting flattens
            return

        macro = MacroCommand(label)
        self._macro = macro
        try:
            yield macro
        except Exception:
            # A failed compound edit should not leave half of it on the stack.
            self._macro = None
            for command in reversed(macro.commands):
                # Best-effort rollback: one child failing to unwind must not
                # stop the rest from being unwound.
                with contextlib.suppress(Exception):
                    command.undo()
            self._notify()
            raise

        self._macro = None
        if not macro.commands:
            return
        if len(macro.commands) == 1:
            self.push(macro.commands[0])
        else:
            self.push(macro)

    @contextmanager
    def suspended(self):
        """Temporarily stop recording - used while replaying history itself."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    @property
    def is_recording(self):
        return self._suspended == 0

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def undo(self):
        """Undo one step and return its label, or ``None`` if there was none."""
        if not self._undo:
            return None
        command = self._undo.pop()
        with self.suspended():
            command.undo()
        self._redo.append(command)
        self._notify()
        return command.label

    def redo(self):
        if not self._redo:
            return None
        command = self._redo.pop()
        with self.suspended():
            command.redo()
        self._undo.append(command)
        self._notify()
        return command.label

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    @property
    def undo_label(self):
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self):
        return self._redo[-1].label if self._redo else None

    def undo_text(self):
        """Menu text such as ``"Undo Extrude"`` or a disabled-looking ``"Undo"``."""
        return f"Undo {self.undo_label}" if self._undo else "Undo"

    def redo_text(self):
        return f"Redo {self.redo_label}" if self._redo else "Redo"

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self):
        self._undo.clear()
        self._redo.clear()
        self._macro = None
        self._trimmed = 0
        self._notify()

    def memory_bytes(self):
        return sum(command.memory_bytes() for command in self._undo) + sum(
            command.memory_bytes() for command in self._redo
        )

    def stats(self):
        """A snapshot for the status bar or a diagnostics panel."""
        return {
            "undo_depth": len(self._undo),
            "redo_depth": len(self._redo),
            "memory_bytes": self.memory_bytes(),
            "memory_budget": self.memory_budget,
            "limit": self.limit,
            "dropped": self._trimmed,
        }

    def _enforce_limits(self):
        while len(self._undo) > self.limit:
            self._undo.pop(0)
            self._trimmed += 1
        # Always keep the most recent step, even if it alone exceeds the budget:
        # refusing to undo the edit a user just made would be worse than the
        # memory cost of remembering it.
        while len(self._undo) > 1 and self.memory_bytes() > self.memory_budget:
            self._undo.pop(0)
            self._trimmed += 1

    def _notify(self):
        if self.on_change is not None:
            self.on_change(self)

    def __len__(self):
        return len(self._undo)

    def __repr__(self):
        return (
            f"History(undo={len(self._undo)}, redo={len(self._redo)}, "
            f"memory={self.memory_bytes() / 1e6:.1f}MB)"
        )
