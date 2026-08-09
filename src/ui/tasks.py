"""Background execution for geometry work that would otherwise freeze the UI.

Generating a TPMS lattice at high resolution samples millions of points, and a
robust boolean voxelises two meshes.  Run either on the GUI thread and the whole
window stops repainting - no progress, no cancel, and Windows eventually paints
"Not Responding" over the viewport.

This module puts that work on a ``QThreadPool`` and delivers results back on the
GUI thread through queued signals.  The contract for a worker function is small:

.. code-block:: python

    def build(context, resolution):
        for step in range(resolution):
            context.raise_if_cancelled()
            context.progress(step / resolution, f"Sampling slice {step}")
        return mesh

    runner.submit(build, resolution, label="TPMS Lattice", on_result=self.add_mesh)

The ``context`` argument is injected; everything after it is the caller's own.

Threading rules that callers must respect
-----------------------------------------
- The worker runs off the GUI thread, so it must not touch widgets, actors, or
  the plotter.  Build geometry, return it, and let the ``on_result`` callback do
  the scene mutation.
- Kernel meshes are plain numpy and safe to hand across threads.  A
  ``pyvista`` actor is not - never construct one inside a worker.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot

__all__ = ["TaskCancelled", "TaskContext", "TaskHandle", "TaskRunner"]


class TaskCancelled(Exception):
    """Raised inside a worker when the user cancels it."""


class _TaskSignals(QObject):
    """Signals for a task.

    ``QRunnable`` is not a ``QObject``, so the signals have to live on a
    separate object that the runnable owns.
    """

    started = Signal(str)
    progress = Signal(float, str)
    result = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    finished = Signal()


class TaskContext:
    """The handle a worker uses to report progress and observe cancellation."""

    __slots__ = ("_label", "_signals", "_state")

    def __init__(self, signals, state, label):
        self._signals = signals
        self._state = state
        self._label = label

    @property
    def label(self):
        return self._label

    @property
    def cancelled(self):
        return bool(self._state["cancelled"])

    def raise_if_cancelled(self):
        """Abort the worker if the user has asked to cancel.

        Call this inside any loop that could run for more than a moment; it is
        the only thing that makes a long operation interruptible.
        """
        if self._state["cancelled"]:
            raise TaskCancelled(f"{self._label} was cancelled.")

    def progress(self, fraction, message=""):
        """Report completion in the range 0-1 with an optional status line."""
        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            fraction = 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        self._signals.progress.emit(fraction, str(message))


class TaskHandle(QRunnable):
    """One unit of background work.

    Connect to the signals on :attr:`signals`, or pass callbacks to
    :meth:`TaskRunner.submit`, which wires them for you.
    """

    def __init__(self, function, args=(), kwargs=None, label="Working"):
        super().__init__()
        self.setAutoDelete(False)  # the runner keeps a reference until finished
        self._function = function
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._state = {"cancelled": False}
        self.label = str(label)
        self.signals = _TaskSignals()

    def cancel(self):
        """Ask the worker to stop at its next cancellation check."""
        self._state["cancelled"] = True

    @property
    def is_cancelled(self):
        return bool(self._state["cancelled"])

    @Slot()
    def run(self):
        context = TaskContext(self.signals, self._state, self.label)
        self.signals.started.emit(self.label)
        try:
            if self._state["cancelled"]:
                raise TaskCancelled(f"{self.label} was cancelled.")
            value = self._function(context, *self._args, **self._kwargs)
        except TaskCancelled:
            self.signals.cancelled.emit()
        except Exception as error:
            # The traceback is far more useful than the message alone when the
            # failure is deep inside a numpy routine.
            self.signals.failed.emit(str(error) or type(error).__name__, traceback.format_exc())
        else:
            self.signals.result.emit(value)
        finally:
            self.signals.finished.emit()


class TaskRunner(QObject):
    """Owns the thread pool and keeps running tasks alive.

    Without holding a reference, Python can collect a ``QRunnable`` while Qt is
    still running it, which crashes the interpreter rather than raising.
    """

    busy_changed = Signal(bool)
    progress = Signal(float, str)
    task_started = Signal(str)
    task_finished = Signal(str)

    def __init__(self, parent=None, max_threads=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        if max_threads is not None:
            self._pool.setMaxThreadCount(int(max_threads))
        else:
            # Leave a core for the GUI and the renderer so the viewport keeps
            # responding while a lattice builds.
            self._pool.setMaxThreadCount(max(1, QThreadPool.globalInstance().maxThreadCount() - 1))
        self._active = {}

    @property
    def active_count(self):
        return len(self._active)

    @property
    def is_busy(self):
        return bool(self._active)

    def submit(
        self,
        function,
        *args,
        label="Working",
        on_result=None,
        on_error=None,
        on_cancelled=None,
        on_progress=None,
        **kwargs,
    ):
        """Run ``function(context, *args, **kwargs)`` off the GUI thread.

        Returns the :class:`TaskHandle` so the caller can cancel it.  Callbacks
        fire on the GUI thread, so they may safely touch widgets and actors.
        """
        task = TaskHandle(function, args, kwargs, label=label)

        if on_result is not None:
            task.signals.result.connect(on_result, Qt.QueuedConnection)
        if on_error is not None:
            task.signals.failed.connect(on_error, Qt.QueuedConnection)
        if on_cancelled is not None:
            task.signals.cancelled.connect(on_cancelled, Qt.QueuedConnection)
        if on_progress is not None:
            task.signals.progress.connect(on_progress, Qt.QueuedConnection)

        task.signals.progress.connect(self.progress, Qt.QueuedConnection)
        task.signals.started.connect(self.task_started, Qt.QueuedConnection)
        task.signals.finished.connect(
            lambda handle=task: self._on_finished(handle), Qt.QueuedConnection
        )

        was_busy = self.is_busy
        self._active[id(task)] = task
        if not was_busy:
            self.busy_changed.emit(True)

        self._pool.start(task)
        return task

    def cancel_all(self):
        """Ask every running task to stop; they finish at their next check."""
        for task in list(self._active.values()):
            task.cancel()

    def wait_for_done(self, timeout_ms=-1):
        """Block until the pool drains. Only for shutdown - never for the UI."""
        return self._pool.waitForDone(int(timeout_ms))

    def shutdown(self, timeout_ms=5000):
        """Cancel outstanding work and wait briefly before the window closes."""
        self.cancel_all()
        self._pool.waitForDone(int(timeout_ms))
        self._active.clear()

    def _on_finished(self, task):
        self._active.pop(id(task), None)
        self.task_finished.emit(task.label)
        if not self._active:
            self.busy_changed.emit(False)
