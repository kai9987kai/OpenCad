"""Tests for the command-based undo/redo history.

The history is what stands between a user and losing work, so these tests care
about the awkward cases: replay must not re-record, a failed macro must not
leave half an edit behind, and trimming must never discard the step the user
just made.
"""

from __future__ import annotations

import pytest

from src.core.history import (
    CallbackCommand,
    Command,
    History,
    MacroCommand,
    SnapshotCommand,
)


class FakeClock:
    """A controllable monotonic clock so merge-window tests are deterministic."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def history(clock):
    return History(clock=clock)


def counter_command(state, key, before, after, label="Set", merge_key=None, size=0):
    """A command that writes a value into a dict, for observable replay."""
    return CallbackCommand(
        label,
        undo=lambda: state.__setitem__(key, before),
        redo=lambda: state.__setitem__(key, after),
        memory_bytes=size,
        merge_key=merge_key,
    )


class TestBasicReplay:
    def test_undo_and_redo_restore_values(self, history):
        state = {"x": 0}
        state["x"] = 1
        history.push(counter_command(state, "x", 0, 1))

        assert history.undo() == "Set"
        assert state["x"] == 0
        assert history.redo() == "Set"
        assert state["x"] == 1

    def test_undo_on_empty_history_returns_none(self, history):
        assert history.undo() is None
        assert history.redo() is None
        assert not history.can_undo
        assert not history.can_redo

    def test_stack_order_is_last_in_first_out(self, history, clock):
        log = []
        for step in range(3):
            clock.advance(10.0)
            history.push(
                CallbackCommand(
                    f"step{step}",
                    undo=lambda step=step: log.append(f"undo{step}"),
                    redo=lambda step=step: log.append(f"redo{step}"),
                )
            )
        history.undo()
        history.undo()
        assert log == ["undo2", "undo1"]

    def test_new_edit_clears_the_redo_branch(self, history, clock):
        state = {"x": 0}
        history.push(counter_command(state, "x", 0, 1))
        history.undo()
        assert history.can_redo

        clock.advance(10.0)
        history.push(counter_command(state, "x", 0, 2))
        assert not history.can_redo

    def test_replay_does_not_record_new_entries(self, history, clock):
        """Undo must not push its own inverse back onto the stack."""
        state = {"x": 0}
        nested = History(clock=clock)

        def undo():
            state["x"] = 0
            # A real call site might route through the same history object.
            assert not history.is_recording

        history.push(CallbackCommand("Set", undo=undo, redo=lambda: None))
        history.undo()
        assert len(history) == 0
        assert len(nested) == 0

    def test_push_rejects_non_commands(self, history):
        with pytest.raises(TypeError):
            history.push("not a command")

    def test_callback_command_requires_callables(self):
        with pytest.raises(TypeError):
            CallbackCommand("bad", undo=None, redo=lambda: None)

    def test_base_command_is_abstract_in_practice(self):
        with pytest.raises(NotImplementedError):
            Command("x").undo()
        with pytest.raises(NotImplementedError):
            Command("x").redo()


class TestMenuLabels:
    def test_labels_track_the_top_of_each_stack(self, history, clock):
        assert history.undo_text() == "Undo"
        assert history.redo_text() == "Redo"

        history.push(CallbackCommand("Extrude", lambda: None, lambda: None))
        assert history.undo_label == "Extrude"
        assert history.undo_text() == "Undo Extrude"

        history.undo()
        assert history.redo_text() == "Redo Extrude"
        assert history.undo_text() == "Undo"

    def test_on_change_fires_for_every_mutation(self, clock):
        calls = []
        history = History(clock=clock, on_change=lambda h: calls.append(len(h)))
        history.push(CallbackCommand("a", lambda: None, lambda: None))
        history.undo()
        history.redo()
        history.clear()
        assert calls == [1, 0, 1, 0]


class TestMerging:
    def test_rapid_edits_with_the_same_key_collapse(self, history, clock):
        state = {"x": 0}
        history.push(counter_command(state, "x", 0, 1, merge_key="pos:obj_0"))
        clock.advance(0.1)
        history.push(counter_command(state, "x", 1, 2, merge_key="pos:obj_0"))
        clock.advance(0.1)
        history.push(counter_command(state, "x", 2, 3, merge_key="pos:obj_0"))
        state["x"] = 3

        assert len(history) == 1
        # One undo must rewind the whole gesture, back to before it started.
        history.undo()
        assert state["x"] == 0
        history.redo()
        assert state["x"] == 3

    def test_edits_outside_the_merge_window_stay_separate(self, history, clock):
        state = {"x": 0}
        history.push(counter_command(state, "x", 0, 1, merge_key="pos:obj_0"))
        clock.advance(5.0)
        history.push(counter_command(state, "x", 1, 2, merge_key="pos:obj_0"))
        assert len(history) == 2

    def test_different_merge_keys_do_not_collapse(self, history, clock):
        state = {"x": 0, "y": 0}
        history.push(counter_command(state, "x", 0, 1, merge_key="pos:obj_0"))
        clock.advance(0.1)
        history.push(counter_command(state, "y", 0, 1, merge_key="pos:obj_1"))
        assert len(history) == 2

    def test_commands_without_a_merge_key_never_collapse(self, history, clock):
        state = {"x": 0}
        history.push(counter_command(state, "x", 0, 1))
        clock.advance(0.01)
        history.push(counter_command(state, "x", 1, 2))
        assert len(history) == 2

    def test_merging_clears_redo(self, history, clock):
        state = {"x": 0}
        history.push(counter_command(state, "x", 0, 1, merge_key="k"))
        history.undo()
        assert history.can_redo
        clock.advance(0.1)
        history.push(counter_command(state, "x", 0, 2, merge_key="k"))
        assert not history.can_redo


class TestMacros:
    def test_macro_groups_into_one_step(self, history, clock):
        state = {"a": 0, "b": 0}
        with history.macro("Array"):
            history.push(counter_command(state, "a", 0, 1))
            history.push(counter_command(state, "b", 0, 1))
        state["a"] = state["b"] = 1

        assert len(history) == 1
        assert history.undo_label == "Array"
        history.undo()
        assert state == {"a": 0, "b": 0}
        history.redo()
        assert state == {"a": 1, "b": 1}

    def test_macro_undo_runs_children_in_reverse(self):
        log = []
        macro = MacroCommand(
            "m",
            [
                CallbackCommand("1", lambda: log.append("u1"), lambda: log.append("r1")),
                CallbackCommand("2", lambda: log.append("u2"), lambda: log.append("r2")),
            ],
        )
        macro.undo()
        macro.redo()
        assert log == ["u2", "u1", "r1", "r2"]

    def test_empty_macro_leaves_no_entry(self, history):
        with history.macro("Nothing"):
            pass
        assert len(history) == 0

    def test_single_command_macro_is_unwrapped(self, history):
        with history.macro("One"):
            history.push(CallbackCommand("Inner", lambda: None, lambda: None))
        assert len(history) == 1
        assert history.undo_label == "Inner"

    def test_failed_macro_rolls_back_and_records_nothing(self, history):
        state = {"a": 0}
        with pytest.raises(RuntimeError), history.macro("Broken"):
            state["a"] = 1
            history.push(counter_command(state, "a", 0, 1))
            raise RuntimeError("operation failed halfway")

        assert len(history) == 0
        assert state["a"] == 0  # the applied half was rolled back

    def test_nested_macros_flatten(self, history):
        state = {"a": 0, "b": 0}
        with history.macro("Outer"):
            history.push(counter_command(state, "a", 0, 1))
            with history.macro("Inner"):
                history.push(counter_command(state, "b", 0, 1))
        assert len(history) == 1
        assert history.undo_label == "Outer"


class TestBudget:
    def test_count_limit_drops_the_oldest_entries(self, clock):
        history = History(limit=3, clock=clock)
        for step in range(5):
            clock.advance(10.0)
            history.push(CallbackCommand(f"s{step}", lambda: None, lambda: None))
        assert len(history) == 3
        assert history.undo_label == "s4"
        assert history.stats()["dropped"] == 2

    def test_memory_budget_trims_before_the_count_limit(self, clock):
        history = History(limit=100, memory_budget=1000, clock=clock)
        for step in range(6):
            clock.advance(10.0)
            history.push(
                CallbackCommand(
                    f"s{step}", lambda: None, lambda: None, memory_bytes=400
                )
            )
        assert history.memory_bytes() <= 1000
        assert len(history) == 2
        assert history.undo_label == "s5"

    def test_the_newest_step_survives_even_if_it_blows_the_budget(self, clock):
        history = History(memory_budget=10, clock=clock)
        history.push(
            CallbackCommand("huge", lambda: None, lambda: None, memory_bytes=10**9)
        )
        assert history.can_undo
        assert history.undo_label == "huge"

    def test_snapshot_command_reports_both_states(self):
        restored = []
        command = SnapshotCommand(
            "Open Project",
            before={"n": 1},
            after={"n": 2},
            restore=restored.append,
            sizer=lambda state: state["n"] * 100,
        )
        assert command.memory_bytes() == 300
        command.undo()
        command.redo()
        assert restored == [{"n": 1}, {"n": 2}]

    def test_snapshot_command_requires_a_restore_callable(self):
        with pytest.raises(TypeError):
            SnapshotCommand("x", before=None, after=None, restore=None)

    def test_stats_shape(self, history):
        history.push(CallbackCommand("a", lambda: None, lambda: None))
        stats = history.stats()
        assert stats["undo_depth"] == 1
        assert stats["redo_depth"] == 0
        assert "memory_bytes" in stats and "memory_budget" in stats

    def test_clear_empties_both_stacks(self, history, clock):
        history.push(CallbackCommand("a", lambda: None, lambda: None))
        history.undo()
        history.clear()
        assert not history.can_undo
        assert not history.can_redo

    def test_suspended_blocks_recording(self, history):
        with history.suspended():
            history.push(CallbackCommand("ignored", lambda: None, lambda: None))
        assert len(history) == 0
        assert history.is_recording
