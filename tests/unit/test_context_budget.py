"""Tests for context budget allocation."""

from __future__ import annotations

from vox_terminal.context.budget import ContextBudget, ContextFragment


class TestContextBudget:
    def test_allocate_full_fragment(self) -> None:
        budget = ContextBudget(total_chars=100)
        fragment = ContextFragment(name="a", content="hello", priority=10)

        allocated = budget.allocate(fragment)

        assert allocated == "hello"
        assert budget.remaining_chars == 95

    def test_allocate_truncates_when_budget_exhausted(self) -> None:
        budget = ContextBudget(total_chars=5)
        fragment = ContextFragment(name="a", content="hello world", priority=10)

        allocated = budget.allocate(fragment)

        assert allocated == "hello"
        assert budget.remaining_chars == 0

    def test_allocate_returns_empty_when_no_remaining_budget(self) -> None:
        budget = ContextBudget(total_chars=0)
        fragment = ContextFragment(name="a", content="hello", priority=10)

        allocated = budget.allocate(fragment)

        assert allocated == ""
        assert budget.remaining_chars == 0

    def test_priority_order_is_greedy(self) -> None:
        fragments = [
            ContextFragment(name="low", content="LOW", priority=10),
            ContextFragment(name="high", content="HIGH", priority=100),
        ]
        budget = ContextBudget(total_chars=4)
        chosen: list[str] = []

        for fragment in sorted(fragments, key=lambda f: f.priority, reverse=True):
            text = budget.allocate(fragment)
            if text:
                chosen.append(text)

        assert chosen == ["HIGH"]
