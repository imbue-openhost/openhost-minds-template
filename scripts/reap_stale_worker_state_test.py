"""Unit tests for the stranded worker/ticket reaper."""

from __future__ import annotations

import subprocess
from pathlib import Path

from reap_stale_worker_state import Ticket
from reap_stale_worker_state import dead_worker_names
from reap_stale_worker_state import live_agent_names
from reap_stale_worker_state import orphaned_tickets
from reap_stale_worker_state import parse_frontmatter
from reap_stale_worker_state import read_tickets
from reap_stale_worker_state import reap
from reap_stale_worker_state import release_tickets


class FakeRunner:
    """Records commands; returns a canned `mngr ls` payload, success otherwise."""

    def __init__(self, agents_json: str = '{"agents": []}', failing: set[str] | None = None) -> None:
        self.agents_json = agents_json
        self.failing = failing or set()
        self.commands: list[list[str]] = []

    def run(self, cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        self.commands.append(cmd)
        if cmd[:3] == ["mngr", "ls", "--format"]:
            return subprocess.CompletedProcess(cmd, 0, self.agents_json, "")
        if cmd[0] in self.failing:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _agent(name: str, state: str, *, worker: bool) -> dict:
    return {"name": name, "state": state, "labels": {"agent_created": "true"} if worker else {}}


def _write_ticket(
    tickets_dir: Path, ticket_id: str, *, status: str, assignee: str, step: bool = False
) -> None:
    tickets_dir.mkdir(parents=True, exist_ok=True)
    step_line = "step: true\n" if step else ""
    (tickets_dir / f"{ticket_id}.md").write_text(
        f"---\nid: {ticket_id}\nstatus: {status}\nassignee: {assignee}\n{step_line}---\n\nbody\n"
    )


# --- frontmatter parsing ---


def test_parse_frontmatter_reads_the_leading_block_only() -> None:
    text = "---\nid: t-1\nstatus: open\n---\n\nbody: not-frontmatter\n"
    assert parse_frontmatter(text) == {"id": "t-1", "status": "open"}


def test_parse_frontmatter_returns_empty_without_a_block() -> None:
    assert parse_frontmatter("no frontmatter here\n") == {}


def test_read_tickets_skips_files_without_an_id(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "junk.md").write_text("---\nstatus: open\n---\n")
    _write_ticket(tmp_path, "t-1", status="open", assignee="")
    assert [t.id for t in read_tickets(tmp_path)] == ["t-1"]


# --- agent classification ---


def test_dead_worker_names_selects_only_stopped_workers() -> None:
    agents = [
        _agent("update-self", "STOPPED", worker=True),
        _agent("busy-worker", "RUNNING", worker=True),
        _agent("minds", "STOPPED", worker=False),
    ]
    assert dead_worker_names(agents) == ["update-self"]


def test_live_agent_names_are_the_running_ones() -> None:
    agents = [_agent("minds", "RUNNING", worker=False), _agent("gone", "STOPPED", worker=False)]
    assert live_agent_names(agents) == {"minds"}


# --- ticket classification ---


def test_orphaned_tickets_ignores_those_owned_by_a_live_agent() -> None:
    tickets = [
        Ticket(Path("a"), "t-1", "in_progress", "minds", is_step=False),
        Ticket(Path("b"), "t-2", "in_progress", "dead-worker", is_step=False),
    ]
    assert [t.id for t in orphaned_tickets(tickets, {"minds"})] == ["t-2"]


def test_orphaned_tickets_ignores_unassigned_and_non_in_progress() -> None:
    tickets = [
        Ticket(Path("a"), "t-1", "in_progress", "", is_step=False),
        Ticket(Path("b"), "t-2", "open", "dead", is_step=False),
        Ticket(Path("c"), "t-3", "closed", "dead", is_step=False),
    ]
    assert orphaned_tickets(tickets, set()) == []


# --- release actions ---


def test_release_reopens_and_unassigns_a_regular_ticket() -> None:
    runner = FakeRunner()
    released = release_tickets(
        [Ticket(Path("a"), "t-1", "in_progress", "dead", is_step=False)], runner, dry_run=False
    )
    assert released == ["t-1"]
    assert ["tk", "reopen", "t-1"] in runner.commands
    assert ["tk", "unassign", "t-1"] in runner.commands


def test_release_closes_an_orphaned_step() -> None:
    runner = FakeRunner()
    released = release_tickets(
        [Ticket(Path("a"), "s-1", "in_progress", "dead", is_step=True)], runner, dry_run=False
    )
    assert released == ["s-1"]
    assert runner.commands[0][:3] == ["tk", "close", "s-1"]
    assert not any(cmd[:2] == ["tk", "reopen"] for cmd in runner.commands)


def test_release_reports_nothing_when_a_command_fails() -> None:
    runner = FakeRunner(failing={"tk"})
    released = release_tickets(
        [Ticket(Path("a"), "t-1", "in_progress", "dead", is_step=False)], runner, dry_run=False
    )
    assert released == []


def test_dry_run_changes_nothing() -> None:
    runner = FakeRunner()
    released = release_tickets(
        [Ticket(Path("a"), "t-1", "in_progress", "dead", is_step=False)], runner, dry_run=True
    )
    assert released == ["t-1"]
    assert runner.commands == []


# --- end to end ---


def test_reap_releases_a_dead_workers_agent_and_ticket_but_spares_live_ones(
    tmp_path: Path,
) -> None:
    agents_json = (
        '{"agents": ['
        '{"name": "update-self", "state": "STOPPED", "labels": {"agent_created": "true"}},'
        '{"name": "live-worker", "state": "RUNNING", "labels": {"agent_created": "true"}},'
        '{"name": "minds", "state": "RUNNING", "labels": {}}'
        "]}"
    )
    runner = FakeRunner(agents_json=agents_json)
    _write_ticket(tmp_path, "t-dead", status="in_progress", assignee="update-self")
    _write_ticket(tmp_path, "t-live", status="in_progress", assignee="live-worker")

    reaped, released = reap(tickets_dir=tmp_path, runner=runner, dry_run=False)

    assert reaped == ["update-self"]
    assert released == ["t-dead"]
    assert ["mngr", "destroy", "update-self", "--force"] in runner.commands
    assert not any("live-worker" in cmd for cmd in runner.commands if cmd[0] == "mngr" and "destroy" in cmd)


def test_reap_releases_nothing_when_agent_liveness_is_unknown(tmp_path: Path) -> None:
    """An unreadable agent list must abort, not read as "nothing is alive".

    Otherwise every in-progress ticket looks orphaned and the sweep strips
    tickets from agents that are very much running.
    """
    runner = FakeRunner(agents_json="not json")
    _write_ticket(tmp_path, "t-1", status="in_progress", assignee="minds")
    reaped, released = reap(tickets_dir=tmp_path, runner=runner, dry_run=False)
    assert reaped == []
    assert released == []
    assert not any(cmd[0] == "tk" for cmd in runner.commands)
