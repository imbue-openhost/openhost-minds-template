"""Release worker/ticket state stranded by a container restart.

A container restart (an OpenHost redeploy, a crash) kills every agent but
leaves the workspace volume untouched, so state that records "someone is
working on this" outlives the worker that owned it. Two kinds of that state
actively block the next attempt:

* **Worker agent records.** ``create_worker.py launch`` runs ``mngr create
  <name>``, which fails outright when an agent of that name already exists --
  and a killed worker's record persists (workers are created with
  ``start_on_boot: false``, so they never come back on their own). Every future
  pass of that skill is then wedged until someone destroys it by hand.
* **Tickets left ``in_progress``.** Their assignee is gone, but single-flight
  checks (``tk ready`` greps for a live ticket) still read them as owned.

Both are released here, deterministically, with no model in the loop. The
liveness test is what makes it safe: nothing owned by a *running* agent is
touched, so this is equally correct on a service-only restart (where the mind
and its workers keep running) as on a full container restart.

Deliberately NOT cleaned: a leftover ``finish_report_path``. That file's
existence is evidence the worker *finished* -- it is how a lead that died
mid-``await`` recovers its worker's result on resume. Clearing it would turn a
recoverable pass into a silent timeout. ``launch``'s stale-report guard already
refuses in the genuinely-stale case and tells the caller how to archive it.

Runs once per container: the marker lives outside the persistent volume, so it
vanishes when the container is replaced and survives a service restart.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Outside /mngr (the persistent app-data volume) on purpose: a fresh container
# has no marker, a service restart within one container does.
DEFAULT_MARKER = Path("/tmp/minds-stale-worker-reap-done")
DEFAULT_TICKETS_DIR = Path("/mngr/code/runtime/tickets")

WORKER_LABEL = "agent_created"
RUNNING_STATE = "RUNNING"

RELEASE_NOTE = "Released automatically: the owning agent did not survive a workspace restart."
STEP_CLOSE_SUMMARY = "Interrupted by a workspace restart."


@dataclass(frozen=True)
class Ticket:
    """The frontmatter fields this script cares about."""

    path: Path
    id: str
    status: str
    assignee: str
    is_step: bool


class Runner:
    """Indirection over subprocess so the reap logic is unit-testable."""

    def run(self, cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, text=True, capture_output=True, check=check)


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse a ticket's leading ``---`` block into flat key/value strings."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def read_tickets(tickets_dir: Path) -> list[Ticket]:
    if not tickets_dir.is_dir():
        return []
    tickets: list[Ticket] = []
    for path in sorted(tickets_dir.glob("*.md")):
        try:
            fields = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        ticket_id = fields.get("id", "")
        if not ticket_id:
            continue
        tickets.append(
            Ticket(
                path=path,
                id=ticket_id,
                status=fields.get("status", ""),
                assignee=fields.get("assignee", ""),
                is_step=fields.get("step", "").lower() in ("true", "yes", "1"),
            )
        )
    return tickets


def list_agents(runner: Runner) -> list[dict] | None:
    """Return mngr's agent records, or None when they cannot be determined.

    None is distinct from an empty list and must stay that way: liveness is
    what protects running agents' state, so an unreadable agent list has to
    abort the whole pass rather than be read as "nothing is alive".
    """
    result = runner.run(["mngr", "ls", "--format", "json"])
    if result.returncode != 0:
        sys.stderr.write(f"reap: `mngr ls` failed: {result.stderr.strip()}\n")
        return None
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        sys.stderr.write("reap: `mngr ls` returned unparseable JSON\n")
        return None
    agents = payload.get("agents")
    return agents if isinstance(agents, list) else None


def live_agent_names(agents: list[dict]) -> set[str]:
    return {
        agent.get("name", "")
        for agent in agents
        if agent.get("state") == RUNNING_STATE and agent.get("name")
    }


def dead_worker_names(agents: list[dict]) -> list[str]:
    """Worker agents (agent_created=true) that are no longer running."""
    names: list[str] = []
    for agent in agents:
        labels = agent.get("labels") or {}
        if str(labels.get(WORKER_LABEL, "")).lower() != "true":
            continue
        if agent.get("state") == RUNNING_STATE:
            continue
        name = agent.get("name")
        if name:
            names.append(name)
    return names


def orphaned_tickets(tickets: list[Ticket], live: set[str]) -> list[Ticket]:
    """In-progress tickets whose assignee is not a currently running agent.

    An unassigned in-progress ticket is left alone: nothing identifies an owner
    to have died, and a human may be driving it.
    """
    return [
        ticket
        for ticket in tickets
        if ticket.status == "in_progress" and ticket.assignee and ticket.assignee not in live
    ]


def reap_agents(names: list[str], runner: Runner, *, dry_run: bool) -> list[str]:
    reaped: list[str] = []
    for name in names:
        print(f"reap: destroying stranded worker agent {name!r}")
        if dry_run:
            reaped.append(name)
            continue
        result = runner.run(["mngr", "destroy", name, "--force"])
        if result.returncode != 0:
            sys.stderr.write(f"reap: could not destroy {name!r}: {result.stderr.strip()}\n")
            continue
        reaped.append(name)
    return reaped


def release_tickets(tickets: list[Ticket], runner: Runner, *, dry_run: bool) -> list[str]:
    """Close orphaned steps; return orphaned regular tickets to the open pool.

    A step is turn-bound and its turn is over, so it terminates as closed. A
    regular ticket is still wanted work, so it is reopened and unassigned for
    whoever picks it up next.
    """
    released: list[str] = []
    for ticket in tickets:
        if ticket.is_step:
            print(f"reap: closing orphaned step {ticket.id} (owner {ticket.assignee!r} is gone)")
            commands = [["tk", "close", ticket.id, STEP_CLOSE_SUMMARY]]
        else:
            print(f"reap: reopening orphaned ticket {ticket.id} (owner {ticket.assignee!r} is gone)")
            commands = [
                ["tk", "reopen", ticket.id],
                ["tk", "unassign", ticket.id],
                ["tk", "add-note", ticket.id, RELEASE_NOTE],
            ]
        if dry_run:
            released.append(ticket.id)
            continue
        failed = False
        for cmd in commands:
            result = runner.run(cmd)
            if result.returncode != 0:
                sys.stderr.write(f"reap: `{' '.join(cmd)}` failed: {result.stderr.strip()}\n")
                failed = True
                break
        if not failed:
            released.append(ticket.id)
    return released


def reap(
    *,
    tickets_dir: Path,
    runner: Runner,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    agents = list_agents(runner)
    if agents is None:
        # Fail closed: without liveness we cannot tell a stranded owner from a
        # running one, and guessing would strip tickets from live agents.
        sys.stderr.write("reap: agent liveness unknown; releasing nothing\n")
        return [], []
    live = live_agent_names(agents)
    reaped = reap_agents(dead_worker_names(agents), runner, dry_run=dry_run)
    released = release_tickets(
        orphaned_tickets(read_tickets(tickets_dir), live), runner, dry_run=dry_run
    )
    return reaped, released


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets-dir", default=str(DEFAULT_TICKETS_DIR))
    parser.add_argument("--marker", default=str(DEFAULT_MARKER))
    parser.add_argument(
        "--once-per-container",
        action="store_true",
        help="No-op when the marker exists; write it after a successful pass",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without changing anything")
    args = parser.parse_args(argv)

    marker = Path(args.marker)
    if args.once_per_container and marker.exists():
        return 0

    reaped, released = reap(
        tickets_dir=Path(args.tickets_dir), runner=Runner(), dry_run=args.dry_run
    )
    if not reaped and not released:
        print("reap: nothing stranded")
    else:
        print(f"reap: released {len(reaped)} worker agent(s), {len(released)} ticket(s)")

    if args.once_per_container and not args.dry_run:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        except OSError as e:
            sys.stderr.write(f"reap: could not write marker {marker}: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
