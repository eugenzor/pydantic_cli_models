"""Periodic ping coordination system for long-running multi-agent tasks.

Demonstrates real-time agent coordination using:
- OpenAI for lightweight 30-second status pings
- Shared state file for cross-agent communication
- Async event loop for non-blocking coordination

Usage:
    OPENAI_API_KEY=sk-... uv run python examples/ping_coordinator.py

The coordinator:
1. Spawns a background ping loop (every 30s) using OpenAI
2. Runs a primary task agent (Claude Code or OpenAI)
3. Both agents read/write a shared coordination state
4. Pings include task status summaries and health checks
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

# Coordination state shared between agents
COORD_STATE_FILE = Path(tempfile.gettempdir()) / "claude_coord_state.json"


class PingReport(BaseModel):
    """Status report from a ping cycle."""

    timestamp: str
    cycle: int
    status: str
    summary: str
    recommendations: list[str] = []


class CoordinationState:
    """Thread-safe shared state for multi-agent coordination."""

    def __init__(self, state_file: Path = COORD_STATE_FILE):
        self.state_file = state_file
        self._ensure_state()

    def _ensure_state(self) -> None:
        if not self.state_file.exists():
            self._write(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ping_count": 0,
                    "task_status": "idle",
                    "last_ping": None,
                    "agents_active": [],
                    "messages": [],
                    "task_log": [],
                }
            )

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            self._ensure_state()
            return json.loads(self.state_file.read_text())

    def _write(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def record_ping(self, agent_name: str, message: str) -> int:
        state = self._read()
        state["ping_count"] += 1
        state["last_ping"] = datetime.now(timezone.utc).isoformat()
        if agent_name not in state["agents_active"]:
            state["agents_active"].append(agent_name)
        state["messages"].append(
            {
                "agent": agent_name,
                "time": datetime.now(timezone.utc).isoformat(),
                "message": message,
                "cycle": state["ping_count"],
            }
        )
        # Keep only last 20 messages
        state["messages"] = state["messages"][-20:]
        self._write(state)
        return state["ping_count"]

    def update_task_status(self, status: str, detail: str = "") -> None:
        state = self._read()
        state["task_status"] = status
        state["task_log"].append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "detail": detail,
            }
        )
        state["task_log"] = state["task_log"][-50:]
        self._write(state)

    def get_summary(self) -> str:
        state = self._read()
        return (
            f"Pings: {state['ping_count']} | "
            f"Status: {state['task_status']} | "
            f"Agents: {', '.join(state['agents_active']) or 'none'} | "
            f"Last ping: {state.get('last_ping', 'never')}"
        )

    def deregister_agent(self, agent_name: str) -> None:
        state = self._read()
        if agent_name in state["agents_active"]:
            state["agents_active"].remove(agent_name)
        self._write(state)


async def ping_loop(
    coord: CoordinationState,
    interval: int = 30,
    max_cycles: int | None = None,
    use_ai: bool = True,
):
    """Background ping loop that reports status every `interval` seconds.

    Args:
        coord: Shared coordination state
        interval: Seconds between pings
        max_cycles: Stop after N cycles (None = run forever)
        use_ai: Whether to use OpenAI for intelligent summaries
    """
    agent_name = "ping-agent"
    cycle = 0

    # Create a lightweight OpenAI agent for status summaries
    ping_agent = None
    if use_ai:
        try:
            ping_agent = Agent(
                "openai:gpt-4o-mini",
                output_type=PingReport,
                system_prompt=(
                    "You are a coordination ping agent. Generate brief status reports. "
                    "Keep summaries under 50 words. Be concise and actionable."
                ),
            )
        except Exception as e:
            print(f"[ping] Could not create AI agent: {e}. Using simple pings.")
            ping_agent = None

    print(f"[ping] Starting ping loop (every {interval}s)")
    coord.record_ping(agent_name, "Ping loop started")

    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")

            if ping_agent:
                try:
                    state_summary = coord.get_summary()
                    result = await ping_agent.run(
                        f"Cycle {cycle} at {now}. Current state: {state_summary}. "
                        f"Generate a brief status ping report."
                    )
                    report = result.output
                    message = f"[{report.status}] {report.summary}"
                    if report.recommendations:
                        message += f" | Recs: {'; '.join(report.recommendations[:2])}"
                except Exception as e:
                    message = f"Cycle {cycle} - AI ping failed: {e!s:.80}"
            else:
                message = f"Cycle {cycle} - heartbeat OK at {now}"

            ping_num = coord.record_ping(agent_name, message)
            print(f"[ping #{ping_num}] {now} | {message}")

            await asyncio.sleep(interval)

    except asyncio.CancelledError:
        coord.record_ping(agent_name, "Ping loop cancelled - shutting down")
        coord.deregister_agent(agent_name)
        print("[ping] Loop cancelled, cleaning up")
    except KeyboardInterrupt:
        coord.deregister_agent(agent_name)
        print("[ping] Interrupted")


async def run_primary_task(
    coord: CoordinationState,
    task_description: str = "Analyze and report on the coordination system health",
    use_claude: bool = False,
):
    """Run a primary task agent alongside the ping loop.

    Args:
        coord: Shared coordination state
        task_description: What the primary agent should do
        use_claude: If True, use Claude Code; otherwise use OpenAI
    """
    agent_name = "primary-agent"
    coord.update_task_status("running", f"Starting: {task_description}")
    coord.record_ping(agent_name, f"Starting task: {task_description}")

    model = "claude-code:sonnet" if use_claude else "openai:gpt-4o-mini"

    try:
        agent = Agent(
            model,
            system_prompt=(
                "You are a task agent in a multi-agent coordination system. "
                "You have a ping agent monitoring your progress every 30 seconds. "
                "Complete the given task and provide a structured report."
            ),
        )

        print(f"[task] Running primary task with {model}...")
        result = await agent.run(task_description)
        output = result.output

        coord.update_task_status("completed", f"Result: {str(output)[:200]}")
        coord.record_ping(agent_name, f"Task completed: {str(output)[:100]}")
        print(f"[task] Completed: {str(output)[:200]}")
        return output

    except Exception as e:
        coord.update_task_status("failed", str(e))
        coord.record_ping(agent_name, f"Task failed: {e}")
        print(f"[task] Failed: {e}")
        return None
    finally:
        coord.deregister_agent(agent_name)


async def main():
    """Run the full coordination system: ping loop + primary task."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("  Agent Ping Coordination System")
    print("=" * 60)
    print(f"  State file: {COORD_STATE_FILE}")
    print("  Ping interval: 30s")
    print(f"  OpenAI key: ...{api_key[-8:]}")
    print("=" * 60)

    coord = CoordinationState()
    coord.update_task_status("initializing")

    # Run ping loop and primary task concurrently
    ping_task = asyncio.create_task(
        ping_loop(coord, interval=30, max_cycles=None, use_ai=True)
    )

    # Give ping loop a moment to start
    await asyncio.sleep(1)

    # Run a few primary tasks
    tasks = [
        "Summarize the health of this multi-agent coordination system in 3 bullet points.",
        "What are the top 3 best practices for long-running AI agent coordination?",
    ]

    for i, task_desc in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)} ---")
        await run_primary_task(coord, task_desc, use_claude=False)
        print(f"[coord] {coord.get_summary()}")

        if i < len(tasks):
            print("[coord] Waiting 30s before next task...")
            await asyncio.sleep(30)

    # Let the ping loop run a few more cycles for demo
    print("\n[coord] Tasks complete. Ping loop continues (Ctrl+C to stop)...")

    try:
        await asyncio.sleep(90)  # Run 3 more ping cycles
    except asyncio.CancelledError:
        pass

    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass

    print(f"\n[coord] Final state: {coord.get_summary()}")
    print(f"[coord] State file: {COORD_STATE_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[coord] Shutdown complete")
