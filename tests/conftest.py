"""Shared test configuration and fixtures."""

import os

import pytest

# Skip integration tests when running inside a Claude Code session
# (nested sessions are not supported)
IN_CLAUDE_SESSION = os.environ.get("CLAUDECODE") is not None

requires_claude_cli = pytest.mark.skipif(
    IN_CLAUDE_SESSION,
    reason="Cannot run nested Claude Code sessions (CLAUDECODE env var is set)",
)
