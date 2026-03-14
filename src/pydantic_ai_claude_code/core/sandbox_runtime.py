"""Sandbox runtime support for Claude Code model.

This module provides sandbox runtime functionality that the Claude Agent SDK
doesn't have, allowing Claude to run in isolated environments with
restricted permissions.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


def resolve_sandbox_runtime_path(settings: dict[str, Any] | None = None) -> str:
    """Resolve path to sandbox-runtime (srt) binary.

    Resolution priority:
    1. sandbox_runtime_path from settings (if provided)
    2. SANDBOX_RUNTIME_PATH environment variable
    3. shutil.which('srt') - auto-resolve from PATH

    Args:
        settings: Optional settings containing sandbox_runtime_path

    Returns:
        Path to srt binary

    Raises:
        RuntimeError: If srt binary cannot be found
    """
    # Priority 1: Settings
    if settings and settings.get("sandbox_runtime_path"):
        srt_path = cast(str, settings["sandbox_runtime_path"])
        logger.debug("Using sandbox-runtime from settings: %s", srt_path)
        return srt_path

    # Priority 2: Environment variable
    env_path = os.environ.get("SANDBOX_RUNTIME_PATH")
    if env_path:
        logger.debug(
            "Using sandbox-runtime from SANDBOX_RUNTIME_PATH env var: %s", env_path
        )
        return env_path

    # Priority 3: Auto-resolve from PATH
    which_path = shutil.which("srt")
    if which_path:
        logger.debug("Auto-resolved sandbox-runtime from PATH: %s", which_path)
        return which_path

    # Not found
    logger.error("Could not find sandbox-runtime (srt) binary")
    raise RuntimeError(
        "Could not find sandbox-runtime (srt) binary. Please either:\n"
        "1. Install sandbox-runtime: npm install -g @anthropic-ai/sandbox-runtime\n"
        "2. Set sandbox_runtime_path in ClaudeCodeSettings\n"
        "3. Set SANDBOX_RUNTIME_PATH environment variable\n"
        "4. Add srt binary to your PATH"
    )


def build_sandbox_config() -> dict[str, Any]:
    """
    Create the sandbox configuration for the srt (sandbox-runtime) tool.

    The configuration uses the srt settings format with network and filesystem
    top-level keys to control domain access and file read/write permissions.

    Returns:
        sandbox_config (dict[str, Any]): A dict with "network" and "filesystem" keys
            compatible with the @anthropic-ai/sandbox-runtime config format.
    """
    return {
        "network": {
            "allowedDomains": [
                "api.anthropic.com",
                "*.anthropic.com",
            ],
            "deniedDomains": [],
        },
        "filesystem": {
            "denyRead": [],
            "allowWrite": ["/tmp"],
            "denyWrite": [],
        },
    }


def wrap_command_with_sandbox(
    cmd: list[str], settings: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, str], str]:
    """
    Wrap a Claude CLI command so it runs under the sandbox-runtime with a sandboxed configuration and environment.

    Parameters:
        cmd (list[str]): The original Claude CLI command and its arguments.
        settings (dict[str, Any] | None): Optional settings; if it contains `sandbox_runtime_path` that path is used to locate the sandbox runtime.

    Returns:
        tuple[list[str], dict[str, str], str]: A tuple containing:
            - wrapped_cmd: The wrapped command invoking the sandbox runtime
            - sandbox_env: Environment dictionary for the sandbox (includes `IS_SANDBOX` and `CLAUDE_CONFIG_DIR`)
            - config_path: Path to the temporary config file (caller must clean up after process exits)
    """
    settings = settings or {}
    srt_path = resolve_sandbox_runtime_path(settings)

    # Build sandbox config
    config = build_sandbox_config()

    # Write config to temp file
    config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="srt_config_")
    try:
        with os.fdopen(config_fd, "w") as f:
            json.dump(config, f)

        # Redirect Claude config/debug to a secure temp dir to avoid ~/.claude/ writes
        claude_config_dir = tempfile.mkdtemp(prefix="claude_sandbox_config_")
        os.chmod(claude_config_dir, 0o700)

        # Copy OAuth credentials from ~/.claude/ to sandbox config dir
        # This allows Claude to authenticate while keeping debug logs in /tmp
        home_claude_dir = Path.home() / ".claude"
        credentials_file = home_claude_dir / ".credentials.json"
        settings_file = home_claude_dir / "settings.json"

        if credentials_file.exists():
            shutil.copy2(
                credentials_file, Path(claude_config_dir) / ".credentials.json"
            )
            logger.debug("Copied credentials to sandbox config dir")

        if settings_file.exists():
            shutil.copy2(settings_file, Path(claude_config_dir) / "settings.json")
            logger.debug("Copied settings to sandbox config dir")

        # Build wrapper: srt -- <claude command>
        wrapped_cmd = [srt_path, "--settings", config_path, "--", *cmd]

        # Environment variables for sandbox
        sandbox_env = {
            "IS_SANDBOX": "1",
            "CLAUDE_CONFIG_DIR": claude_config_dir,
        }

        logger.info(
            "Wrapped Claude command with sandbox (IS_SANDBOX=1, CLAUDE_CONFIG_DIR=%s)",
            claude_config_dir,
        )
        logger.debug("Full sandboxed command: %s", " ".join(wrapped_cmd))

        return wrapped_cmd, sandbox_env, config_path

    except Exception:
        # Clean up config file on error
        try:
            os.unlink(config_path)
        except Exception:  # noqa: BLE001, S110
            pass
        raise
