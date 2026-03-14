"""Utility modules for Claude Code model.

This package contains shared utility functions that are used throughout
the pydantic_ai_claude_code package.
"""

import os

from .file_utils import (
    copy_additional_files,
    get_next_call_subdirectory,
)
from .json_utils import strip_markdown_code_fence
from .type_utils import convert_primitive_value

# Keys that must be stripped from the subprocess environment to avoid
# the "nested session" check in the Claude CLI binary.
_NESTED_SESSION_VARS = ("CLAUDECODE",)


def clean_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Create a subprocess environment safe for spawning Claude CLI.

    Removes variables that trigger the nested-session guard in the CLI,
    then merges any caller-supplied overrides.
    """
    env = os.environ.copy()
    for key in _NESTED_SESSION_VARS:
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


__all__ = [
    "strip_markdown_code_fence",
    "convert_primitive_value",
    "copy_additional_files",
    "get_next_call_subdirectory",
    "clean_subprocess_env",
]
