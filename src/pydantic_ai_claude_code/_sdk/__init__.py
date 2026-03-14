"""SDK core types and utilities for Claude Code integration.

This module contains copied SDK types to avoid external dependencies
while maintaining compatibility with the Claude Agent SDK patterns.
"""

from .errors import (
    AuthenticationError,
    ClaudeSDKError,
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
    RateLimitError,
    TimeoutError,
)
from .types import (
    AssistantMessage,
    CanUseTool,
    # Options
    ClaudeAgentOptions,
    ContentBlock,
    HookConfig,
    HookEvent,
    # Hook types
    HookMatcher,
    Message,
    # Permission modes
    PermissionMode,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SDKResponse,
    # Usage and response
    SDKUsage,
    # Content blocks
    TextBlock,
    # Permission types
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    # Messages
    UserMessage,
)

__all__ = [
    # Permission modes
    "PermissionMode",
    # Content blocks
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "ResultMessage",
    "Message",
    # Permission types
    "ToolPermissionContext",
    "PermissionResultAllow",
    "PermissionResultDeny",
    "PermissionResult",
    "CanUseTool",
    # Hook types
    "HookMatcher",
    "HookEvent",
    "HookConfig",
    # Options
    "ClaudeAgentOptions",
    # Usage and response
    "SDKUsage",
    "SDKResponse",
    # Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "TimeoutError",
    "AuthenticationError",
    "RateLimitError",
]
