"""Provider for Claude Code CLI model - infrastructure only.

This provider is stateless and handles ONLY infrastructure concerns:
- Finding/configuring the CLI binary
- Creating model instances via factory method

Model configuration, tools, and prompts are specified at Agent level.
Provider presets (deepseek, kimi) are part of the model string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .model import ClaudeCodeModel

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Provider for Claude Code CLI - infrastructure only.

    This provider is stateless and handles ONLY infrastructure concerns.
    Model configuration, tools, and prompts are specified at Agent level.
    Provider presets are part of the model string.

    Examples:
        >>> provider = ClaudeCodeProvider()
        >>> agent = Agent(model='claude-code:sonnet', ...)  # Uses Anthropic
        >>> agent2 = Agent(model='claude-code:deepseek:sonnet', ...)  # Uses DeepSeek

        # Or create model directly
        >>> model = provider.create_model('sonnet')
        >>> model_with_preset = provider.create_model('sonnet', provider_preset='deepseek')
    """

    _default_instance: ClassVar[ClaudeCodeProvider | None] = None

    def __init__(
        self,
        default_settings: dict | None = None,
        *,
        cli_path: str | Path | None = None,
    ):
        """Initialize provider with optional CLI path and default settings.

        Args:
            default_settings: Optional dict of default model settings
                (e.g., {"use_sandbox_runtime": False}) applied to all models
                created by this provider.
            cli_path: Path to claude CLI binary. If not provided, searches PATH.
        """
        self._cli_path = str(cli_path) if cli_path else None
        self._default_settings: dict = default_settings or {}

        # When created with custom settings, set as default for model string resolution
        if default_settings is not None or cli_path is not None:
            ClaudeCodeProvider._default_instance = self

        logger.debug(
            "Initialized ClaudeCodeProvider with cli_path=%s, default_settings=%s",
            self._cli_path,
            self._default_settings,
        )

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "claude-code"

    @property
    def cli_path(self) -> str | None:
        """Get configured CLI path."""
        return self._cli_path

    def create_model(
        self,
        model_name: str,
        *,
        provider_preset: str | None = None,
    ) -> ClaudeCodeModel:
        """Create a ClaudeCodeModel instance.

        Called by registration logic when parsing 'claude-code:*' strings.

        Args:
            model_name: Model alias (sonnet, opus, haiku) or full model name
            provider_preset: Optional preset ID (deepseek, kimi, etc.)

        Returns:
            Configured ClaudeCodeModel instance
        """
        # Import here to avoid circular dependency
        from .model import ClaudeCodeModel

        return ClaudeCodeModel(
            model_name=model_name,
            provider_preset=provider_preset,
            cli_path=self._cli_path,
            default_settings=self._default_settings,
        )

    def __repr__(self) -> str:
        """String representation."""
        parts = [f"cli_path={self._cli_path!r}"]
        if self._default_settings:
            parts.append(f"default_settings={self._default_settings!r}")
        return f"ClaudeCodeProvider({', '.join(parts)})"
