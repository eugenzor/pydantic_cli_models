"""Additional tests for supporting modules to increase coverage.

Covers:
- response_utils.py: extract_model_parameters, get_working_directory
- sdk_adapter.py: object-format extraction paths
- messages.py: binary content handling, extract_text_from_response, count functions
- transport/sdk_transport.py: retry paths, verbose response, error classification
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage


# ======================================================================
# response_utils.py
# ======================================================================


class TestResponseUtilsExtractModelParameters:
    """Tests for extract_model_parameters."""

    def test_with_none(self):
        from pydantic_ai_claude_code.response_utils import extract_model_parameters

        output_tools, function_tools = extract_model_parameters(None)
        assert output_tools == []
        assert function_tools == []

    def test_with_parameters(self):
        from pydantic_ai_claude_code.response_utils import extract_model_parameters

        tool = mock.MagicMock()
        out = mock.MagicMock()
        mrp = ModelRequestParameters(function_tools=[tool], output_tools=[out])

        output_tools, function_tools = extract_model_parameters(mrp)
        assert len(output_tools) == 1
        assert len(function_tools) == 1

    def test_with_empty_parameters(self):
        from pydantic_ai_claude_code.response_utils import extract_model_parameters

        mrp = ModelRequestParameters()
        output_tools, function_tools = extract_model_parameters(mrp)
        assert output_tools == []
        assert function_tools == []


class TestResponseUtilsGetWorkingDirectory:
    """Tests for get_working_directory."""

    def test_returns_from_settings(self):
        from pydantic_ai_claude_code.response_utils import get_working_directory

        settings = {"__working_directory": "/custom/path"}
        assert get_working_directory(settings) == "/custom/path"

    def test_returns_default(self):
        from pydantic_ai_claude_code.response_utils import get_working_directory

        settings = {}
        assert get_working_directory(settings) == "/tmp"

    def test_custom_default(self):
        from pydantic_ai_claude_code.response_utils import get_working_directory

        settings = {}
        assert get_working_directory(settings, default="/custom") == "/custom"


class TestResponseUtilsCreateToolCallPart:
    """Tests for create_tool_call_part."""

    def test_creates_with_unique_id(self):
        from pydantic_ai_claude_code.response_utils import create_tool_call_part

        part = create_tool_call_part("test_tool", {"key": "value"})
        assert part.tool_name == "test_tool"
        assert part.args == {"key": "value"}
        assert part.tool_call_id.startswith("call_")


# ======================================================================
# sdk_adapter.py
# ======================================================================


class TestSDKAdapterObjectFormat:
    """Tests for SDKAdapter with object-format messages (non-dict paths)."""

    def test_extract_assistant_content_object_string(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(type="assistant", content="Hello from object")
        result = adapter._extract_assistant_content(msg)
        assert result == "Hello from object"

    def test_extract_assistant_content_object_blocks(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        block1 = SimpleNamespace(type="text", text="Part 1")
        block2 = SimpleNamespace(text="Part 2")
        msg = SimpleNamespace(type="assistant", content=[block1, block2])

        result = adapter._extract_assistant_content(msg)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_extract_assistant_content_object_empty_iterable(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(type="assistant", content=[])
        result = adapter._extract_assistant_content(msg)
        assert result is None

    def test_extract_assistant_content_dict_string(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = {"type": "assistant", "content": "Hello string"}
        result = adapter._extract_assistant_content(msg)
        assert result == "Hello string"

    def test_extract_assistant_content_dict_blocks_with_string(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = {"type": "assistant", "content": ["raw text", {"type": "text", "text": "block text"}]}
        result = adapter._extract_assistant_content(msg)
        assert "raw text" in result
        assert "block text" in result

    def test_extract_result_content_object(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(type="result", result="Object result")
        result = adapter._extract_result_content(msg)
        assert result == "Object result"

    def test_extract_usage_object(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        msg = SimpleNamespace(type="result", result="test", usage=usage)

        result = adapter._extract_usage(msg)
        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_extract_usage_object_no_usage(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(type="result", result="test", usage=None)

        result = adapter._extract_usage(msg)
        assert result is None

    def test_extract_tool_call_object(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(
            type="tool_use",
            name="my_tool",
            input={"key": "val"},
            id="tool_123",
        )

        result = adapter._extract_tool_call(msg)
        assert result.tool_name == "my_tool"
        assert result.args == {"key": "val"}
        assert result.tool_call_id == "tool_123"

    def test_extract_tool_call_object_defaults(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        msg = SimpleNamespace(type="tool_use")

        result = adapter._extract_tool_call(msg)
        assert result.tool_name == "unknown"
        assert result.args == {}
        assert result.tool_call_id.startswith("call_")


class TestSDKAdapterSdkToModelResponse:
    """Tests for sdk_to_model_response with various message types."""

    def test_tool_use_message(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        messages = [
            {"type": "tool_use", "name": "search", "input": {"q": "test"}, "id": "t1"}
        ]
        response = adapter.sdk_to_model_response(messages)
        assert any(isinstance(p, ToolCallPart) for p in response.parts)

    def test_none_message_skipped(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        messages = [None, {"type": "result", "result": "ok"}]
        response = adapter.sdk_to_model_response(messages)
        assert len(response.parts) == 1

    def test_empty_messages_gets_empty_text(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = adapter.sdk_to_model_response([])
        assert len(response.parts) == 1
        assert response.parts[0].content == ""


class TestSDKAdapterModelResponseToDict:
    """Tests for model_response_to_dict."""

    def test_converts_text_parts(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = ModelResponse(
            parts=[TextPart(content="Hello")],
            model_name="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        result = adapter.model_response_to_dict(response)
        assert result["model_name"] == "test"
        assert result["parts"][0]["type"] == "text"

    def test_converts_tool_call_parts(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = ModelResponse(
            parts=[ToolCallPart(tool_name="test", args={"a": 1}, tool_call_id="c1")],
            model_name="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        result = adapter.model_response_to_dict(response)
        assert result["parts"][0]["type"] == "tool_call"
        assert result["parts"][0]["tool_name"] == "test"

    def test_includes_usage(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = ModelResponse(
            parts=[TextPart(content="Hi")],
            model_name="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

        result = adapter.model_response_to_dict(response)
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["total_tokens"] == 15

    def test_no_usage(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = ModelResponse(
            parts=[TextPart(content="Hi")],
            model_name="test",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        result = adapter.model_response_to_dict(response)
        # When no usage is provided, adapter still includes usage with zero defaults
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_no_timestamp(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        response = ModelResponse(
            parts=[TextPart(content="Hi")],
            model_name="test",
        )

        result = adapter.model_response_to_dict(response)
        # timestamp could be None or auto-set
        assert "timestamp" in result


class TestSDKAdapterGetAdapter:
    """Tests for get_adapter singleton."""

    def test_returns_same_instance(self):
        from pydantic_ai_claude_code import sdk_adapter

        # Reset singleton
        sdk_adapter._adapter = None

        a1 = sdk_adapter.get_adapter()
        a2 = sdk_adapter.get_adapter()
        assert a1 is a2

        # Cleanup
        sdk_adapter._adapter = None


class TestSDKAdapterMessagesToPrompt:
    """Tests for messages_to_prompt."""

    def test_includes_tool_return(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Hello"),
                ToolReturnPart(tool_name="search", content="results", tool_call_id="t1"),
            ])
        ]

        prompt = adapter.messages_to_prompt(messages)
        assert "User: Hello" in prompt
        assert "Tool Result (search): results" in prompt

    def test_includes_tool_call_in_response(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter

        adapter = SDKAdapter()
        messages = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="search", args={"q": "test"}, tool_call_id="t1")],
                model_name="test",
            )
        ]

        prompt = adapter.messages_to_prompt(messages)
        assert "Tool Call: search" in prompt

    def test_skip_system_prompt(self):
        from pydantic_ai_claude_code.sdk_adapter import SDKAdapter
        from pydantic_ai.messages import SystemPromptPart

        adapter = SDKAdapter()
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="System prompt"),
                UserPromptPart(content="Hello"),
            ])
        ]

        prompt = adapter.messages_to_prompt(messages, include_system=False)
        assert "System:" not in prompt
        assert "User: Hello" in prompt


# ======================================================================
# messages.py
# ======================================================================


class TestMessagesExtractText:
    """Tests for extract_text_from_response."""

    def test_strips_assistant_prefix(self):
        from pydantic_ai_claude_code.messages import extract_text_from_response

        assert extract_text_from_response("Assistant: Hello") == "Hello"

    def test_no_prefix(self):
        from pydantic_ai_claude_code.messages import extract_text_from_response

        assert extract_text_from_response("Hello world") == "Hello world"


class TestMessagesBinaryContent:
    """Tests for binary content handling in messages."""

    def test_create_binary_content_file(self):
        from pydantic_ai_claude_code.messages import _create_binary_content_file

        binary = BinaryContent(data=b"\x89PNG\r\n", media_type="image/png")
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = _create_binary_content_file(binary, 1, tmpdir)
            assert ref.startswith("@")
            # Check file was created
            files = list(Path(tmpdir).glob("*.png"))
            assert len(files) == 1

    def test_create_binary_content_file_with_identifier(self):
        from pydantic_ai_claude_code.messages import _create_binary_content_file

        binary = BinaryContent(
            data=b"data", media_type="image/jpeg", identifier="my-image"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = _create_binary_content_file(binary, 1, tmpdir)
            assert "my_image" in ref or "my-image" in ref

    def test_process_user_prompt_part_with_binary(self):
        from pydantic_ai_claude_code.messages import _process_user_prompt_part

        binary = BinaryContent(data=b"imagedata", media_type="image/png")
        part = UserPromptPart(content=["Look at this:", binary])

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt, counter = _process_user_prompt_part(part, 0, tmpdir)
            assert "Request:" in prompt
            assert "@" in prompt
            assert counter == 1

    def test_process_tool_return_binary_content(self):
        from pydantic_ai_claude_code.messages import _process_tool_return_part

        binary = BinaryContent(data=b"screenshot", media_type="image/png")
        part = ToolReturnPart(tool_name="screenshot", content=binary, tool_call_id="t1")

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt, tool_counter, binary_counter = _process_tool_return_part(
                part, None, 0, 0, tmpdir
            )
            assert prompt is not None
            assert "@" in prompt
            assert binary_counter == 1

    def test_process_tool_return_skips_when_binary_follows(self):
        from pydantic_ai_claude_code.messages import _process_tool_return_part

        part = ToolReturnPart(tool_name="tool", content="text result", tool_call_id="t1")
        # Next part has binary content
        binary = BinaryContent(data=b"img", media_type="image/png")
        next_part = UserPromptPart(content=["caption", binary])

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt, tool_counter, binary_counter = _process_tool_return_part(
                part, next_part, 0, 0, tmpdir
            )
            assert prompt is None  # skipped because binary follows


class TestMessagesCountFunctions:
    """Tests for _count_request_parts and _count_response_parts."""

    def test_count_request_parts(self):
        from pydantic_ai.messages import SystemPromptPart
        from pydantic_ai_claude_code.messages import _count_request_parts

        parts = [
            SystemPromptPart(content="System"),
            UserPromptPart(content="Hello"),
            UserPromptPart(content="World"),
            ToolReturnPart(tool_name="t", content="r", tool_call_id="1"),
        ]
        counts = _count_request_parts(parts)
        assert counts["has_system_prompt"] is True
        assert counts["num_user_messages"] == 2
        assert counts["num_tool_returns"] == 1

    def test_count_response_parts(self):
        from pydantic_ai_claude_code.messages import _count_response_parts

        parts = [
            TextPart(content="Hello"),
            TextPart(content="World"),
            ToolCallPart(tool_name="t", args={}, tool_call_id="1"),
        ]
        counts = _count_response_parts(parts)
        assert counts["num_assistant_messages"] == 2
        assert counts["num_tool_calls"] == 1


class TestMessagesBuildConversationContext:
    """Tests for build_conversation_context."""

    def test_builds_context(self):
        from pydantic_ai_claude_code.messages import build_conversation_context

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi")], model_name="test"),
        ]

        ctx = build_conversation_context(messages)
        assert ctx["num_messages"] == 2
        assert ctx["num_user_messages"] == 1
        assert ctx["num_assistant_messages"] == 1


class TestMessagesProcessResponseParts:
    """Tests for _process_response_parts."""

    def test_processes_text_parts(self):
        from pydantic_ai_claude_code.messages import _process_response_parts

        parts = [TextPart(content="Hello"), TextPart(content="World")]
        result = _process_response_parts(parts)
        assert len(result) == 2
        assert "Assistant: Hello" in result[0]

    def test_skips_tool_call_parts(self):
        from pydantic_ai_claude_code.messages import _process_response_parts

        parts = [
            TextPart(content="Hello"),
            ToolCallPart(tool_name="t", args={}, tool_call_id="1"),
        ]
        result = _process_response_parts(parts)
        assert len(result) == 1  # Only text part


# ======================================================================
# transport/sdk_transport.py
# ======================================================================


class TestEnhancedCLITransportClassifyError:
    """Tests for EnhancedCLITransport._classify_error."""

    def test_oauth_error_raises(self):
        from pydantic_ai_claude_code.exceptions import ClaudeOAuthError
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test prompt", {})
        stdout = '{"type":"result","is_error":true,"result":"OAuth token revoked · Please run /login"}'

        with pytest.raises(ClaudeOAuthError):
            transport._classify_error(stdout, "", 1, 5.0, True, "/tmp")

    def test_rate_limit_returns_retry(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        action, wait = transport._classify_error(
            "5-hour limit reached ∙ resets 3PM", "", 1, 5.0, True, "/tmp"
        )
        assert action == "retry_rate_limit"
        assert wait > 0

    def test_infra_failure_returns_retry(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        action, _ = transport._classify_error(
            "", "Cannot find module 'yoga'", 1, 5.0, True, "/tmp"
        )
        assert action == "retry_infra"

    def test_generic_error_raises(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        with pytest.raises(RuntimeError, match="Generic error"):
            transport._classify_error("", "Generic error", 1, 5.0, True, "/tmp")

    def test_long_runtime_error(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        with pytest.raises(RuntimeError, match="Long runtime"):
            transport._classify_error("", "timeout", 1, 700.0, False, "/tmp")


class TestEnhancedCLITransportProcessResponse:
    """Tests for EnhancedCLITransport._process_response."""

    def test_parses_simple_json(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        response = transport._process_response('{"type":"result","result":"hello"}')
        assert response["result"] == "hello"

    def test_strips_srt_diagnostic(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        raw = 'Running: /usr/bin/claude\n{"type":"result","result":"ok"}'
        response = transport._process_response(raw)
        assert response["result"] == "ok"

    def test_verbose_format(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        raw = '[{"type":"other"},{"type":"result","result":"verbose_ok"}]'
        response = transport._process_response(raw)
        assert response["result"] == "verbose_ok"

    def test_verbose_format_no_result_raises(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        with pytest.raises(RuntimeError, match="No result event"):
            transport._process_response('[{"type":"other"}]')

    def test_error_response_raises(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})
        with pytest.raises(RuntimeError, match="Bad things"):
            transport._process_response(
                '{"type":"result","is_error":true,"error":"Bad things"}'
            )


class TestEnhancedCLITransportSetupWorkingDirectory:
    """Tests for EnhancedCLITransport._setup_working_directory."""

    def test_creates_working_dir(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test prompt", {})
        cwd = transport._setup_working_directory()

        assert Path(cwd).exists()
        assert (Path(cwd) / "prompt.md").exists()
        assert transport.settings["__working_directory"] == cwd

    def test_reuses_existing_working_dir(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = EnhancedCLITransport(
                "test", {"__working_directory": tmpdir}
            )
            cwd = transport._setup_working_directory()
            assert cwd == tmpdir

    def test_copies_additional_files(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.txt"
            src.write_text("content")

            transport = EnhancedCLITransport(
                "test", {"additional_files": {"dest.txt": src}}
            )
            cwd = transport._setup_working_directory()
            assert (Path(cwd) / "dest.txt").exists()


class TestEnhancedCLITransportBuildCommand:
    """Tests for EnhancedCLITransport._build_command."""

    def test_basic_command(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with mock.patch(
            "pydantic_ai_claude_code.utils_legacy.resolve_claude_cli_path",
            return_value="/usr/bin/claude",
        ):
            transport = EnhancedCLITransport("test", {"model": "sonnet"})
            cmd = transport._build_command()

            assert cmd[0] == "/usr/bin/claude"
            assert "--print" in cmd
            assert "--model" in cmd
            assert "sonnet" in cmd

    def test_command_with_tools(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with mock.patch(
            "pydantic_ai_claude_code.utils_legacy.resolve_claude_cli_path",
            return_value="/usr/bin/claude",
        ):
            transport = EnhancedCLITransport("test", {
                "allowed_tools": ["Bash", "Read"],
                "disallowed_tools": ["WebFetch"],
            })
            cmd = transport._build_command()

            assert "--allowed-tools" in cmd
            assert "Bash" in cmd
            assert "--disallowed-tools" in cmd
            assert "WebFetch" in cmd

    def test_command_with_extra_args(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with mock.patch(
            "pydantic_ai_claude_code.utils_legacy.resolve_claude_cli_path",
            return_value="/usr/bin/claude",
        ):
            transport = EnhancedCLITransport("test", {
                "extra_cli_args": ["--verbose", "--debug"],
            })
            cmd = transport._build_command()

            assert "--verbose" in cmd
            assert "--debug" in cmd

    def test_command_with_session_id(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        with mock.patch(
            "pydantic_ai_claude_code.utils_legacy.resolve_claude_cli_path",
            return_value="/usr/bin/claude",
        ):
            transport = EnhancedCLITransport("test", {
                "session_id": "sess-123",
            })
            cmd = transport._build_command()

            assert "--session-id" in cmd
            assert "sess-123" in cmd


class TestConvertSettingsToSdkOptions:
    """Tests for convert_settings_to_sdk_options."""

    def test_maps_working_directory(self):
        from pydantic_ai_claude_code.transport.sdk_transport import (
            convert_settings_to_sdk_options,
        )

        settings = {"working_directory": "/work", "append_system_prompt": "sys prompt"}
        opts = convert_settings_to_sdk_options(settings)
        assert opts["cwd"] == "/work"
        assert opts["system_prompt"] == "sys prompt"

    def test_maps_permissions(self):
        from pydantic_ai_claude_code.transport.sdk_transport import (
            convert_settings_to_sdk_options,
        )

        settings = {
            "dangerously_skip_permissions": True,
            "allowed_tools": ["Bash"],
            "claude_cli_path": "/custom/claude",
        }
        opts = convert_settings_to_sdk_options(settings)
        assert opts["permission_mode"] == "acceptEdits"
        assert opts["allowed_tools"] == ["Bash"]
        assert opts["cli_path"] == "/custom/claude"

    def test_empty_settings(self):
        from pydantic_ai_claude_code.transport.sdk_transport import (
            convert_settings_to_sdk_options,
        )

        opts = convert_settings_to_sdk_options({})
        assert opts == {}


class TestEnhancedCLITransportExecuteCommand:
    """Tests for _execute_command."""

    @pytest.mark.asyncio
    async def test_timeout_raises_runtime_error(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {})

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = mock.AsyncMock()
        mock_process.wait = mock.AsyncMock()

        with mock.patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            with pytest.raises(RuntimeError, match="timeout"):
                await transport._execute_command(["claude"], "/tmp", 1)

    @pytest.mark.asyncio
    async def test_success_with_prompt(self):
        from pydantic_ai_claude_code.transport.sdk_transport import EnhancedCLITransport

        transport = EnhancedCLITransport("test", {"__prompt_text": "hello"})

        mock_process = mock.AsyncMock()
        mock_process.communicate = mock.AsyncMock(
            return_value=(b'{"type":"result"}', b"")
        )
        mock_process.returncode = 0

        with mock.patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            stdout, stderr, rc = await transport._execute_command(["claude"], "/tmp", 60)
            assert rc == 0

            # Verify prompt was passed
            call_args = mock_process.communicate.call_args
            assert call_args.kwargs.get("input") == b"hello"


# ======================================================================
# sdk_original_files/__init__.py
# ======================================================================


class TestSdkOriginalFiles:
    """Tests for sdk_original_files module."""

    def test_get_sdk_info(self):
        from pydantic_ai_claude_code.sdk_original_files import get_sdk_info

        info = get_sdk_info()
        assert "version" in info
        assert "last_import" in info
        assert "next_review" in info

    def test_sdk_constants(self):
        from pydantic_ai_claude_code.sdk_original_files import (
            SDK_AVAILABLE,
            SDK_VERSION,
        )

        assert isinstance(SDK_VERSION, str)
        assert isinstance(SDK_AVAILABLE, bool)


# ======================================================================
# structured_output.py
# ======================================================================


class TestStructuredOutputHandler:
    """Tests for StructuredOutputHandler."""

    def test_init(self):
        from pydantic_ai_claude_code.structured_output import StructuredOutputHandler

        handler = StructuredOutputHandler()
        assert handler.get_output_directory() is None

    def test_sdk_options_to_settings_none(self):
        from pydantic_ai_claude_code.structured_output import StructuredOutputHandler

        handler = StructuredOutputHandler()
        settings = handler._sdk_options_to_settings(None)
        assert settings == {}

    def test_sdk_options_to_settings_with_values(self):
        from pydantic_ai_claude_code.structured_output import StructuredOutputHandler

        handler = StructuredOutputHandler()
        opts = SimpleNamespace(
            cwd="/work",
            allowed_tools=["Bash"],
            disallowed_tools=["Write"],
            permission_mode="strict",
            model="sonnet",
        )
        settings = handler._sdk_options_to_settings(opts)
        assert settings["working_directory"] == "/work"
        assert settings["allowed_tools"] == ["Bash"]
        assert settings["disallowed_tools"] == ["Write"]
        assert settings["permission_mode"] == "strict"
        assert settings["model"] == "sonnet"

    def test_sdk_options_partial(self):
        from pydantic_ai_claude_code.structured_output import StructuredOutputHandler

        handler = StructuredOutputHandler()
        opts = SimpleNamespace(cwd="/work")
        settings = handler._sdk_options_to_settings(opts)
        assert settings == {"working_directory": "/work"}
