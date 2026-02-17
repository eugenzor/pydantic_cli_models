"""Additional tests to increase model.py coverage.

Covers methods that are testable without a live Claude CLI:
- _build_function_tools_prompt
- _build_system_prompt_parts
- _assemble_final_prompt
- _handle_function_selection_followup (routing logic)
- _build_argument_collection_instruction
- _build_retry_prompt
- _handle_structured_output_response (various file paths)
- _convert_response (routing)
- request_stream ValueError paths
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_claude_code.model import ClaudeCodeModel
from pydantic_ai_claude_code.types import ClaudeCodeSettings, ClaudeJSONResponse


# ===== Helpers =====


def _make_mock_tool(name="get_weather", desc="Get weather", schema=None):
    """Create a mock function tool."""
    tool = mock.MagicMock()
    tool.name = name
    tool.description = desc
    tool.parameters_json_schema = schema or {
        "properties": {"city": {"type": "string"}},
    }
    return tool


def _make_mock_output_tool(name="final_answer", schema=None):
    """Create a mock output tool."""
    tool = mock.MagicMock()
    tool.name = name
    tool.parameters_json_schema = schema or {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    return tool


def _base_response(**overrides) -> ClaudeJSONResponse:
    """Create a minimal ClaudeJSONResponse."""
    r: ClaudeJSONResponse = {"type": "result", "result": "", "is_error": False}
    r.update(overrides)
    return r


# ===== _build_function_tools_prompt =====


class TestBuildFunctionToolsPrompt:
    """Tests for _build_function_tools_prompt."""

    def test_returns_prompt_and_available_functions(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        prompt, available = model._build_function_tools_prompt([tool])

        assert "get_weather" in prompt
        assert "Function Selection Task" in prompt
        assert "CHOICE:" in prompt
        assert "get_weather" in available
        assert available["get_weather"] is tool

    def test_multiple_tools(self):
        model = ClaudeCodeModel("sonnet")
        t1 = _make_mock_tool("tool_a", "Tool A")
        t2 = _make_mock_tool("tool_b", "Tool B")
        prompt, available = model._build_function_tools_prompt([t1, t2])

        assert "tool_a" in prompt
        assert "tool_b" in prompt
        assert "3. none" in prompt  # 2 tools + none option
        assert len(available) == 2

    def test_tool_with_no_description(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool(desc=None)
        prompt, _ = model._build_function_tools_prompt([tool])
        assert "No description" in prompt

    def test_tool_with_xml_description(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool(desc="<summary>Fetch data</summary>")
        prompt, _ = model._build_function_tools_prompt([tool])
        assert "Fetch data" in prompt


# ===== _build_system_prompt_parts =====


class TestBuildSystemPromptParts:
    """Tests for _build_system_prompt_parts."""

    def test_with_tool_results_adds_synthesis_instruction(self):
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            ModelRequestParameters(), has_tool_results=True, settings=settings
        )
        assert any("Synthesize Tool Results" in p for p in parts)

    def test_with_function_tools_adds_selection_prompt(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        mrp = ModelRequestParameters(function_tools=[tool])
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            mrp, has_tool_results=False, settings=settings
        )
        assert any("Function Selection Task" in p for p in parts)
        assert settings.get("__function_selection_mode__") is True

    def test_function_tools_skipped_when_tool_results_present(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        mrp = ModelRequestParameters(function_tools=[tool])
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            mrp, has_tool_results=True, settings=settings
        )
        assert not any("Function Selection Task" in p for p in parts)

    def test_output_tools_adds_structured_instruction(self):
        model = ClaudeCodeModel("sonnet")
        out_tool = _make_mock_output_tool()
        mrp = ModelRequestParameters(output_tools=[out_tool])
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            mrp, has_tool_results=False, settings=settings
        )
        assert any("mkdir" in p or "File Structure" in p for p in parts)

    def test_no_output_tools_adds_unstructured_instruction(self):
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            ModelRequestParameters(),
            has_tool_results=False,
            settings=settings,
        )
        assert any("Output Instructions" in p for p in parts)

    def test_streaming_mode_skips_output_instructions(self):
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            ModelRequestParameters(),
            has_tool_results=False,
            settings=settings,
            is_streaming=True,
        )
        assert not any("Output Instructions" in p for p in parts)

    def test_with_tool_results_adds_output_instructions(self):
        model = ClaudeCodeModel("sonnet")
        out_tool = _make_mock_output_tool()
        mrp = ModelRequestParameters(output_tools=[out_tool])
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        parts = model._build_system_prompt_parts(
            mrp, has_tool_results=True, settings=settings
        )
        # Should have both synthesis + structured output
        assert any("Synthesize Tool Results" in p for p in parts)
        assert any("mkdir" in p or "File Structure" in p for p in parts)


# ===== _assemble_final_prompt =====


class TestAssembleFinalPrompt:
    """Tests for _assemble_final_prompt."""

    def test_writes_user_request_file(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {"__working_directory": tmpdir}
            messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

            prompt = model._assemble_final_prompt(
                messages, ["System instruction"], settings, has_tool_results=False
            )

            assert "System instruction" in prompt
            req_file = Path(tmpdir) / "user_request.md"
            assert req_file.exists()
            assert "Hello" in req_file.read_text()

    def test_prepends_append_system_prompt(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {
                "__working_directory": tmpdir,
                "append_system_prompt": "Custom prefix",
            }
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]

            prompt = model._assemble_final_prompt(
                messages, ["Main instruction"], settings, has_tool_results=False
            )

            assert prompt.startswith("Custom prefix")
            assert "Main instruction" in prompt
            # append_system_prompt should be consumed
            assert "append_system_prompt" not in settings

    def test_empty_system_prompt_parts(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {"__working_directory": tmpdir}
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]

            prompt = model._assemble_final_prompt(
                messages, [], settings, has_tool_results=False
            )
            # Should return empty or near-empty prompt
            assert isinstance(prompt, str)


# ===== _handle_function_selection_followup =====


class TestHandleFunctionSelectionFollowup:
    """Tests for _handle_function_selection_followup routing logic."""

    @pytest.mark.asyncio
    async def test_returns_result_when_no_function_tools(self):
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="direct")])

        out = await model._handle_function_selection_followup(
            [], ModelRequestParameters(), settings, response, result
        )
        assert out is result

    @pytest.mark.asyncio
    async def test_returns_result_when_no_selection_mode(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        settings: ClaudeCodeSettings = {}  # no __function_selection_mode__
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="direct")])

        out = await model._handle_function_selection_followup(
            [], ModelRequestParameters(function_tools=[tool]),
            settings, response, result,
        )
        assert out is result

    @pytest.mark.asyncio
    async def test_routes_none_selection_to_unstructured(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__function_selection_result__": "none",
        }
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="placeholder")])

        with mock.patch.object(
            model, "_handle_unstructured_follow_up",
            return_value=ModelResponse(parts=[TextPart(content="unstructured")]),
        ) as mock_follow_up:
            out = await model._handle_function_selection_followup(
                [], ModelRequestParameters(function_tools=[tool]),
                settings, response, result,
            )
            mock_follow_up.assert_awaited_once()
            assert out.parts[0].content == "unstructured"

    @pytest.mark.asyncio
    async def test_routes_none_selection_to_structured(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        out_tool = _make_mock_output_tool()
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__function_selection_result__": "none",
        }
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="placeholder")])

        with mock.patch.object(
            model, "_handle_structured_follow_up",
            return_value=ModelResponse(parts=[TextPart(content="structured")]),
        ) as mock_follow_up:
            out = await model._handle_function_selection_followup(
                [],
                ModelRequestParameters(function_tools=[tool], output_tools=[out_tool]),
                settings, response, result,
            )
            mock_follow_up.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_selected_to_argument_collection(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__function_selection_result__": "selected",
            "__selected_function__": "get_weather",
            "__available_functions__": {"get_weather": tool},
        }
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="placeholder")])

        with mock.patch.object(
            model, "_handle_argument_collection",
            return_value=ModelResponse(parts=[TextPart(content="args collected")]),
        ) as mock_collect:
            out = await model._handle_function_selection_followup(
                [],
                ModelRequestParameters(function_tools=[tool]),
                settings, response, result,
            )
            mock_collect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_through_when_no_selection_result(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            # no __function_selection_result__
        }
        response = _base_response()
        result = ModelResponse(parts=[TextPart(content="original")])

        out = await model._handle_function_selection_followup(
            [], ModelRequestParameters(function_tools=[tool]),
            settings, response, result,
        )
        assert out is result


# ===== _build_argument_collection_instruction =====


class TestBuildArgumentCollectionInstruction:
    """Tests for _build_argument_collection_instruction."""

    def test_produces_instruction_string(self):
        model = ClaudeCodeModel("sonnet")
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {"__working_directory": tmpdir}
            instruction = model._build_argument_collection_instruction(
                schema, settings, "get_weather", "Get weather for a city"
            )

            assert "city" in instruction
            assert "units" in instruction
            assert "__structured_output_file" in settings
            assert "__temp_json_dir" in settings


# ===== _build_retry_prompt =====


class TestBuildRetryPrompt:
    """Tests for _build_retry_prompt."""

    def test_includes_error_message(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            settings: ClaudeCodeSettings = {
                "__working_directory": tmpdir,
                "__tool_name": "get_weather",
                "__tool_description": "Get weather",
            }
            schema = {"type": "object", "properties": {"city": {"type": "string"}}}
            messages = [ModelRequest(parts=[UserPromptPart(content="Weather in NYC")])]

            prompt = model._build_retry_prompt(
                messages, schema, settings, "Missing required field: city"
            )

            assert "PREVIOUS ATTEMPT HAD ERRORS" in prompt
            assert "Missing required field: city" in prompt
            assert "city" in prompt


# ===== _handle_structured_output_response =====


class TestHandleStructuredOutputResponse:
    """Tests for _handle_structured_output_response with various file states."""

    def test_reads_from_temp_json_dir(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure files
            (Path(tmpdir) / "answer.txt").write_text("42")
            (Path(tmpdir) / ".complete").touch()

            settings: ClaudeCodeSettings = {
                "__structured_output_file": "/nonexistent.json",
                "__temp_json_dir": tmpdir,
            }

            result = model._handle_structured_output_response(
                "raw text", response, [out_tool], settings
            )
            assert any(isinstance(p, ToolCallPart) for p in result.parts)

    def test_falls_back_to_json_file(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "output.json"
            json_file.write_text('{"answer": "hello"}')

            settings: ClaudeCodeSettings = {
                "__structured_output_file": str(json_file),
            }

            result = model._handle_structured_output_response(
                "raw text", response, [out_tool], settings
            )
            assert any(isinstance(p, ToolCallPart) for p in result.parts)

    def test_falls_back_to_text_extraction(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool()

        settings: ClaudeCodeSettings = {
            "__structured_output_file": "/nonexistent.json",
        }

        result = model._handle_structured_output_response(
            '{"answer": "extracted"}', response, [out_tool], settings
        )
        assert any(isinstance(p, ToolCallPart) for p in result.parts)

    def test_no_structured_file_falls_back_to_text(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool()

        settings: ClaudeCodeSettings = {}

        result = model._handle_structured_output_response(
            '{"answer": "from text"}', response, [out_tool], settings
        )
        assert any(isinstance(p, ToolCallPart) for p in result.parts)

    def test_json_decode_error_returns_text(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool(schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        })
        settings: ClaudeCodeSettings = {}

        result = model._handle_structured_output_response(
            "totally not json at all", response, [out_tool], settings
        )
        assert any(isinstance(p, TextPart) for p in result.parts)

    def test_validation_error_from_temp_dir(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        out_tool = _make_mock_output_tool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid data
            (Path(tmpdir) / "answer.txt").write_text("not_json_valid")
            (Path(tmpdir) / ".complete").touch()

            # Schema expects more fields
            out_tool.parameters_json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "integer"},
                },
                "required": ["answer"],
            }

            settings: ClaudeCodeSettings = {
                "__structured_output_file": "/nonexistent.json",
                "__temp_json_dir": tmpdir,
            }

            result = model._handle_structured_output_response(
                "raw text", response, [out_tool], settings
            )
            # Should return error as TextPart
            assert any(isinstance(p, TextPart) for p in result.parts)


# ===== _convert_response routing =====


class TestConvertResponseRouting:
    """Tests for _convert_response dispatch logic."""

    def test_routes_to_function_selection_mode(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response(result="CHOICE: none")
        tool = _make_mock_tool()
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__available_functions__": {"get_weather": tool},
        }

        result = model._convert_response(
            response, function_tools=[tool], settings=settings
        )
        assert settings.get("__function_selection_result__") == "none"

    def test_routes_to_structured_output(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response(result='{"answer": "hello"}')
        out_tool = _make_mock_output_tool()

        result = model._convert_response(
            response, output_tools=[out_tool], settings={}
        )
        assert any(isinstance(p, ToolCallPart) for p in result.parts)

    def test_routes_to_unstructured_output(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response(result="Hello world")

        result = model._convert_response(response, settings={})
        assert any(isinstance(p, TextPart) for p in result.parts)


# ===== request_stream ValueError paths =====


class TestRequestStreamErrors:
    """Tests for request_stream validation errors."""

    @pytest.mark.asyncio
    async def test_stream_rejects_output_tools(self):
        model = ClaudeCodeModel("sonnet")
        out_tool = _make_mock_output_tool()
        mrp = ModelRequestParameters(output_tools=[out_tool])
        messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]

        with pytest.raises(ValueError, match="Streaming is not supported with structured output"):
            async with model.request_stream(messages, None, mrp):
                pass

    @pytest.mark.asyncio
    async def test_stream_rejects_function_tools(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        mrp = ModelRequestParameters(function_tools=[tool])
        messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]

        with pytest.raises(ValueError, match="Streaming is not supported with function tools"):
            async with model.request_stream(messages, None, mrp):
                pass


# ===== _read_structured_output_file additional paths =====


class TestReadStructuredOutputFileEdgeCases:
    """Additional tests for _read_structured_output_file."""

    def test_reads_from_temp_json_dir_success(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "name.txt").write_text("Alice")
            (Path(tmpdir) / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
            settings: ClaudeCodeSettings = {"__temp_json_dir": tmpdir}

            data, error = model._read_structured_output_file(
                "/nonexistent.json", schema, settings
            )
            assert error is None
            assert data == {"name": "Alice"}

    def test_reads_from_temp_json_dir_runtime_error(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create incomplete structure (missing required)
            (Path(tmpdir) / ".complete").touch()

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
            settings: ClaudeCodeSettings = {"__temp_json_dir": tmpdir}

            data, error = model._read_structured_output_file(
                "/nonexistent.json", schema, settings
            )
            assert data is None
            assert error is not None
            assert "Missing" in error

    def test_validation_error_on_json_file(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "output.json"
            json_file.write_text('{"age": "not_an_int"}')

            schema = {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            }

            data, error = model._read_structured_output_file(str(json_file), schema)
            assert data is None
            assert error is not None
            assert "integer" in error

    def test_file_read_exception(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory where a file is expected (will cause read error)
            bad_path = Path(tmpdir) / "output.json"
            bad_path.mkdir()

            schema = {"type": "object", "properties": {}}

            data, error = model._read_structured_output_file(str(bad_path), schema)
            assert data is None
            assert error is not None
            assert "Failed to read" in error or "directory" in error.lower()


# ===== _handle_unstructured_output_response edge cases =====


class TestHandleUnstructuredOutputEdgeCases:
    """Edge cases for _handle_unstructured_output_response."""

    def test_no_settings(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response(result="fallback text")

        result = model._handle_unstructured_output_response(
            "fallback text", response, None
        )
        assert result.parts[0].content == "fallback text"

    def test_file_read_error_falls_back(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory where a file is expected
            bad_file = Path(tmpdir) / "output.txt"
            bad_file.mkdir()

            settings: ClaudeCodeSettings = {
                "__unstructured_output_file": str(bad_file),
            }

            result = model._handle_unstructured_output_response(
                "fallback", response, settings
            )
            assert result.parts[0].content == "fallback"


# ===== _handle_function_selection_response edge cases =====


class TestHandleFunctionSelectionEdgeCases:
    """Edge cases for _handle_function_selection_response."""

    def test_unparseable_choice(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        settings: ClaudeCodeSettings = {
            "__available_functions__": {"get_weather": mock.MagicMock()},
        }

        result = model._handle_function_selection_response(
            "I'm not sure which to pick", response, settings
        )
        assert any("Could not parse" in p.content for p in result.parts if isinstance(p, TextPart))

    def test_available_functions_not_dict(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        settings: ClaudeCodeSettings = {
            "__available_functions__": "not a dict",
        }

        result = model._handle_function_selection_response(
            "CHOICE: none", response, settings
        )
        # Should still handle gracefully
        assert len(result.parts) > 0


# ===== _create_usage edge cases =====


class TestCreateUsageEdgeCases:
    """Edge cases for _create_usage."""

    def test_usage_not_dict(self):
        response = _base_response(usage="not a dict")
        usage = ClaudeCodeModel._create_usage(response)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_server_tool_use_not_dict(self):
        response = _base_response(
            usage={"input_tokens": 10, "output_tokens": 5, "server_tool_use": "bad"}
        )
        usage = ClaudeCodeModel._create_usage(response)
        assert usage.input_tokens == 10
        assert usage.details["web_search_requests"] == 0

    def test_no_usage_key(self):
        response = _base_response()
        usage = ClaudeCodeModel._create_usage(response)
        assert usage.input_tokens == 0


# ===== _extract_json_robust edge cases =====


class TestExtractJsonRobustEdgeCases:
    """Edge cases for _extract_json_robust."""

    def test_quoted_string_autowrap(self):
        model = ClaudeCodeModel("sonnet")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = model._extract_json_robust('"hello world"', schema)
        assert result == {"name": "hello world"}

    def test_single_quoted_string_autowrap(self):
        model = ClaudeCodeModel("sonnet")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = model._extract_json_robust("'hello world'", schema)
        assert result == {"name": "hello world"}

    def test_array_match_skipped_for_multi_field_schema(self):
        model = ClaudeCodeModel("sonnet")
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        }

        with pytest.raises(json.JSONDecodeError):
            model._extract_json_robust("[1, 2, 3]", schema)

    def test_json_object_match_with_decode_error(self):
        """Test that invalid JSON in braces continues to next match."""
        model = ClaudeCodeModel("sonnet")
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        # First brace match is invalid, second is valid
        text = 'Some {invalid json here} and then {"x": "good"}'
        result = model._extract_json_robust(text, schema)
        assert result == {"x": "good"}

    def test_array_match_json_decode_error(self):
        """Test that invalid array JSON continues to next match."""
        model = ClaudeCodeModel("sonnet")
        schema = {"type": "object", "properties": {"items": {"type": "array"}}}
        # Invalid array followed by valid
        text = "[not valid] and [1, 2, 3]"
        result = model._extract_json_robust(text, schema)
        assert result == {"items": [1, 2, 3]}


# ===== _build_options additional paths =====


class TestBuildOptionsSettings:
    """Tests for _build_options with various settings keys."""

    def test_cli_path_included(self):
        model = ClaudeCodeModel("sonnet")
        model._cli_path = "/usr/local/bin/claude"
        settings = model._build_options(None, ModelRequestParameters())
        assert settings.get("claude_cli_path") == "/usr/local/bin/claude"

    def test_preset_env_vars_included(self):
        model = ClaudeCodeModel("sonnet")
        model._preset_env_vars = {"ANTHROPIC_API_KEY": "test-key"}
        settings = model._build_options(None, ModelRequestParameters())
        assert settings.get("__provider_env") == {"ANTHROPIC_API_KEY": "test-key"}

    def test_model_settings_additional_files(self):
        model = ClaudeCodeModel("sonnet")
        ms = {"additional_files": ["/tmp/extra.py"]}
        settings = model._build_options(ms, ModelRequestParameters())
        assert settings.get("additional_files") == ["/tmp/extra.py"]

    def test_model_settings_append_system_prompt(self):
        model = ClaudeCodeModel("sonnet")
        ms = {"append_system_prompt": "Be concise"}
        settings = model._build_options(ms, ModelRequestParameters())
        assert settings.get("append_system_prompt") == "Be concise"

    def test_model_settings_verbose(self):
        model = ClaudeCodeModel("sonnet")
        ms = {"verbose": True}
        settings = model._build_options(ms, ModelRequestParameters())
        assert settings.get("verbose") is True

    def test_model_settings_extra_cli_args(self):
        model = ClaudeCodeModel("sonnet")
        ms = {"extra_cli_args": ["--no-cache"]}
        settings = model._build_options(ms, ModelRequestParameters())
        assert settings.get("extra_cli_args") == ["--no-cache"]

    def test_system_prompt_on_model_request_parameters(self):
        """Test that system_prompt attribute on MRP is included in parts."""
        model = ClaudeCodeModel("sonnet")
        settings: ClaudeCodeSettings = {}
        model._prepare_working_directory(settings)

        mrp = ModelRequestParameters()
        # Attach system_prompt attribute
        mrp.system_prompt = "Custom system prompt"

        parts = model._build_system_prompt_parts(
            mrp, has_tool_results=False, settings=settings
        )
        assert any("Custom system prompt" in p for p in parts)


# ===== request() with mocked run_claude_async =====


class TestRequestMethod:
    """Tests for request() method with mocked CLI calls."""

    @pytest.mark.asyncio
    async def test_request_basic_flow(self):
        model = ClaudeCodeModel("sonnet")
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        mrp = ModelRequestParameters()

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": "Hello back!",
            "is_error": False,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            return_value=mock_response,
        ):
            result = await model.request(messages, None, mrp)

        assert any(isinstance(p, TextPart) for p in result.parts)
        assert "Hello back" in result.parts[0].content

    @pytest.mark.asyncio
    async def test_request_with_function_tools(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        messages = [ModelRequest(parts=[UserPromptPart(content="Weather?")])]
        mrp = ModelRequestParameters(function_tools=[tool])

        # First call: function selection
        selection_response: ClaudeJSONResponse = {
            "type": "result",
            "result": "CHOICE: 1. get_weather",
            "is_error": False,
        }
        # Second call: argument collection
        arg_response: ClaudeJSONResponse = {
            "type": "result",
            "result": '{"city": "NYC"}',
            "is_error": False,
        }

        call_count = 0

        async def mock_run_claude(prompt, settings=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return selection_response
            return arg_response

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            side_effect=mock_run_claude,
        ):
            result = await model.request(messages, None, mrp)

        assert any(
            isinstance(p, (ToolCallPart, TextPart)) for p in result.parts
        )


# ===== _handle_structured_follow_up =====


class TestHandleStructuredFollowUp:
    """Tests for _handle_structured_follow_up."""

    @pytest.mark.asyncio
    async def test_basic_flow(self):
        model = ClaudeCodeModel("sonnet")
        messages = [ModelRequest(parts=[UserPromptPart(content="Give answer")])]
        out_tool = _make_mock_output_tool()
        mrp = ModelRequestParameters(output_tools=[out_tool])

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": '{"answer": "42"}',
            "is_error": False,
        }

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            return_value=mock_response,
        ):
            result = await model._handle_structured_follow_up(
                messages, mrp, {"timeout_seconds": 60}
            )

        assert any(isinstance(p, ToolCallPart) for p in result.parts)

    @pytest.mark.asyncio
    async def test_preserves_original_settings(self):
        model = ClaudeCodeModel("sonnet")
        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
        mrp = ModelRequestParameters()

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": "text output",
            "is_error": False,
        }

        captured_settings = {}

        async def mock_run(prompt, settings=None, **kwargs):
            captured_settings.update(settings or {})
            return mock_response

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            side_effect=mock_run,
        ):
            await model._handle_structured_follow_up(
                messages,
                mrp,
                {"additional_files": ["/tmp/f.py"], "timeout_seconds": 120},
            )

        assert captured_settings.get("additional_files") == ["/tmp/f.py"]
        assert captured_settings.get("timeout_seconds") == 120


# ===== _handle_unstructured_follow_up =====


class TestHandleUnstructuredFollowUp:
    """Tests for _handle_unstructured_follow_up."""

    @pytest.mark.asyncio
    async def test_basic_flow(self):
        model = ClaudeCodeModel("sonnet")
        messages = [ModelRequest(parts=[UserPromptPart(content="Answer")])]
        mrp = ModelRequestParameters()

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": "The answer is 42",
            "is_error": False,
        }

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            return_value=mock_response,
        ):
            result = await model._handle_unstructured_follow_up(messages, mrp)

        assert any(isinstance(p, TextPart) for p in result.parts)

    @pytest.mark.asyncio
    async def test_preserves_original_settings(self):
        model = ClaudeCodeModel("sonnet")
        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
        mrp = ModelRequestParameters()

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": "response",
            "is_error": False,
        }

        captured_settings = {}

        async def mock_run(prompt, settings=None, **kwargs):
            captured_settings.update(settings or {})
            return mock_response

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            side_effect=mock_run,
        ):
            await model._handle_unstructured_follow_up(
                messages,
                mrp,
                {"debug_save_prompts": True, "timeout_seconds": 30},
            )

        assert captured_settings.get("debug_save_prompts") is True
        assert captured_settings.get("timeout_seconds") == 30


# ===== _handle_argument_collection =====


class TestHandleArgumentCollection:
    """Tests for _handle_argument_collection."""

    @pytest.mark.asyncio
    async def test_function_not_found(self):
        model = ClaudeCodeModel("sonnet")
        response = _base_response()
        result = await model._handle_argument_collection(
            [ModelRequest(parts=[UserPromptPart(content="test")])],
            "nonexistent_func",
            {"get_weather": _make_mock_tool()},
            response,
        )
        assert any("not found" in p.content for p in result.parts if isinstance(p, TextPart))

    @pytest.mark.asyncio
    async def test_successful_collection(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        messages = [ModelRequest(parts=[UserPromptPart(content="Weather in NYC")])]

        mock_response: ClaudeJSONResponse = {
            "type": "result",
            "result": '{"city": "NYC"}',
            "is_error": False,
        }

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            return_value=mock_response,
        ):
            result = await model._handle_argument_collection(
                messages,
                "get_weather",
                {"get_weather": tool},
                mock_response,
            )

        # Should have a tool call or text result
        assert len(result.parts) > 0

    @pytest.mark.asyncio
    async def test_collection_with_retry_on_error(self):
        model = ClaudeCodeModel("sonnet")
        tool = _make_mock_tool()
        messages = [ModelRequest(parts=[UserPromptPart(content="Weather?")])]

        call_count = 0

        async def mock_run(prompt, settings=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "type": "result",
                "result": "not parseable json",
                "is_error": False,
            }

        with mock.patch(
            "pydantic_ai_claude_code.model.run_claude_async",
            side_effect=mock_run,
        ):
            result = await model._handle_argument_collection(
                messages,
                "get_weather",
                {"get_weather": tool},
                _base_response(),
            )

        # Should have attempted retries
        assert call_count >= 1
        assert len(result.parts) > 0


# ===== _read_structured_output_file generic Exception path =====


class TestReadStructuredOutputGenericException:
    """Test the generic Exception fallback in _read_structured_output_file."""

    def test_generic_exception_in_temp_dir(self):
        model = ClaudeCodeModel("sonnet")
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {"type": "object", "properties": {"x": {"type": "string"}}}
            settings: ClaudeCodeSettings = {"__temp_json_dir": tmpdir}

            # Patch read_structure_from_filesystem to raise a generic exception
            with mock.patch(
                "pydantic_ai_claude_code.model.read_structure_from_filesystem",
                side_effect=TypeError("unexpected error"),
            ):
                data, error = model._read_structured_output_file(
                    "/nonexistent.json", schema, settings
                )

            assert data is None
            assert "unexpected error" in error
