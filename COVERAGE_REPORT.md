# Coverage Analysis Report

**Date**: 2026-02-17
**Python**: 3.10.19
**Test results**: 685 passed, 49 skipped, 0 failed

## Summary

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Overall coverage | 73% | 82% | +9% |
| Total statements | 3,596 | 3,596 | 0 |
| Missed statements | 975 | 656 | -319 |
| Tests passing | 550 | 685 | +135 |

## Per-Module Coverage

### High Coverage (95-100%) — No Action Needed

| Module | Coverage | Notes |
|--------|----------|-------|
| `_sdk/` (all) | 100% | SDK types, errors |
| `_utils/` (all) | 96-100% | File/JSON/type utils |
| `core/` (all) | 89-100% | Retry, OAuth, sandbox, debug |
| `structured/` (all) | 95-100% | Converter, file handler, function selector |
| `tools/` (all) | 97-100% | CC tools, MCP |
| `types.py` | 100% | Type definitions |
| `provider.py` | 100% | Provider registration |
| `registration.py` | 100% | Model registration |
| `streaming.py` | 100% | Streaming utilities |
| `temp_path_utils.py` | 100% | Temp path helpers |
| `provider_presets.py` | 97% | Provider presets |
| `streamed_response.py` | 96% | Streamed response |
| `cc_tools.py` | 97% | Claude Code tools |
| `messages.py` | 99% | Message conversion |
| `response_utils.py` | 100% | Response utilities |
| `sdk_adapter.py` | 96% | SDK adapter |
| `model.py` | 95% | Core model (was 55%) |
| `transport/sdk_transport.py` | 90% | CLI transport |

### Medium Coverage — Acceptable with Justification

| Module | Coverage | Missed Lines | Justification |
|--------|----------|--------------|---------------|
| `model.py` | 95% | 26 lines | Remaining: `request_stream()` body (L645-685) requires live CLI streaming infrastructure; inner retry loops in `_handle_argument_collection` for the structured-file-found-then-validate path (L1056-1068). |
| `transport/sdk_transport.py` | 90% | 20 lines | Remaining: `execute()` retry loop paths (L155-182) require real subprocess orchestration; timeout/kill paths (L337-338, 391-392) covered by mock but coroutine warning. |
| `utils.py` | 91% | 46 lines | Remaining: OAuth flow paths, real subprocess execution, interactive auth prompts. All require live Claude CLI. |
| `structure_converter.py` | 86% | 45 lines | Edge cases in schema conversion (nested anyOf, complex discriminated unions, unusual JSON Schema patterns). Would need very specific schema inputs. |
| `sdk_original_files/__init__.py` | 89% | 2 lines | The `get_credentials_from_sdk()` function (L54-61) tries to import non-existent `claude_agent_sdk` package. **Dead code** — see recommendations. |

### Low Coverage — Legacy/Dead Code

| Module | Coverage | Missed Lines | Assessment |
|--------|----------|--------------|------------|
| `utils_legacy.py` | 11% | 465 lines | **Legacy module.** Only 3 functions are actually imported: `resolve_claude_cli_path`, `build_claude_command`, `create_subprocess_async`. The other ~465 lines are duplicated/superseded by `utils.py`. |
| `structured_output.py` | 70% | 11 lines | **Dead code.** Not imported by any module. Contains `StructuredOutputHandler` class that wraps `run_claude_async`. The `handle()` method (L57-91) is untestable without the import chain. |

## What Can't Be Tested (Without Live CLI)

These code paths require a running Claude CLI binary and are appropriately skipped:

1. **`model.py:request_stream()`** (L645-685) — Requires real CLI process for stream-json events
2. **`utils.py` subprocess paths** (L249-257, 690-696, etc.) — Real `asyncio.create_subprocess_exec` with Claude binary
3. **`transport/sdk_transport.py` execute retry loops** (L155-182) — Full retry orchestration with real subprocesses
4. **Integration tests** (49 skipped) — All marked with `@pytest.mark.skipif` for missing Claude CLI

## Recommendations

### 1. Remove Dead Code (structured_output.py, sdk_original_files/)

**`structured_output.py`**: Not imported by any module. The `StructuredOutputHandler` class is superseded by the structured output handling built into `model.py` and `structured/`. **Recommend deletion.**

**`sdk_original_files/__init__.py`**: Not imported by any module. Contains SDK compatibility stubs for a `claude_agent_sdk` package that doesn't exist in this project. The `get_credentials_from_sdk()` function will always fail. **Recommend deletion.**

Impact: Removing these two files would eliminate 156 lines of dead code and increase overall coverage to ~85% without adding any tests.

### 2. Consolidate utils_legacy.py

Only 3 functions from `utils_legacy.py` are used:
- `resolve_claude_cli_path` — imported by `utils.py` and `transport/sdk_transport.py`
- `build_claude_command` — imported by `utils.py`
- `create_subprocess_async` — imported by `utils.py`

**Recommendation**: Move these 3 functions into `utils.py` directly and delete `utils_legacy.py`. This eliminates 521 statements of mostly-dead code and would push coverage to ~88%.

### 3. Coverage After Cleanup (Projected)

If dead/legacy code is removed:

| Metric | Current | After Cleanup |
|--------|---------|---------------|
| Total statements | 3,596 | ~2,919 (-677) |
| Missed statements | 656 | ~180 |
| Overall coverage | 82% | ~94% |

## New Test Files Created

1. **`tests/test_model_coverage.py`** (66 tests) — Covers model.py: `_build_options`, `_build_function_tools_prompt`, `_build_system_prompt_parts`, `_assemble_final_prompt`, `_handle_function_selection_followup`, `_build_argument_collection_instruction`, `_build_retry_prompt`, `_handle_structured_output_response`, `_convert_response`, `request_stream` errors, `_read_structured_output_file`, `_extract_json_robust`, `request()` with mocked CLI, `_handle_structured_follow_up`, `_handle_unstructured_follow_up`, `_handle_argument_collection`

2. **`tests/test_supporting_coverage.py`** (69 tests) — Covers response_utils.py, sdk_adapter.py (object-format paths), messages.py (binary content, extract text), transport/sdk_transport.py (classify error, process response, build command, execute command), sdk_original_files, structured_output.py
