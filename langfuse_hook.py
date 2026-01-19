#!/usr/bin/env python3
"""
Sends Claude Code traces to Langfuse and OTEL after each response.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

# Check if Langfuse is available
try:
    from langfuse import Langfuse
except ImportError:
    print("Error: langfuse package not installed. Run: pip install langfuse", file=sys.stderr)
    sys.exit(0)

# Check if OpenTelemetry is available (optional)
OTEL_ENABLED = False
OTEL_LOGS_ENABLED = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode
    OTEL_ENABLED = True

    # OTEL Logging SDK for AgentCore
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        OTEL_LOGS_ENABLED = True
    except ImportError:
        pass
except ImportError:
    pass

# CloudWatch Logs client (optional)
try:
    import boto3
    cloudwatch_logs = boto3.client('logs', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
    CLOUDWATCH_ENABLED = True
except ImportError:
    CLOUDWATCH_ENABLED = False
    cloudwatch_logs = None

# Configuration
LOG_FILE = Path.home() / ".claude" / "state" / "langfuse_hook.log"
STATE_FILE = Path.home() / ".claude" / "state" / "langfuse_state.json"
DEBUG = os.environ.get("CC_LANGFUSE_DEBUG", "").lower() == "true"
OTEL_TO_CLOUDWATCH = os.environ.get("OTEL_TO_CLOUDWATCH", "").lower() == "true"
CLOUDWATCH_LOG_GROUP = os.environ.get("CLOUDWATCH_LOG_GROUP", "/aws/claude-code/telemetry")
CLOUDWATCH_LOG_STREAM = os.environ.get("CLOUDWATCH_LOG_STREAM", "langfuse-hook-traces")
# AgentCore runtime log group (following AWS AgentCore external agent requirements)
CLOUDWATCH_SPANS_LOG_GROUP = "/aws/bedrock-agentcore/runtimes/claude-code"
CLOUDWATCH_SPANS_LOG_STREAM = "runtime-logs"
# AgentCore evaluation log group (where evaluation reads from)
CLOUDWATCH_EVAL_LOG_GROUP = "aws/spans"
CLOUDWATCH_EVAL_LOG_STREAM = "default"

# Global OTEL tracer
otel_tracer = None


def log(level: str, message: str) -> None:
    """Log a message to the log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} [{level}] {message}\n")


def debug(message: str) -> None:
    """Log a debug message (only if DEBUG is enabled)."""
    if DEBUG:
        log("DEBUG", message)


def load_state() -> dict:
    """Load the state file containing session tracking info."""
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def save_state(state: dict) -> None:
    """Save the state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_content(msg: dict) -> Any:
    """Extract content from a message."""
    if isinstance(msg, dict):
        if "message" in msg:
            return msg["message"].get("content")
        return msg.get("content")
    return None


def is_tool_result(msg: dict) -> bool:
    """Check if a message contains tool results."""
    content = get_content(msg)
    if isinstance(content, list):
        return any(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in content
        )
    return False


def get_tool_calls(msg: dict) -> list:
    """Extract tool use blocks from a message."""
    content = get_content(msg)
    if isinstance(content, list):
        return [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"
        ]
    return []


def get_text_content(msg: dict) -> str:
    """Extract text content from a message."""
    content = get_content(msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return ""


def merge_assistant_parts(parts: list) -> dict:
    """Merge multiple assistant message parts into one."""
    if not parts:
        return {}

    merged_content = []
    for part in parts:
        content = get_content(part)
        if isinstance(content, list):
            merged_content.extend(content)
        elif content:
            merged_content.append({"type": "text", "text": str(content)})

    # Use the structure from the first part
    result = parts[0].copy()
    if "message" in result:
        result["message"] = result["message"].copy()
        result["message"]["content"] = merged_content
    else:
        result["content"] = merged_content

    return result


def init_otel(session_id: str) -> None:
    """Initialize OpenTelemetry tracer with AgentCore resource attributes."""
    global otel_tracer

    if not OTEL_ENABLED or not OTEL_TO_CLOUDWATCH:
        return

    try:
        # Get OTEL endpoint from environment or use default
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

        # Create resource with AgentCore semantic conventions
        resource = Resource.create({
            "service.name": "claude-code",
            "service.namespace": "aws.bedrock.agentcore",
            "service.version": "2.1.12",
            "gen_ai.system": "anthropic.claude",
            "gen_ai.request.model": "claude-sonnet-4-5",
            "ai.agent.name": "claude-code",
            "ai.agent.type": "autonomous",
            "telemetry.sdk.name": "opentelemetry-python",
            "telemetry.sdk.language": "python",
            "cloud.provider": "aws",
            "session.id": session_id,
            "source": "langfuse-hook",
        })

        # Initialize tracer provider
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(trace_provider)
        otel_tracer = trace.get_tracer("claude-code.langfuse-hook", "1.0.0")

        debug(f"OTEL initialized with endpoint: {otlp_endpoint}")
    except Exception as e:
        log("WARN", f"Failed to initialize OTEL: {e}")


def emit_cloudwatch_log(
    session_id: str,
    turn_num: int,
    user_text: str,
    final_output: str,
    model: str,
    all_tool_calls: list,
) -> None:
    """Emit structured log directly to CloudWatch Logs."""
    if not CLOUDWATCH_ENABLED or not cloudwatch_logs or not OTEL_TO_CLOUDWATCH:
        return

    try:
        import time

        # Extract trace and span IDs from current OTEL span context
        trace_id = ""
        span_id = ""
        if OTEL_ENABLED:
            try:
                from opentelemetry import trace as otel_trace
                current_span = otel_trace.get_current_span()
                if current_span:
                    span_context = current_span.get_span_context()
                    if span_context.is_valid:
                        # Format as hex strings (32 chars for trace, 16 chars for span)
                        trace_id = format(span_context.trace_id, '032x')
                        span_id = format(span_context.span_id, '016x')
                        debug(f"Extracted trace_id={trace_id}, span_id={span_id}")
            except Exception as e:
                log("WARN", f"Failed to extract trace/span IDs: {e}")

        # Create proper OTEL LogRecord format for AgentCore evaluation
        otel_log = {
            "resource": {
                "attributes": {
                    "service.name": "claude-code",
                    "service.namespace": "aws.bedrock.agentcore",
                    "aws.log.group.names": "/aws/bedrock-agentcore/runtimes/claude-code",
                    "cloud.provider": "aws",
                    "cloud.resource_id": "claude-code-agent",
                    "cloud.region": "us-east-1",
                    "gen_ai.system": "anthropic.claude",
                    "ai.agent.name": "claude-code",
                    "ai.agent.type": "autonomous",
                    "telemetry.sdk.name": "opentelemetry",
                    "telemetry.sdk.language": "python",
                    "aws.service.type": "gen_ai_agent",
                }
            },
            "scope": {
                "name": "claude-code.langfuse-hook",
                "version": "1.0.0"
            },
            "timeUnixNano": int(time.time() * 1_000_000_000),
            "observedTimeUnixNano": int(time.time() * 1_000_000_000),
            "severityNumber": 9,  # INFO
            "severityText": "INFO",
            "body": f"Claude Code conversation turn - session: {session_id}, turn: {turn_num}, tools: {len(all_tool_calls)}",
            "attributes": {
                "session.id": session_id,
                "turn.number": turn_num,
                "model": model,
                "tool.count": len(all_tool_calls),
                "event.type": "conversation_turn",
                "otelServiceName": "claude-code",
                "user.message.length": len(user_text),
                "assistant.response.length": len(final_output),
            },
            "traceId": trace_id,
            "spanId": span_id
        }

        # Also create legacy format for telemetry log group
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "langfuse-hook",
            "event_type": "conversation_turn",
            "session_id": session_id,
            "turn_number": turn_num,
            "model": model,
            "user_message": {
                "content_preview": user_text[:200],
                "content_length": len(user_text),
            },
            "assistant_response": {
                "content_preview": final_output[:200],
                "content_length": len(final_output),
            },
            "tool_calls": [
                {
                    "name": tc["name"],
                    "tool_id": tc["id"],
                    "has_output": tc["output"] is not None,
                    "input_preview": str(tc["input"])[:100],
                }
                for tc in all_tool_calls
            ],
            "tool_count": len(all_tool_calls),
        }

        # Ensure log stream exists
        try:
            cloudwatch_logs.create_log_stream(
                logGroupName=CLOUDWATCH_LOG_GROUP,
                logStreamName=CLOUDWATCH_LOG_STREAM
            )
        except cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
            pass  # Stream already exists
        except Exception:
            pass  # Log group might not exist, will fail on put_log_events

        # Put log event to primary log group
        log_event = {
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
            'message': json.dumps(log_entry, indent=2)
        }

        cloudwatch_logs.put_log_events(
            logGroupName=CLOUDWATCH_LOG_GROUP,
            logStreamName=CLOUDWATCH_LOG_STREAM,
            logEvents=[log_event]
        )

        # Also write to AgentCore Evaluation spans log group
        try:
            # Ensure spans log stream exists
            try:
                cloudwatch_logs.create_log_stream(
                    logGroupName=CLOUDWATCH_SPANS_LOG_GROUP,
                    logStreamName=CLOUDWATCH_SPANS_LOG_STREAM
                )
            except cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
                pass
            except cloudwatch_logs.exceptions.ResourceNotFoundException:
                # Create log group if it doesn't exist
                try:
                    cloudwatch_logs.create_log_group(logGroupName=CLOUDWATCH_SPANS_LOG_GROUP)
                    cloudwatch_logs.create_log_stream(
                        logGroupName=CLOUDWATCH_SPANS_LOG_GROUP,
                        logStreamName=CLOUDWATCH_SPANS_LOG_STREAM
                    )
                except Exception:
                    pass

            # Write OTEL LogRecord format to AgentCore runtime log group
            otel_event = {
                'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                'message': json.dumps(otel_log)
            }
            cloudwatch_logs.put_log_events(
                logGroupName=CLOUDWATCH_SPANS_LOG_GROUP,
                logStreamName=CLOUDWATCH_SPANS_LOG_STREAM,
                logEvents=[otel_event]
            )

            # Also write to aws/spans log group (for AgentCore evaluation)
            try:
                cloudwatch_logs.create_log_stream(
                    logGroupName=CLOUDWATCH_EVAL_LOG_GROUP,
                    logStreamName=CLOUDWATCH_EVAL_LOG_STREAM
                )
            except cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
                pass
            except Exception:
                pass  # Log group or stream may not exist

            cloudwatch_logs.put_log_events(
                logGroupName=CLOUDWATCH_EVAL_LOG_GROUP,
                logStreamName=CLOUDWATCH_EVAL_LOG_STREAM,
                logEvents=[otel_event]
            )

            debug(f"Emitted CloudWatch log to telemetry, spans, and eval log groups for turn {turn_num}")
        except Exception as spans_err:
            log("WARN", f"Failed to write to spans log group: {spans_err}")
            debug(f"Emitted CloudWatch log to telemetry only for turn {turn_num}")

    except Exception as e:
        log("WARN", f"Failed to emit CloudWatch log: {e}")


def serialize_otel_span_to_cloudwatch(
    span_name: str,
    span_kind: int,
    start_time_ns: int,
    end_time_ns: int,
    trace_id: str,
    span_id: str,
    parent_span_id: str,
    attributes: dict,
    events: list,
    status_code: str = "OK",
) -> dict:
    """Serialize OTEL span to CloudWatch format for Agentcore Evaluation."""
    span_data = {
        "resource": {
            "attributes": {
                "service.name": "claude-code",
                "service.namespace": "aws.bedrock.agentcore",
                "cloud.provider": "aws",
                "cloud.region": "us-east-1",
                "aws.service.type": "gen_ai_agent",
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
            }
        },
        "scope": {
            "name": "claude-code.langfuse-hook",
            "version": "1.0.0"
        },
        "name": span_name,
        "kind": span_kind,  # 1=INTERNAL, 2=SERVER, 3=CLIENT
        "startTimeUnixNano": start_time_ns,
        "endTimeUnixNano": end_time_ns,
        "traceId": trace_id,
        "spanId": span_id,
        "status": {"code": status_code},
        "attributes": attributes
    }

    if parent_span_id:
        span_data["parentSpanId"] = parent_span_id

    if events:
        span_data["events"] = events

    return span_data


def emit_otel_trace(
    session_id: str,
    turn_num: int,
    user_text: str,
    final_output: str,
    model: str,
    all_tool_calls: list,
) -> None:
    """Emit OTEL trace to CloudWatch via OTEL collector and as spans to CloudWatch logs."""
    if not otel_tracer:
        return

    try:
        import time

        # Record start time for the turn span
        turn_start_ns = time.time_ns()

        # Create a span for the turn
        with otel_tracer.start_as_current_span(
            f"Turn-{turn_num}",
            attributes={
                "ai.operation.name": "conversation_turn",
                "ai.agent.turn_number": turn_num,
                "session.id": session_id,
                "gen_ai.request.model": model,
                "gen_ai.response.model": model,
                "gen_ai.system": "anthropic.claude",
                "gen_ai.operation.name": "chat",
                "tool.count": len(all_tool_calls),
                "user.input.length": len(user_text),
                "assistant.output.length": len(final_output),
            }
        ) as turn_span:
            # Add event for user message
            turn_span.add_event(
                "user_message",
                attributes={
                    "event.type": "user_message",
                    "content.preview": user_text[:200],  # Truncate
                    "content.length": len(user_text),
                }
            )

            # Add event for assistant response
            turn_span.add_event(
                "assistant_response",
                attributes={
                    "event.type": "assistant_response",
                    "model": model,
                    "content.preview": final_output[:200],  # Truncate
                    "content.length": len(final_output),
                    "tool_count": len(all_tool_calls),
                }
            )

            # Create child spans for each tool call
            for idx, tool_call in enumerate(all_tool_calls):
                with otel_tracer.start_as_current_span(
                    f"Tool-{tool_call['name']}",
                    attributes={
                        "tool.name": tool_call["name"],
                        "tool.id": tool_call["id"],
                        "tool.index": idx,
                        "tool.has_output": tool_call["output"] is not None,
                        "tool.input.preview": str(tool_call["input"])[:200],  # Truncate
                    }
                ) as tool_span:
                    # Add event for tool execution
                    tool_span.add_event(
                        "tool_executed",
                        attributes={
                            "event.type": "tool_execution",
                            "tool_name": tool_call["name"],
                        }
                    )

            # Set span status to OK
            turn_span.set_status(Status(StatusCode.OK))

            # Capture span data before context exits
            turn_end_ns = time.time_ns()
            turn_span_context = turn_span.get_span_context()
            turn_trace_id = format(turn_span_context.trace_id, '032x')
            turn_span_id = format(turn_span_context.span_id, '016x')

            # Serialize turn span for CloudWatch (for Agentcore Evaluation)
            if CLOUDWATCH_ENABLED and cloudwatch_logs and OTEL_TO_CLOUDWATCH:
                try:
                    # Create span attributes with gen_ai semantic conventions
                    turn_attributes = {
                        "ai.operation.name": "conversation_turn",
                        "ai.agent.turn_number": turn_num,
                        "session.id": session_id,
                        "gen_ai.request.model": model,
                        "gen_ai.response.model": model,
                        "gen_ai.system": "anthropic.claude",
                        "gen_ai.operation.name": "chat",
                        "tool.count": len(all_tool_calls),
                        "user.input.length": len(user_text),
                        "assistant.output.length": len(final_output),
                    }

                    # Create events array
                    turn_events = [
                        {
                            "name": "user_message",
                            "timeUnixNano": turn_start_ns,
                            "attributes": {
                                "event.type": "user_message",
                                "content.preview": user_text[:200],
                                "content.length": len(user_text),
                            }
                        },
                        {
                            "name": "assistant_response",
                            "timeUnixNano": turn_end_ns - 1000000,  # Slightly before end
                            "attributes": {
                                "event.type": "assistant_response",
                                "model": model,
                                "content.preview": final_output[:200],
                                "content.length": len(final_output),
                                "tool_count": len(all_tool_calls),
                            }
                        }
                    ]

                    # Serialize turn span
                    turn_span_data = serialize_otel_span_to_cloudwatch(
                        span_name=f"Turn-{turn_num}",
                        span_kind=2,  # SERVER
                        start_time_ns=turn_start_ns,
                        end_time_ns=turn_end_ns,
                        trace_id=turn_trace_id,
                        span_id=turn_span_id,
                        parent_span_id=None,  # Top-level span
                        attributes=turn_attributes,
                        events=turn_events,
                        status_code="OK"
                    )

                    # Write turn span to CloudWatch
                    cloudwatch_logs.put_log_events(
                        logGroupName="/aws/bedrock-agentcore/runtimes/claude-code",
                        logStreamName="runtime-logs",
                        logEvents=[{
                            'timestamp': int(time.time() * 1000),
                            'message': json.dumps(turn_span_data)
                        }]
                    )

                    debug(f"Wrote turn span to CloudWatch for evaluation")

                except Exception as cw_err:
                    log("WARN", f"Failed to write turn span to CloudWatch: {cw_err}")

            # Also emit CloudWatch log (legacy format, for backwards compatibility)
            emit_cloudwatch_log(session_id, turn_num, user_text, final_output, model, all_tool_calls)

        debug(f"Emitted OTEL trace for turn {turn_num} with {len(all_tool_calls)} tools")
    except Exception as e:
        log("WARN", f"Failed to emit OTEL trace: {e}")
        import traceback
        debug(traceback.format_exc())


def find_latest_transcript() -> Optional[Tuple[str, Path]]:
    """Find the most recently modified transcript file.

    Claude Code stores transcripts as *.jsonl files directly in the project directory.
    Main conversation files have UUID names, agent files have agent-*.jsonl names.
    The session ID is stored inside each JSON line.
    """
    projects_dir = Path.home() / ".claude" / "projects"

    if not projects_dir.exists():
        debug(f"Projects directory not found: {projects_dir}")
        return None

    latest_file = None
    latest_mtime = 0

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        # Look for all .jsonl files directly in the project directory
        for transcript_file in project_dir.glob("*.jsonl"):
            mtime = transcript_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = transcript_file

    if latest_file:
        # Extract session ID from the first line of the file
        try:
            first_line = latest_file.read_text().split("\n")[0]
            first_msg = json.loads(first_line)
            session_id = first_msg.get("sessionId", latest_file.stem)
            debug(f"Found transcript: {latest_file}, session: {session_id}")
            return (session_id, latest_file)
        except (json.JSONDecodeError, IOError, IndexError) as e:
            debug(f"Error reading transcript {latest_file}: {e}")
            return None

    debug("No transcript files found")
    return None


def create_trace(
    langfuse: Langfuse,
    session_id: str,
    turn_num: int,
    user_msg: dict,
    assistant_msgs: list,
    tool_results: list,
) -> None:
    """Create a Langfuse trace for a single turn using the new SDK API."""
    # Extract user text
    user_text = get_text_content(user_msg)

    # Extract final assistant text
    final_output = ""
    if assistant_msgs:
        final_output = get_text_content(assistant_msgs[-1])

    # Get model info from first assistant message
    model = "claude"
    if assistant_msgs and isinstance(assistant_msgs[0], dict) and "message" in assistant_msgs[0]:
        model = assistant_msgs[0]["message"].get("model", "claude")

    # Collect all tool calls and results
    all_tool_calls = []
    for assistant_msg in assistant_msgs:
        tool_calls = get_tool_calls(assistant_msg)
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", "")

            # Find matching tool result
            tool_output = None
            for tr in tool_results:
                tr_content = get_content(tr)
                if isinstance(tr_content, list):
                    for item in tr_content:
                        if isinstance(item, dict) and item.get("tool_use_id") == tool_id:
                            tool_output = item.get("content")
                            break

            all_tool_calls.append({
                "name": tool_name,
                "input": tool_input,
                "output": tool_output,
                "id": tool_id,
            })

    # Create trace using the new API with context managers
    with langfuse.start_as_current_span(
        name=f"Turn {turn_num}",
        input={"role": "user", "content": user_text},
        metadata={
            "source": "claude-code",
            "turn_number": turn_num,
            "session_id": session_id,
        },
    ) as trace_span:
        # Create generation for the LLM response
        with langfuse.start_as_current_observation(
            name="Claude Response",
            as_type="generation",
            model=model,
            input={"role": "user", "content": user_text},
            output={"role": "assistant", "content": final_output},
            metadata={
                "tool_count": len(all_tool_calls),
            },
        ) as generation:
            pass  # Generation is auto-completed when exiting context

        # Create spans for tool calls
        for tool_call in all_tool_calls:
            with langfuse.start_as_current_span(
                name=f"Tool: {tool_call['name']}",
                input=tool_call["input"],
                metadata={
                    "tool_name": tool_call["name"],
                    "tool_id": tool_call["id"],
                },
            ) as tool_span:
                tool_span.update(output=tool_call["output"])
            debug(f"Created span for tool: {tool_call['name']}")

        # Update trace with output
        trace_span.update(output={"role": "assistant", "content": final_output})

    debug(f"Created Langfuse trace for turn {turn_num}")

    # Also emit OTEL trace if enabled (CloudWatch logging happens inside this function)
    emit_otel_trace(session_id, turn_num, user_text, final_output, model, all_tool_calls)

    # Note: emit_cloudwatch_log is now called from WITHIN emit_otel_trace
    # to capture trace/span IDs from the active OTEL span context


def process_transcript(langfuse: Langfuse, session_id: str, transcript_file: Path, state: dict) -> int:
    """Process a transcript file and create traces for new turns."""
    # Get previous state for this session
    session_state = state.get(session_id, {})
    last_line = session_state.get("last_line", 0)
    turn_count = session_state.get("turn_count", 0)

    # Read transcript
    lines = transcript_file.read_text().strip().split("\n")
    total_lines = len(lines)

    if last_line >= total_lines:
        debug(f"No new lines to process (last: {last_line}, total: {total_lines})")
        return 0

    # Parse new messages
    new_messages = []
    for i in range(last_line, total_lines):
        try:
            msg = json.loads(lines[i])
            new_messages.append(msg)
        except json.JSONDecodeError:
            continue

    if not new_messages:
        return 0

    debug(f"Processing {len(new_messages)} new messages")

    # Group messages into turns (user -> assistant(s) -> tool_results)
    turns = 0
    current_user = None
    current_assistants = []
    current_assistant_parts = []
    current_msg_id = None
    current_tool_results = []

    for msg in new_messages:
        role = msg.get("type") or (msg.get("message", {}).get("role"))

        if role == "user":
            # Check if this is a tool result
            if is_tool_result(msg):
                current_tool_results.append(msg)
                continue

            # New user message - finalize previous turn
            if current_msg_id and current_assistant_parts:
                merged = merge_assistant_parts(current_assistant_parts)
                current_assistants.append(merged)
                current_assistant_parts = []
                current_msg_id = None

            if current_user and current_assistants:
                turns += 1
                turn_num = turn_count + turns
                create_trace(langfuse, session_id, turn_num, current_user, current_assistants, current_tool_results)

            # Start new turn
            current_user = msg
            current_assistants = []
            current_assistant_parts = []
            current_msg_id = None
            current_tool_results = []

        elif role == "assistant":
            msg_id = None
            if isinstance(msg, dict) and "message" in msg:
                msg_id = msg["message"].get("id")

            if not msg_id:
                # No message ID, treat as continuation
                current_assistant_parts.append(msg)
            elif msg_id == current_msg_id:
                # Same message ID, add to current parts
                current_assistant_parts.append(msg)
            else:
                # New message ID - finalize previous message
                if current_msg_id and current_assistant_parts:
                    merged = merge_assistant_parts(current_assistant_parts)
                    current_assistants.append(merged)

                # Start new assistant message
                current_msg_id = msg_id
                current_assistant_parts = [msg]

    # Process final turn
    if current_msg_id and current_assistant_parts:
        merged = merge_assistant_parts(current_assistant_parts)
        current_assistants.append(merged)

    if current_user and current_assistants:
        turns += 1
        turn_num = turn_count + turns
        create_trace(langfuse, session_id, turn_num, current_user, current_assistants, current_tool_results)

    # Update state
    state[session_id] = {
        "last_line": total_lines,
        "turn_count": turn_count + turns,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)

    return turns


def main():
    script_start = datetime.now()
    debug("Hook started")

    # Check if tracing is enabled
    if os.environ.get("TRACE_TO_LANGFUSE", "").lower() != "true":
        debug("Tracing disabled (TRACE_TO_LANGFUSE != true)")
        sys.exit(0)

    # Check for required environment variables
    public_key = os.environ.get("CC_LANGFUSE_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("CC_LANGFUSE_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("CC_LANGFUSE_HOST") or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        log("ERROR", "Langfuse API keys not set (CC_LANGFUSE_PUBLIC_KEY / CC_LANGFUSE_SECRET_KEY)")
        sys.exit(0)

    # Initialize Langfuse client
    try:
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
    except Exception as e:
        log("ERROR", f"Failed to initialize Langfuse client: {e}")
        sys.exit(0)

    # Load state
    state = load_state()

    # Find the most recently modified transcript
    result = find_latest_transcript()
    if not result:
        debug("No transcript file found")
        sys.exit(0)

    session_id, transcript_file = result

    if not transcript_file:
        debug("No transcript file found")
        sys.exit(0)

    debug(f"Processing session: {session_id}")

    # Initialize OTEL if enabled
    init_otel(session_id)

    # Process the transcript
    try:
        turns = process_transcript(langfuse, session_id, transcript_file, state)

        # Flush to ensure all data is sent
        langfuse.flush()

        # Flush OTEL tracer if enabled
        if otel_tracer:
            trace.get_tracer_provider().force_flush()
            debug("Flushed OTEL traces")

        # Log execution time
        duration = (datetime.now() - script_start).total_seconds()
        destinations = ["Langfuse"]
        if OTEL_TO_CLOUDWATCH and OTEL_ENABLED:
            destinations.append("OTEL")
        if OTEL_TO_CLOUDWATCH and CLOUDWATCH_ENABLED:
            destinations.append("CloudWatch")
        log_msg = f"Processed {turns} turns in {duration:.1f}s ({', '.join(destinations)})"
        log("INFO", log_msg)

        if duration > 180:
            log("WARN", f"Hook took {duration:.1f}s (>3min), consider optimizing")

    except Exception as e:
        log("ERROR", f"Failed to process transcript: {e}")
        import traceback
        debug(traceback.format_exc())
    finally:
        langfuse.shutdown()

        # Shutdown OTEL tracer
        if otel_tracer:
            trace.get_tracer_provider().shutdown()

    sys.exit(0)


if __name__ == "__main__":
    main()
